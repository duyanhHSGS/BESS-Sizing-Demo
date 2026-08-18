from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from bess.agents.ppo_agent import PPOAgent, RolloutBuffer, _gae_advantages
from bess.training.runners.train_ppo_dataset import (
    _action_mismatch_penalty_vnd,
    _behavior_clone_actor,
    _behavior_clone_critic,
    _champion_curve_point,
    _initialize_champion,
    _oracle_dispatch_wear_cost_vnd,
    _oracle_teacher_action,
    _resolve_challenger,
    _save_accepted_champion,
)


def _make_buffer(agent: PPOAgent, *, size: int = 8, seed: int = 0) -> RolloutBuffer:
    rng = np.random.default_rng(seed)
    obs_dim = agent.net.actor[0].in_features
    buffer = RolloutBuffer(size, obs_dim)
    for index in range(size):
        obs = rng.normal(size=obs_dim).astype(np.float32)
        action, logp, latent, value = agent.act_with_latent(obs)
        buffer.add(
            obs,
            action,
            logp,
            float(index + 1) / 10.0,
            value,
            0.0,
            latent,
        )
    return buffer


def _assert_nested_equal(testcase: unittest.TestCase, left, right) -> None:
    testcase.assertEqual(type(left), type(right))
    if isinstance(left, torch.Tensor):
        testcase.assertTrue(torch.equal(left, right))
        return
    if isinstance(left, dict):
        testcase.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_nested_equal(testcase, left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        testcase.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(testcase, left_item, right_item)
        return
    testcase.assertEqual(left, right)


class PPOTrainingStateTests(unittest.TestCase):
    def test_explicit_restore_recovers_network_optimizer_and_collector_but_not_rng(self):
        agent = PPOAgent(obs_dim=4, seed=7, device="cpu", epochs=1, minibatch=4)

        # Give Adam real momentum/history so rollback tests more than just weights.
        agent.update(_make_buffer(agent, seed=1), 0.0)
        challenger_buffer = _make_buffer(agent, seed=2)
        champion_state = agent.snapshot_training_state()
        rng_before_update = copy.deepcopy(agent.rng.bit_generator.state)

        agent.update(challenger_buffer, 0.0)
        rng_after_update = copy.deepcopy(agent.rng.bit_generator.state)
        self.assertNotEqual(rng_before_update, rng_after_update)
        self.assertTrue(
            any(
                not torch.equal(value, champion_state["network"][key])
                for key, value in agent.net.state_dict().items()
            )
        )

        agent.restore_training_state(champion_state)

        _assert_nested_equal(self, agent.net.state_dict(), champion_state["network"])
        _assert_nested_equal(self, agent.opt.state_dict(), champion_state["optimizer"])
        for key, value in agent.collector_net.state_dict().items():
            self.assertTrue(torch.equal(value, champion_state["network"][key].cpu()))
        self.assertEqual(agent.rng.bit_generator.state, rng_after_update)


class PPOOracleTeacherTests(unittest.TestCase):
    def test_oracle_teacher_action_converts_outside_power_to_battery_side(self):
        cfg = SimpleNamespace(eta_dis=0.9, eta_ch=0.9, P_rated_nominal=450.0)
        discharge = _oracle_teacher_action(
            {"discharge": [90.0], "grid_charge": [0.0], "solar_charge": [0.0]},
            0,
            1,
            cfg,
        )
        charge = _oracle_teacher_action(
            {"discharge": [0.0], "grid_charge": [100.0], "solar_charge": [0.0]},
            0,
            1,
            cfg,
        )
        self.assertAlmostEqual(discharge, 100.0 / 450.0)
        self.assertAlmostEqual(charge, -90.0 / 450.0)

    def test_oracle_wear_score_counts_cached_charge_and_discharge(self):
        wear = _oracle_dispatch_wear_cost_vnd(
            [
                {
                    "discharge": [100.0, 0.0],
                    "grid_charge": [0.0, 80.0],
                    "solar_charge": [0.0, 20.0],
                }
            ],
            timestep_hours=0.25,
            wear_vnd_per_kwh=500.0,
        )
        self.assertAlmostEqual(wear, 25_000.0)

    def test_behavior_clone_actor_reduces_teacher_mse(self):
        agent = PPOAgent(7, seed=0, device="cpu")
        rng = np.random.default_rng(123)
        observations = rng.normal(size=(64, 7)).astype(np.float32)
        targets = np.tanh(observations[:, 0] * 0.5).astype(np.float32)
        stats = _behavior_clone_actor(agent, observations, targets, seed=0)
        self.assertLess(stats["final_mse"], stats["initial_mse"])
        self.assertEqual(stats["samples"], 64)

    def test_behavior_clone_critic_reduces_oracle_return_mse(self):
        agent = PPOAgent(7, seed=0, device="cpu")
        rng = np.random.default_rng(321)
        observations = rng.normal(size=(64, 7)).astype(np.float32)
        rewards = (0.05 * observations[:, 0] - 0.02 * observations[:, 1]).astype(np.float32)
        stats = _behavior_clone_critic(
            agent,
            observations,
            rewards,
            gamma=0.999,
            seed=0,
        )
        self.assertLess(stats["critic_final_mse"], stats["critic_initial_mse"])
        self.assertEqual(stats["critic_epochs_completed"], 100)


class PPOActionMismatchPenaltyTests(unittest.TestCase):
    def test_penalty_matches_phantom_wear_of_rejected_requested_energy(self):
        transition = SimpleNamespace(
            native_results=(
                SimpleNamespace(
                    requested_battery_kw=-450.0,
                    bess=SimpleNamespace(
                        physics=SimpleNamespace(final_battery_kw=0.0),
                    ),
                ),
                SimpleNamespace(
                    requested_battery_kw=200.0,
                    bess=SimpleNamespace(
                        physics=SimpleNamespace(final_battery_kw=100.0),
                    ),
                ),
            )
        )
        penalty = _action_mismatch_penalty_vnd(
            transition,
            timestep_hours=0.25,
            wear_vnd_per_kwh=500.0,
        )
        self.assertEqual(penalty, (450.0 + 100.0) * 0.25 * 500.0)

    def test_penalty_is_zero_when_physics_executes_the_requested_action(self):
        transition = SimpleNamespace(
            native_results=(
                SimpleNamespace(
                    requested_battery_kw=123.0,
                    bess=SimpleNamespace(
                        physics=SimpleNamespace(final_battery_kw=123.0),
                    ),
                ),
            )
        )
        self.assertEqual(
            _action_mismatch_penalty_vnd(
                transition,
                timestep_hours=0.25,
                wear_vnd_per_kwh=500.0,
            ),
            0.0,
        )


class PPOGAETests(unittest.TestCase):
    def test_done_boundary_blocks_advantage_leak_from_next_episode(self):
        advantages = _gae_advantages(
            np.array([0.0, 1.0, 100.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            last_val=0.0,
            gamma=1.0,
            lam=1.0,
        )
        np.testing.assert_allclose(advantages, [1.0, 1.0, 100.0])

    def test_terminal_transition_never_bootstraps_last_value(self):
        advantages = _gae_advantages(
            np.array([5.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            last_val=999.0,
            gamma=1.0,
            lam=1.0,
        )
        np.testing.assert_allclose(advantages, [3.0])


class PPODeterminismTests(unittest.TestCase):
    def test_same_seed_replays_the_same_collection_and_update(self):
        def run_once(seed: int):
            agent = PPOAgent(obs_dim=4, seed=seed, device="cpu", epochs=2, minibatch=4)
            buffer = _make_buffer(agent, size=8, seed=seed + 100)
            stats = agent.update(buffer, 0.0)
            state = {
                key: value.detach().clone()
                for key, value in agent.net.state_dict().items()
            }
            return state, stats

        first_state, first_stats = run_once(23)
        second_state, second_stats = run_once(23)

        for key in first_state:
            self.assertTrue(torch.equal(first_state[key], second_state[key]), key)
        self.assertEqual(first_stats, second_stats)


class PPOChampionSelectionTests(unittest.TestCase):
    def test_accept_improvement_and_reject_same_or_worse(self):
        improved_cost, accepted = _resolve_challenger(
            champion_cost=100.0,
            candidate_cost=90.0,
        )
        self.assertTrue(accepted)
        self.assertEqual(improved_cost, 90.0)

        for candidate_cost in (90.0, 95.0):
            with self.subTest(candidate_cost=candidate_cost):
                restored_cost, accepted = _resolve_challenger(
                    champion_cost=90.0,
                    candidate_cost=candidate_cost,
                )
                self.assertFalse(accepted)
                self.assertEqual(restored_cost, 90.0)

    def test_champion_cost_is_monotonic_for_fake_candidates(self):
        champion_cost = 100.0
        accepted_sequence = [champion_cost]
        for candidate_cost in (80.0, 95.0, 70.0, 120.0):
            champion_cost, _accepted = _resolve_challenger(
                champion_cost,
                candidate_cost,
            )
            accepted_sequence.append(champion_cost)

        self.assertEqual(accepted_sequence, [100.0, 80.0, 80.0, 70.0, 70.0])

    def test_initial_champion_is_validated_and_saved_before_any_challenger(self):
        events: list[str] = []

        class RecordingAgent:
            def save(self, path: Path) -> None:
                events.append("save")
                path.write_text("champion", encoding="utf-8")

        def validate() -> float:
            events.append("validate")
            return 100.0

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "policy_test.pt"
            champion_cost = _initialize_champion(RecordingAgent(), validate, checkpoint_path)
            self.assertTrue(checkpoint_path.exists())

        self.assertEqual(champion_cost, 100.0)
        self.assertEqual(events, ["validate", "save"])

    def test_rejected_challenger_never_overwrites_champion_checkpoint(self):
        class RecordingAgent:
            def __init__(self) -> None:
                self.saved_payload = "champion"

            def save(self, path: Path) -> None:
                path.write_text(self.saved_payload, encoding="utf-8")

        agent = RecordingAgent()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "policy_test.pt"
            agent.save(checkpoint_path)
            agent.saved_payload = "rejected challenger"
            wrote = _save_accepted_champion(agent, checkpoint_path, accepted=False)
            self.assertFalse(wrote)
            self.assertEqual(checkpoint_path.read_text(encoding="utf-8"), "champion")

            agent.saved_payload = "accepted challenger"
            wrote = _save_accepted_champion(agent, checkpoint_path, accepted=True)
            self.assertTrue(wrote)
            self.assertEqual(
                checkpoint_path.read_text(encoding="utf-8"),
                "accepted challenger",
            )

    def test_curve_keeps_candidate_separate_from_compatibility_champion_cost(self):
        point = _champion_curve_point(
            steps=123,
            candidate_cost=120.0,
            champion_cost=90.0,
            accepted=False,
            val_base=150.0,
            val_oracle=75.0,
        )

        self.assertEqual(point["candidate_val_cost_vnd"], 120.0)
        self.assertEqual(point["champion_val_cost_vnd"], 90.0)
        self.assertEqual(point["val_cost_vnd"], 90.0)
        self.assertFalse(point["accepted"])
        self.assertAlmostEqual(point["saving_vs_nobess_pct"], 40.0)
        self.assertAlmostEqual(point["oracle_gap_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()
