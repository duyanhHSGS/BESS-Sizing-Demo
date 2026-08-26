from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from bess.agents.ppo_agent import (
    PPOAgent,
    RolloutBuffer,
    _critic_value_loss,
    _gae_advantages,
)
from bess.training.runners.train_ppo_dataset import (
    _action_mismatch_penalty_vnd,
    _behavior_clone_actor,
    _behavior_clone_critic,
    _champion_curve_point,
    _chronological_month_holdout_split,
    _initialize_champion,
    _midpoint_challenger_state,
    _oracle_dispatch_wear_cost_vnd,
    _oracle_teacher_action,
    _resolve_challenger,
    _restore_reanchor_state,
    _save_accepted_champion,
    _should_reanchor_rejected_candidate,
    _training_reference_power_kw,
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

    def test_hybrid_reanchor_restores_policy_keeps_live_critic_and_clears_adam(self):
        agent = PPOAgent(obs_dim=4, seed=17, device="cpu", epochs=1, minibatch=4)
        agent.update(_make_buffer(agent, seed=10), 0.0)
        champion_state = agent.snapshot_training_state()

        agent.update(_make_buffer(agent, seed=11), 0.0)
        live_critic = {
            key: value.detach().clone()
            for key, value in agent.net.critic.state_dict().items()
        }
        self.assertTrue(agent.opt.state)

        _restore_reanchor_state(
            agent,
            champion_state,
            preserve_critic=True,
            reset_optimizer=True,
        )

        champion_network = champion_state["network"]
        for key, value in agent.net.actor.state_dict().items():
            self.assertTrue(torch.equal(value, champion_network[f"actor.{key}"]))
        self.assertTrue(torch.equal(agent.net.log_std, champion_network["log_std"]))
        for key, value in agent.net.critic.state_dict().items():
            self.assertTrue(torch.equal(value, live_critic[key]))
        self.assertFalse(agent.opt.state)
        for key, value in agent.collector_net.state_dict().items():
            self.assertTrue(torch.equal(value, agent.net.state_dict()[key].cpu()))

    def test_midpoint_challenger_state_halves_floating_network_delta_and_clears_adam(self):
        champion_state = {
            "network": {
                "weight": torch.tensor([0.0, 2.0]),
                "counter": torch.tensor([1], dtype=torch.int64),
            },
            "optimizer": {
                "state": {1: {"step": torch.tensor(3.0)}},
                "param_groups": [{"lr": 1e-4, "params": [1, 2]}],
            },
        }
        candidate_state = {
            "network": {
                "weight": torch.tensor([2.0, 6.0]),
                "counter": torch.tensor([9], dtype=torch.int64),
            },
            "optimizer": {
                "state": {1: {"step": torch.tensor(9.0)}},
                "param_groups": [{"lr": 1e-4, "params": [1, 2]}],
            },
        }
        candidate_optimizer_before = copy.deepcopy(candidate_state["optimizer"])

        midpoint_state = _midpoint_challenger_state(
            champion_state,
            candidate_state,
        )

        self.assertTrue(
            torch.equal(midpoint_state["network"]["weight"], torch.tensor([1.0, 4.0]))
        )
        self.assertTrue(
            torch.equal(
                midpoint_state["network"]["counter"],
                candidate_state["network"]["counter"],
            )
        )
        self.assertEqual(midpoint_state["optimizer"]["state"], {})
        self.assertEqual(
            midpoint_state["optimizer"]["param_groups"],
            candidate_state["optimizer"]["param_groups"],
        )
        _assert_nested_equal(self, candidate_state["optimizer"], candidate_optimizer_before)

    def test_midpoint_fresh_adam_state_restores_and_repopulates_on_next_update(self):
        agent = PPOAgent(obs_dim=4, seed=23, device="cpu", epochs=1, minibatch=4)
        agent.update(_make_buffer(agent, seed=20), 0.0)
        champion_state = agent.snapshot_training_state()

        agent.update(_make_buffer(agent, seed=21), 0.0)
        candidate_state = agent.snapshot_training_state()
        self.assertTrue(candidate_state["optimizer"]["state"])

        midpoint_state = _midpoint_challenger_state(champion_state, candidate_state)
        agent.restore_training_state(midpoint_state)
        self.assertFalse(agent.opt.state)
        for key, value in agent.collector_net.state_dict().items():
            self.assertTrue(torch.equal(value, agent.net.state_dict()[key].cpu()))

        agent.update(_make_buffer(agent, seed=22), 0.0)
        self.assertTrue(agent.opt.state)

    def test_midpoint_fresh_adam_keeps_optimizer_hyperparameters(self):
        agent = PPOAgent(obs_dim=4, seed=29, device="cpu", epochs=1, minibatch=4)
        agent.update(_make_buffer(agent, seed=30), 0.0)
        champion_state = agent.snapshot_training_state()
        agent.update(_make_buffer(agent, seed=31), 0.0)
        candidate_state = agent.snapshot_training_state()

        midpoint_state = _midpoint_challenger_state(champion_state, candidate_state)

        self.assertEqual(midpoint_state["optimizer"]["state"], {})
        self.assertEqual(
            midpoint_state["optimizer"]["param_groups"],
            candidate_state["optimizer"]["param_groups"],
        )


class PPORealDataSplitTests(unittest.TestCase):
    def test_chronological_holdout_is_disjoint_and_keeps_whole_months(self):
        month_lengths = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30}
        days = [
            SimpleNamespace(date_iso=f"2026-{month:02d}-{day:02d}")
            for month, month_length in month_lengths.items()
            for day in range(1, month_length + 1)
        ]

        train, validation, test, ignored = _chronological_month_holdout_split(days, 2, 1)

        self.assertEqual(sorted({day.date_iso[:7] for day in train}), ["2026-01", "2026-02", "2026-03"])
        self.assertEqual(sorted({day.date_iso[:7] for day in validation}), ["2026-04", "2026-05"])
        self.assertEqual(sorted({day.date_iso[:7] for day in test}), ["2026-06"])
        self.assertEqual(ignored, [])
        self.assertTrue(set(map(id, train)).isdisjoint(map(id, validation)))
        self.assertTrue(set(map(id, train)).isdisjoint(map(id, test)))
        self.assertTrue(set(map(id, validation)).isdisjoint(map(id, test)))

    def test_partial_trailing_month_is_ignored_instead_of_becoming_fake_test_month(self):
        month_lengths = {1: 31, 2: 28, 3: 31}
        days = [
            SimpleNamespace(date_iso=f"2026-{month:02d}-{day:02d}")
            for month, month_length in month_lengths.items()
            for day in range(1, month_length + 1)
        ]
        days.extend(
            SimpleNamespace(date_iso=f"2026-04-{day:02d}")
            for day in range(1, 10)
        )

        train, validation, test, ignored = _chronological_month_holdout_split(days, 1, 1)

        self.assertEqual(sorted({day.date_iso[:7] for day in train}), ["2026-01"])
        self.assertEqual(sorted({day.date_iso[:7] for day in validation}), ["2026-02"])
        self.assertEqual(sorted({day.date_iso[:7] for day in test}), ["2026-03"])
        self.assertEqual(len(ignored), 9)
        self.assertEqual({day.date_iso[:7] for day in ignored}, {"2026-04"})

    def test_incomplete_internal_month_is_rejected(self):
        month_lengths = {1: 31, 2: 28, 3: 31, 4: 30}
        days = [
            SimpleNamespace(date_iso=f"2026-{month:02d}-{day:02d}")
            for month, month_length in month_lengths.items()
            for day in range(1, month_length + 1)
            if not (month == 2 and day == 15)
        ]

        with self.assertRaisesRegex(ValueError, "incomplete internal calendar month"):
            _chronological_month_holdout_split(days, 1, 1)

    def test_reference_power_uses_training_days_only(self):
        training_days = [
            SimpleNamespace(load=np.asarray([120.0, 740.0], dtype=np.float32)),
            SimpleNamespace(load=np.asarray([250.0, 910.0], dtype=np.float32)),
        ]
        future_holdout = SimpleNamespace(
            load=np.asarray([9999.0], dtype=np.float32)
        )

        self.assertEqual(_training_reference_power_kw(training_days), 1000.0)
        self.assertGreater(float(future_holdout.load.max()), 1000.0)


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

    def test_behavior_clone_critic_resets_returns_between_real_months(self):
        agent = PPOAgent(7, seed=0, device="cpu")
        observations = np.zeros((4, 7), dtype=np.float32)
        rewards = np.asarray([1.0, 1.0, 10.0, 10.0], dtype=np.float32)

        stats = _behavior_clone_critic(
            agent,
            observations,
            rewards,
            gamma=1.0,
            seed=0,
            max_epochs=0,
            episode_lengths=[2, 2],
        )

        # Month-local returns are [2, 1] and [20, 10]; no future-month reward leaks backward.
        self.assertAlmostEqual(stats["critic_target_mean"], 8.25)


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


class PPOCriticLossTests(unittest.TestCase):
    def test_smooth_l1_critic_loss_bounds_large_residual_gradient(self):
        prediction = torch.tensor([10.0], requires_grad=True)
        target = torch.tensor([0.0])

        loss = _critic_value_loss(prediction, target)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 9.5)
        self.assertAlmostEqual(float(prediction.grad.item()), 1.0)
        self.assertAlmostEqual(float(prediction.grad.item()), 1.0)


class PPORecurrentMemoryTests(unittest.TestCase):
    def test_recurrent_rollout_update_and_checkpoint_roundtrip(self):
        agent = PPOAgent(
            obs_dim=4,
            seed=7,
            device="cpu",
            hidden_size=8,
            recurrent_enabled=True,
            recurrent_sequence_length=4,
            epochs=1,
            minibatch=8,
        )
        buffer = RolloutBuffer(8, 4, recurrent_hidden_size=8)
        rng = np.random.default_rng(99)
        for index in range(8):
            obs = rng.normal(size=4).astype(np.float32)
            action, logp, latent, value = agent.act_with_latent(obs)
            actor_hidden, critic_hidden = agent.recurrent_rollout_inputs()
            buffer.add(
                obs,
                action,
                logp,
                float(index % 3) * 0.1,
                value,
                float(index == 7),
                latent=latent,
                actor_hidden=actor_hidden,
                critic_hidden=critic_hidden,
            )

        stats = agent.update(buffer, 0.0)
        self.assertEqual(stats["recurrent_sequence_length"], 4)
        self.assertEqual(stats["recurrent_chunk_count"], 2)
        self.assertTrue(np.isfinite(stats["value_loss"]))

        probe = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
        agent.reset_recurrent_state()
        first_action = agent.predict_action(probe)
        first_hidden = agent._actor_hidden.detach().clone()
        second_action = agent.predict_action(probe)
        second_hidden = agent._actor_hidden.detach().clone()
        self.assertFalse(torch.equal(first_hidden, second_hidden))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recurrent.pt"
            agent.save(path)
            loaded = PPOAgent(obs_dim=4, device="cpu", hidden_size=8)
            loaded.load(path)
            self.assertTrue(loaded.recurrent_enabled)
            self.assertEqual(loaded.recurrent_sequence_length, 4)
            loaded.reset_recurrent_state()
            self.assertAlmostEqual(first_action, loaded.predict_action(probe), places=6)
            self.assertAlmostEqual(second_action, loaded.predict_action(probe), places=6)

    def test_recurrent_behavior_clone_restores_lowest_cost_epoch(self):
        agent = PPOAgent(
            obs_dim=7,
            seed=3,
            device="cpu",
            hidden_size=8,
            recurrent_enabled=True,
            recurrent_sequence_length=4,
        )
        rng = np.random.default_rng(1234)
        observations = rng.normal(size=(12, 7)).astype(np.float32)
        targets = np.tanh(0.4 * observations[:, 0] - 0.2 * observations[:, 3]).astype(np.float32)
        scores = iter((30.0, 10.0, 20.0))
        scored_predictions: list[np.ndarray] = []

        def score_policy_cost() -> float:
            with torch.inference_mode():
                obs_t = torch.as_tensor(observations, dtype=torch.float32)
                mean, _, _ = agent.net.actor_sequence(obs_t.unsqueeze(0), None)
                prediction = torch.tanh(mean.squeeze(0)).squeeze(-1).cpu().numpy().copy()
            scored_predictions.append(prediction)
            return next(scores)

        stats = _behavior_clone_actor(
            agent,
            observations,
            targets,
            seed=3,
            max_epochs=3,
            target_mse=0.0,
            score_policy_cost=score_policy_cost,
        )

        with torch.inference_mode():
            obs_t = torch.as_tensor(observations, dtype=torch.float32)
            mean, _, _ = agent.net.actor_sequence(obs_t.unsqueeze(0), None)
            selected_prediction = torch.tanh(mean.squeeze(0)).squeeze(-1).cpu().numpy()

        self.assertEqual(stats["economic_best_epoch"], 2)
        self.assertEqual(stats["economic_best_validation_cost_vnd"], 10.0)
        np.testing.assert_allclose(selected_prediction, scored_predictions[1], atol=1e-6)


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

    def test_reanchor_gate_respects_enable_allow_and_patience_controls(self):
        common = {
            "allow_reset": True,
            "reset_patience": 1,
        }
        self.assertTrue(
            _should_reanchor_rejected_candidate(
                **common,
                resets_enabled=True,
                consecutive_rejections=1,
            )
        )
        self.assertFalse(
            _should_reanchor_rejected_candidate(
                **common,
                resets_enabled=False,
                consecutive_rejections=99,
            )
        )
        self.assertFalse(
            _should_reanchor_rejected_candidate(
                resets_enabled=True,
                allow_reset=False,
                consecutive_rejections=99,
                reset_patience=1,
            )
        )
        self.assertFalse(
            _should_reanchor_rejected_candidate(
                resets_enabled=True,
                allow_reset=True,
                consecutive_rejections=2,
                reset_patience=3,
            )
        )

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
