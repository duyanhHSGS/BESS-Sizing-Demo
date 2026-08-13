from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bess.agents.ppo_agent import PPOAgent, RolloutBuffer
from bess.training.runners.train_ppo_dataset import (
    _champion_curve_point,
    _initialize_champion,
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
    def test_rejected_challenger_restores_network_optimizer_and_collector_but_not_rng(self):
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

        restored_cost, accepted = _resolve_challenger(
            agent,
            champion_state,
            champion_cost=100.0,
            candidate_cost=120.0,
        )

        self.assertFalse(accepted)
        self.assertEqual(restored_cost, 100.0)
        _assert_nested_equal(self, agent.net.state_dict(), champion_state["network"])
        _assert_nested_equal(self, agent.opt.state_dict(), champion_state["optimizer"])
        for key, value in agent.collector_net.state_dict().items():
            self.assertTrue(torch.equal(value, champion_state["network"][key].cpu()))
        self.assertEqual(agent.rng.bit_generator.state, rng_after_update)


class PPOChampionSelectionTests(unittest.TestCase):
    def test_accept_improvement_and_reject_same_or_worse(self):
        agent = PPOAgent(obs_dim=3, seed=3, device="cpu")
        champion_state = agent.snapshot_training_state()

        improved_cost, accepted = _resolve_challenger(
            agent,
            champion_state,
            champion_cost=100.0,
            candidate_cost=90.0,
        )
        self.assertTrue(accepted)
        self.assertEqual(improved_cost, 90.0)

        for candidate_cost in (90.0, 95.0):
            with self.subTest(candidate_cost=candidate_cost):
                restored_cost, accepted = _resolve_challenger(
                    agent,
                    agent.snapshot_training_state(),
                    champion_cost=90.0,
                    candidate_cost=candidate_cost,
                )
                self.assertFalse(accepted)
                self.assertEqual(restored_cost, 90.0)

    def test_champion_cost_is_monotonic_for_fake_candidates(self):
        agent = PPOAgent(obs_dim=3, seed=4, device="cpu")
        champion_cost = 100.0
        accepted_sequence = [champion_cost]
        for candidate_cost in (80.0, 95.0, 70.0, 120.0):
            champion_cost, _accepted = _resolve_challenger(
                agent,
                agent.snapshot_training_state(),
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
