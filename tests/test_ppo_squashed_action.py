from __future__ import annotations

import math
import tempfile
import unittest

import numpy as np
import torch

from bess.agents.ppo_agent import (
    ActorCritic,
    PPOAgent,
    _mlp,
    _squashed_log_prob_from_latent,
)


class PPOSquashedActionTests(unittest.TestCase):
    def test_adaptive_head_does_not_advance_legacy_initialization_rng(self):
        torch.manual_seed(123)
        _mlp(7, 1, hidden=64)
        _mlp(7, 1, hidden=64)
        expected_rng = torch.random.get_rng_state().clone()

        torch.manual_seed(123)
        ActorCritic(7, hidden_size=64, initial_log_std=-1.5)
        actual_rng = torch.random.get_rng_state()

        torch.testing.assert_close(actual_rng, expected_rng, rtol=0.0, atol=0.0)

    def test_adaptive_exploration_uses_faster_optimizer_group(self):
        agent = PPOAgent(
            obs_dim=7,
            seed=3,
            device="cpu",
            lr=1e-4,
            exploration_lr_multiplier=10.0,
        )

        self.assertEqual(len(agent.opt.param_groups), 2)
        self.assertAlmostEqual(agent.opt.param_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(agent.opt.param_groups[1]["lr"], 1e-3)
        exploration_ids = {id(parameter) for parameter in agent.net.log_std_delta.parameters()}
        optimizer_exploration_ids = {
            id(parameter) for parameter in agent.opt.param_groups[1]["params"]
        }
        self.assertEqual(optimizer_exploration_ids, exploration_ids)

    def test_adaptive_exploration_starts_at_exact_scalar_baseline(self):
        agent = PPOAgent(obs_dim=7, seed=3, device="cpu", initial_log_std=-1.5)
        observations = torch.tensor(
            [
                [0.0, 1.0, 0.2, 0.0, 0.4, 0.2, 1.0],
                [1.0, 0.0, 0.8, 0.5, 1.0, 0.7, 1.0],
                [0.0, -1.0, 0.4, 1.0, 0.2, 0.9, 0.0],
            ],
            dtype=torch.float32,
        )

        effective = agent.net.effective_log_std(observations)

        torch.testing.assert_close(
            effective,
            torch.full((3, 1), -1.5, dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )

    def test_adaptive_exploration_can_learn_different_state_widths(self):
        agent = PPOAgent(obs_dim=7, seed=3, device="cpu", initial_log_std=-1.5)
        with torch.no_grad():
            first = agent.net.log_std_delta[0]
            last = agent.net.log_std_delta[2]
            first.weight.zero_()
            first.bias.zero_()
            first.weight[0, 3] = 1.0
            last.weight.zero_()
            last.weight[0, 0] = 1.0

        low_soc = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]])
        high_soc = torch.tensor([[0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]])

        low_std = float(agent.net.effective_log_std(low_soc).item())
        high_std = float(agent.net.effective_log_std(high_soc).item())

        self.assertNotEqual(low_std, high_std)

    def test_pre_iq29_checkpoint_loads_with_zero_adaptive_delta(self):
        source = PPOAgent(obs_dim=3, seed=5, device="cpu", initial_log_std=-1.5)
        legacy_state = {
            key: value
            for key, value in source.net.state_dict().items()
            if not key.startswith("log_std_delta.")
        }
        payload = {
            "algo": "ppo",
            "state_dict": legacy_state,
            "meta": {
                "hidden_size": source.hidden_size,
                "initial_log_std": source.initial_log_std,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/legacy-ppo.pt"
            torch.save(payload, path)
            loaded = PPOAgent(obs_dim=3, seed=9, device="cpu")
            loaded.load(path)

        observations = torch.tensor(
            [[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]],
            dtype=torch.float32,
        )
        effective = loaded.net.effective_log_std(observations)
        expected = torch.full((2, 1), -1.5, dtype=torch.float32)
        torch.testing.assert_close(effective, expected, rtol=0.0, atol=0.0)

    def test_deterministic_action_is_squashed_not_hard_clipped(self):
        agent = PPOAgent(obs_dim=3, seed=1, device="cpu")
        with torch.no_grad():
            for parameter in agent.collector_net.actor.parameters():
                parameter.zero_()
            agent.collector_net.actor[-1].bias.fill_(2.0)

        observation = np.zeros(3, dtype=np.float32)
        action, _log_probability, latent, _value = agent.act_with_latent(
            observation,
            deterministic=True,
        )

        self.assertAlmostEqual(latent, 2.0, places=6)
        self.assertAlmostEqual(action, math.tanh(2.0), places=6)
        self.assertLess(action, 0.999)

    def test_action_and_log_probability_use_the_same_pre_tanh_sample(self):
        agent = PPOAgent(obs_dim=3, seed=7, device="cpu")
        observation = np.asarray([0.2, -0.3, 0.4], dtype=np.float32)

        action, log_probability, latent, _value = agent.act_with_latent(observation)

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        distribution = agent.collector_net.dist(observation_tensor)
        latent_tensor = torch.tensor([[latent]], dtype=torch.float32)
        expected_log_probability = _squashed_log_prob_from_latent(
            distribution,
            latent_tensor,
        )

        self.assertGreaterEqual(action, -1.0)
        self.assertLessEqual(action, 1.0)
        self.assertAlmostEqual(action, math.tanh(latent), places=6)
        self.assertAlmostEqual(log_probability, float(expected_log_probability.item()), places=6)


if __name__ == "__main__":
    unittest.main()
