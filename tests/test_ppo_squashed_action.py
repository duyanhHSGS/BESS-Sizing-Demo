from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from bess.agents.ppo_agent import PPOAgent, _compute_gae, _squashed_log_prob_from_latent


class PPOSquashedActionTests(unittest.TestCase):
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

    def test_gae_does_not_bootstrap_or_leak_across_done_transition(self):
        rewards = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        values = np.asarray([10.0, 20.0, 999.0], dtype=np.float32)
        dones = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        advantages = _compute_gae(
            rewards,
            values,
            dones,
            last_value=123.0,
            gamma=1.0,
            lam=1.0,
        )

        # Transition 1 terminates its episode, so value[2] belongs to a fresh
        # episode and must never affect transitions 0 or 1.
        self.assertAlmostEqual(float(advantages[1]), 1.0 - 20.0, places=6)
        self.assertAlmostEqual(float(advantages[0]), 1.0 - 10.0, places=6)
        # The final non-terminal transition still bootstraps from last_value.
        self.assertAlmostEqual(float(advantages[2]), 123.0 - 999.0, places=6)


if __name__ == "__main__":
    unittest.main()
