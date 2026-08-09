from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from bess.agents.ppo_agent import PPOAgent, _squashed_log_prob_from_latent


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


if __name__ == "__main__":
    unittest.main()
