from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bess.agents.ppo_agent import PPOAgent
from bess.core.settings import PPO_DECOMPOSED_CRITIC, PPO_HIDDEN_SIZE


class PPOWideBrainTests(unittest.TestCase):
    def test_iq62_defaults_to_128_wide_scalar_recurrent_brain(self):
        self.assertEqual(PPO_HIDDEN_SIZE, 128)
        self.assertFalse(PPO_DECOMPOSED_CRITIC)

        agent = PPOAgent(obs_dim=7, device="cpu", recurrent_enabled=True)

        self.assertEqual(agent.hidden_size, 128)
        self.assertFalse(agent.decomposed_critic)
        self.assertEqual(agent.net.actor_encoder[0].in_features, 7)
        self.assertEqual(agent.net.actor_encoder[0].out_features, 128)
        self.assertEqual(agent.net.actor_gru.input_size, 128)
        self.assertEqual(agent.net.actor_gru.hidden_size, 128)
        self.assertEqual(agent.net.actor.in_features, 128)
        self.assertEqual(agent.net.critic_encoder[0].in_features, 7)
        self.assertEqual(agent.net.critic_encoder[0].out_features, 128)
        self.assertEqual(agent.net.critic_gru.input_size, 128)
        self.assertEqual(agent.net.critic_gru.hidden_size, 128)
        self.assertEqual(agent.net.critic.in_features, 128)
        self.assertFalse(hasattr(agent.net, "critic_demand"))
        self.assertFalse(hasattr(agent.net, "critic_wear"))
        self.assertEqual(agent.policy_architecture_name(), "brain7_separate_actor_critic_gru_v1")

    def test_iq62_keeps_exactly_seven_observation_inputs(self):
        agent = PPOAgent(obs_dim=7, device="cpu", recurrent_enabled=True)

        self.assertEqual(agent.obs_dim, 7)
        self.assertEqual(agent.net.obs_dim, 7)
        self.assertEqual(agent.net.actor_encoder[0].in_features, 7)
        self.assertEqual(agent.net.critic_encoder[0].in_features, 7)

    def test_explicit_64_width_remains_supported_for_old_experiments(self):
        agent = PPOAgent(
            obs_dim=7,
            device="cpu",
            recurrent_enabled=True,
            hidden_size=64,
        )

        self.assertEqual(agent.hidden_size, 64)
        self.assertEqual(agent.net.actor_gru.hidden_size, 64)
        self.assertEqual(agent.net.critic_gru.hidden_size, 64)
        self.assertFalse(agent.decomposed_critic)

    def test_128_scalar_checkpoint_roundtrip_rebuilds_smaller_loader(self):
        trained = PPOAgent(obs_dim=7, device="cpu", recurrent_enabled=True)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "iq62.pt"
            trained.save(checkpoint)

            loaded = PPOAgent(
                obs_dim=7,
                device="cpu",
                recurrent_enabled=True,
                hidden_size=64,
                decomposed_critic=True,
            )
            loaded.load(checkpoint)

        self.assertEqual(loaded.hidden_size, 128)
        self.assertFalse(loaded.decomposed_critic)
        self.assertEqual(loaded.net.actor_gru.hidden_size, 128)
        self.assertEqual(loaded.net.critic_gru.hidden_size, 128)
        self.assertFalse(hasattr(loaded.net, "critic_demand"))
        self.assertEqual(loaded.meta["hidden_size"], 128)
        self.assertFalse(loaded.meta["decomposed_critic"])
        self.assertEqual(loaded.meta["critic_components"], ["total"])

    def test_wider_brain_has_substantially_more_trainable_parameters(self):
        wide = PPOAgent(obs_dim=7, device="cpu", recurrent_enabled=True, hidden_size=128)
        old = PPOAgent(obs_dim=7, device="cpu", recurrent_enabled=True, hidden_size=64)

        wide_count = sum(parameter.numel() for parameter in wide.net.parameters())
        old_count = sum(parameter.numel() for parameter in old.net.parameters())

        self.assertGreater(wide_count, old_count * 3)
        self.assertLess(wide_count, old_count * 5)


if __name__ == "__main__":
    unittest.main()
