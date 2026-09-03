from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from bess.agents.ppo_agent import PPOAgent, RolloutBuffer, _gae_advantages
from bess.training.runners.train_ppo_dataset import (
    _behavior_clone_critic,
    _control_reward_components_million_vnd,
    _restore_reanchor_state,
)


class PPODecomposedCriticTests(unittest.TestCase):
    def test_rollout_buffer_derives_scalar_reward_and_value_from_components(self):
        buffer = RolloutBuffer(1, 3, decomposed_rewards=True)
        buffer.add(
            np.zeros(3, dtype=np.float32),
            0.0,
            0.0,
            999.0,
            999.0,
            1.0,
            np.zeros(1, dtype=np.float32),
            reward_components=(2.0, 5.0, 1.5),
            value_components=(3.0, 7.0, 2.0),
        )
        self.assertAlmostEqual(float(buffer.rew[0]), 5.5)
        self.assertAlmostEqual(float(buffer.val[0]), 8.0)
        self.assertAlmostEqual(float(buffer.rew_energy[0]), 2.0)
        self.assertAlmostEqual(float(buffer.rew_demand[0]), 5.0)
        self.assertAlmostEqual(float(buffer.rew_wear[0]), 1.5)

    def test_rollout_buffer_rejects_missing_component_targets(self):
        buffer = RolloutBuffer(1, 3, decomposed_rewards=True)
        with self.assertRaisesRegex(ValueError, "decomposed rollout transition"):
            buffer.add(
                np.zeros(3, dtype=np.float32),
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                np.zeros(1, dtype=np.float32),
            )

    def test_component_gae_sums_before_single_normalization(self):
        agent = PPOAgent(
            obs_dim=3,
            device="cpu",
            decomposed_critic=True,
            gamma=0.9,
            lam=0.8,
        )
        buffer = RolloutBuffer(3, 3, decomposed_rewards=True)
        rewards = [(1.0, 5.0, 0.5), (2.0, 0.0, 0.2), (3.0, 7.0, 0.1)]
        values = [(0.4, 0.2, 0.1), (0.3, 0.5, 0.2), (0.2, 0.7, 0.3)]
        for index in range(3):
            buffer.add(
                np.full(3, index, dtype=np.float32),
                0.0,
                0.0,
                0.0,
                0.0,
                float(index == 2),
                np.zeros(1, dtype=np.float32),
                reward_components=rewards[index],
                value_components=values[index],
            )

        normalized, _total_return, component_returns, stats = agent._prepare_rollout_targets(
            buffer,
            0.0,
            (0.0, 0.0, 0.0),
        )
        done = buffer.done[:3]
        adv_e = _gae_advantages(
            buffer.rew_energy[:3], buffer.val_energy[:3], done,
            last_val=0.0, gamma=0.9, lam=0.8,
        )
        adv_d = _gae_advantages(
            buffer.rew_demand[:3], buffer.val_demand[:3], done,
            last_val=0.0, gamma=0.9, lam=0.8,
        )
        adv_w = _gae_advantages(
            buffer.rew_wear[:3], buffer.val_wear[:3], done,
            last_val=0.0, gamma=0.9, lam=0.8,
        )
        raw = adv_e + adv_d - adv_w
        expected = (raw - raw.mean()) / (raw.std() + 1e-8)

        np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            component_returns[0] + component_returns[1] - component_returns[2],
            raw + buffer.val[:3],
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertAlmostEqual(stats["advantage_std_raw"], float(raw.std()), places=6)

    def test_recurrent_update_trains_all_three_heads_and_reports_diagnostics(self):
        agent = PPOAgent(
            obs_dim=7,
            seed=11,
            device="cpu",
            decomposed_critic=True,
            recurrent_enabled=True,
            recurrent_sequence_length=4,
            epochs=1,
            minibatch=8,
        )
        buffer = RolloutBuffer(
            8,
            7,
            recurrent_hidden_size=agent.hidden_size,
            decomposed_rewards=True,
        )
        before = {
            name: getattr(agent.net, name).weight.detach().clone()
            for name in ("critic", "critic_demand", "critic_wear")
        }
        for index in range(8):
            obs = np.linspace(0.0, 1.0, 7, dtype=np.float32) + index * 0.01
            action, logp, latent, value = agent.act_with_latent(obs)
            actor_hidden, critic_hidden = agent.recurrent_rollout_inputs()
            buffer.add(
                obs,
                action,
                logp,
                0.0,
                value,
                float(index == 7),
                latent,
                actor_hidden=actor_hidden,
                critic_hidden=critic_hidden,
                reward_components=(0.1 * (index + 1), 0.3 if index == 7 else 0.0, 0.02),
                value_components=agent.last_value_components,
            )
        stats = agent.update(buffer, 0.0, (0.0, 0.0, 0.0))

        for name in ("critic", "critic_demand", "critic_wear"):
            self.assertFalse(torch.equal(getattr(agent.net, name).weight, before[name]), name)
        for key in (
            "energy_value_loss",
            "demand_value_loss",
            "wear_value_loss",
            "energy_explained_variance",
            "demand_explained_variance",
            "wear_explained_variance",
        ):
            self.assertTrue(np.isfinite(stats[key]), key)

    def test_decomposed_checkpoint_roundtrip_rebuilds_architecture(self):
        agent = PPOAgent(
            obs_dim=7,
            seed=19,
            device="cpu",
            decomposed_critic=True,
            recurrent_enabled=True,
        )
        obs = np.linspace(0.0, 1.0, 7, dtype=np.float32)
        expected_action = agent.predict_action(obs)
        agent.reset_recurrent_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            agent.save(path)
            loaded = PPOAgent(obs_dim=7, seed=99, device="cpu")
            loaded.load(path)

        self.assertTrue(loaded.decomposed_critic)
        self.assertTrue(loaded.recurrent_enabled)
        self.assertEqual(loaded.meta["critic_components"], ["energy", "demand", "wear"])
        self.assertEqual(loaded.policy_architecture_name(), "brain7_actor_gru_shared_critic_gru_3head_v2")
        loaded.reset_recurrent_state()
        self.assertAlmostEqual(loaded.predict_action(obs), expected_action, places=7)

    def test_scalar_checkpoint_stays_scalar_for_backward_compatibility(self):
        scalar = PPOAgent(obs_dim=4, seed=23, device="cpu", decomposed_critic=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            scalar.save(path)
            loaded = PPOAgent(obs_dim=4, seed=24, device="cpu", decomposed_critic=True)
            loaded.load(path)
        self.assertFalse(loaded.decomposed_critic)
        self.assertFalse(hasattr(loaded.net, "critic_demand"))

    def test_reanchor_preserves_every_live_decomposed_critic_tensor(self):
        agent = PPOAgent(
            obs_dim=7,
            seed=31,
            device="cpu",
            decomposed_critic=True,
            recurrent_enabled=True,
        )
        champion = agent.snapshot_training_state()
        with torch.no_grad():
            for parameter in agent.net.parameters():
                parameter.add_(0.125)
        live = {key: value.detach().clone() for key, value in agent.net.state_dict().items()}

        _restore_reanchor_state(
            agent,
            champion,
            preserve_critic=True,
            reset_optimizer=True,
        )
        critic_prefixes = agent.critic_state_prefixes()
        for key, value in agent.net.state_dict().items():
            expected = live[key] if key.startswith(critic_prefixes) else champion["network"][key]
            self.assertTrue(torch.equal(value, expected), key)

    def test_reward_components_preserve_old_scalar_learning_objective(self):
        native_results = (
            SimpleNamespace(
                electricity_energy_savings_vnd=2_000_000.0,
                demand_savings_vnd=5_000_000.0,
                battery_wear_cost_vnd=400_000.0,
            ),
            SimpleNamespace(
                electricity_energy_savings_vnd=-500_000.0,
                demand_savings_vnd=1_000_000.0,
                battery_wear_cost_vnd=100_000.0,
            ),
        )
        transition = SimpleNamespace(native_results=native_results)
        energy, demand, wear = _control_reward_components_million_vnd(
            transition,
            mismatch_penalty_vnd=2_000_000.0,
            mismatch_shaping_scale=0.10,
        )
        self.assertAlmostEqual(energy, 1.5)
        self.assertAlmostEqual(demand, 6.0)
        self.assertAlmostEqual(wear, 0.7)
        self.assertAlmostEqual(energy + demand - wear, 6.8)

    def test_oracle_critic_bc_learns_component_returns_without_month_leakage(self):
        agent = PPOAgent(obs_dim=7, seed=41, device="cpu", decomposed_critic=True)
        observations = np.zeros((4, 7), dtype=np.float32)
        rewards = np.asarray(
            [
                (1.0, 10.0, 1.0),
                (2.0, 20.0, 2.0),
                (100.0, 1000.0, 100.0),
                (200.0, 2000.0, 200.0),
            ],
            dtype=np.float32,
        )
        stats = _behavior_clone_critic(
            agent,
            observations,
            rewards,
            gamma=1.0,
            seed=42,
            max_epochs=0,
            episode_lengths=[2, 2],
        )
        expected_returns = np.asarray(
            [
                (3.0, 30.0, 3.0),
                (2.0, 20.0, 2.0),
                (300.0, 3000.0, 300.0),
                (200.0, 2000.0, 200.0),
            ],
            dtype=np.float32,
        )
        self.assertAlmostEqual(stats["critic_energy_target_mean"], float(expected_returns[:, 0].mean()))
        self.assertAlmostEqual(stats["critic_demand_target_mean"], float(expected_returns[:, 1].mean()))
        self.assertAlmostEqual(stats["critic_wear_target_mean"], float(expected_returns[:, 2].mean()))

    def test_oracle_critic_bc_reduces_decomposed_return_mse(self):
        rng = np.random.default_rng(51)
        observations = rng.normal(size=(32, 7)).astype(np.float32)
        rewards = np.stack(
            (
                0.2 + observations[:, 0] * 0.1,
                np.maximum(observations[:, 1], 0.0) * 0.3,
                np.abs(observations[:, 2]) * 0.05,
            ),
            axis=1,
        ).astype(np.float32)
        agent = PPOAgent(obs_dim=7, seed=52, device="cpu", decomposed_critic=True)
        stats = _behavior_clone_critic(
            agent,
            observations,
            rewards,
            gamma=0.9,
            seed=53,
            max_epochs=20,
            learning_rate=1e-3,
            minibatch=16,
            target_mse=0.0,
            episode_lengths=[16, 16],
        )
        self.assertLessEqual(stats["critic_final_mse"], stats["critic_initial_mse"])
        for name in ("energy", "demand", "wear"):
            self.assertTrue(np.isfinite(stats[f"critic_{name}_final_mse"]))


if __name__ == "__main__":
    unittest.main()
