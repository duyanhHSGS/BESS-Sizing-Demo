from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bess.evaluation.benchmark import (
    _demand_windows,
    _rolling_30_minute_average,
)
from bess.core.bess_env import BESSEnv
from bess.core.common import load_system_config, make_bess_config
from bess.agents.ppo_agent import PPOAgent, RolloutBuffer
from bess.core.scenario_gen import DayData, MonthData


def _reference_rolling(values, dt):
    values = list(values)
    return [
        sum(values[step] * weight for step, weight in window)
        for window in _demand_windows(len(values), dt)
    ]


class RollingDemandTests(unittest.TestCase):
    def test_vectorized_rolling_matches_reference_at_supported_resolutions(self):
        rng = np.random.default_rng(123)
        for minutes in (1, 5, 15, 30):
            with self.subTest(minutes=minutes):
                values = rng.uniform(0.0, 1500.0, 1440 // minutes)
                dt = minutes / 60.0
                actual = _rolling_30_minute_average(values, dt)
                expected = _reference_rolling(values, dt)
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-10)

    def test_partial_end_of_day_windows_keep_existing_normalization(self):
        values = np.asarray([10.0, 20.0, 30.0, 40.0])
        actual = _rolling_30_minute_average(values, 5.0 / 60.0)
        expected = _reference_rolling(values, 5.0 / 60.0)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        self.assertEqual(actual[-1], 40.0)


class InferenceAndEnvironmentTests(unittest.TestCase):
    def test_actor_only_prediction_matches_deterministic_act(self):
        agent = PPOAgent(obs_dim=13, seed=7)
        obs = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
        expected, _, _ = agent.act(obs, deterministic=True)
        self.assertEqual(agent.predict_action(obs), expected)

    def test_native_trajectory_constraints_and_running_peak(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1000.0, 500.0, base.P_target_user)
        cfg.dt = 1.0 / 60.0
        steps = 1440
        x = np.arange(steps, dtype=np.float64)
        day = DayData(
            load=700.0 + 100.0 * np.sin(2.0 * np.pi * x / steps),
            pv=np.maximum(0.0, 300.0 * np.sin(np.pi * x / steps)),
            day_type="working",
            weather="test",
            day_index=0,
            date_iso="2026-01-01",
        )
        env = BESSEnv(
            cfg,
            p_ref_kw=1000.0,
            d_run_init_kw=100.0,
            control_dt_minutes=1.0,
        )
        obs = env.reset(MonthData(days=[day], source="test"))
        rewards = []
        done = False
        for step in range(steps):
            action = math.sin(2.0 * math.pi * step / steps)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            self.assertEqual(info["native_rows"], 1)

        self.assertTrue(done)
        self.assertIsNone(obs)
        self.assertTrue(np.isfinite(rewards).all())
        self.assertTrue((env.log_grid[0] >= 0.0).all())
        self.assertGreaterEqual(env.log_soc[0].min(), cfg.SOC_min - 1e-12)
        self.assertLessEqual(env.log_soc[0].max(), cfg.SOC_max + 1e-12)
        trailing = np.convolve(
            env.log_grid[0], np.ones(env.roll_k) / env.roll_k, mode="valid"
        )
        self.assertAlmostEqual(
            env.d_run,
            max(env.d_run_init, float(trailing.max())),
            places=9,
        )

    def test_seeded_update_stays_finite_and_checkpoint_is_compatible(self):
        agent = PPOAgent(obs_dim=13, seed=11, epochs=1, minibatch=8)
        buffer = RolloutBuffer(16, 13)
        rng = np.random.default_rng(11)
        obs = rng.normal(size=13).astype(np.float32)
        for index in range(buffer.size):
            action, logp, value = agent.act(obs)
            buffer.add(obs, action, logp, float(index) / 100.0, value, 0.0)
            obs = rng.normal(size=13).astype(np.float32)
        agent.update(buffer, 0.0)
        self.assertTrue(
            all(torch.isfinite(parameter).all() for parameter in agent.net.parameters())
        )

        agent.meta = {"native_dt_minutes": 1.0, "control_dt_minutes": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            agent.save(path)
            loaded = PPOAgent(obs_dim=13)
            loaded.load(path)
            self.assertEqual(loaded.meta, agent.meta)
            probe = np.zeros(13, dtype=np.float32)
            self.assertEqual(loaded.predict_action(probe), agent.predict_action(probe))


if __name__ == "__main__":
    unittest.main()
