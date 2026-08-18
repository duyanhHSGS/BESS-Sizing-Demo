from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bess.agents.ppo_agent import PPOAgent, RolloutBuffer
from bess.core.bess_env import OBSERVATION_DIM
from bess.core.brain_runtime import (
    BrainTrajectoryRecorder,
    make_brain_env,
    observation_array,
    step_brain_control,
)
from bess.core.common import load_system_config, make_bess_config
from bess.core.scenario_gen import DayData, MonthData
from bess.evaluation.baselines import run_drl_policy
from bess.evaluation.benchmark import _rolling_30_minute_average


def _reference_fixed_30_minute_meter(values, dt):
    values = np.asarray(values, dtype=np.float64)
    samples_per_block = round(0.5 / dt)
    if len(values) % samples_per_block:
        raise ValueError("Grid day must contain complete 30-minute meter intervals")
    block_averages = values.reshape(-1, samples_per_block).mean(axis=1)
    return np.repeat(block_averages, samples_per_block)


class FixedDemandBlockTests(unittest.TestCase):
    def test_vectorized_fixed_blocks_match_reference_at_supported_resolutions(self):
        rng = np.random.default_rng(123)
        for minutes in (1, 5, 15, 30):
            with self.subTest(minutes=minutes):
                values = rng.uniform(0.0, 1500.0, 1440 // minutes)
                dt = minutes / 60.0
                actual = _rolling_30_minute_average(values, dt)
                expected = _reference_fixed_30_minute_meter(values, dt)
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-10)

    def test_partial_end_of_day_block_is_rejected(self):
        values = np.asarray([10.0, 20.0, 30.0, 40.0])
        with self.assertRaisesRegex(ValueError, "complete 30-minute meter intervals"):
            _rolling_30_minute_average(values, 5.0 / 60.0)


class InferenceAndEnvironmentTests(unittest.TestCase):
    def test_actor_only_prediction_matches_deterministic_act(self):
        agent = PPOAgent(obs_dim=OBSERVATION_DIM, seed=7)
        obs = np.linspace(-1.0, 1.0, OBSERVATION_DIM, dtype=np.float32)
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
        month = MonthData(days=[day], source="test")
        env = make_brain_env(
            month,
            cfg,
            power_scale_kw=1000.0,
            initial_state_of_charge=0.50,
        )
        obs = observation_array(env.reset())
        self.assertEqual(obs.shape, (OBSERVATION_DIM,))
        recorder = BrainTrajectoryRecorder(month, 0.50)
        rewards = []
        done = False
        for step in range(steps):
            action = math.sin(2.0 * math.pi * step / steps)
            transition = step_brain_control(
                env,
                action,
                native_steps=1,
                recorder=recorder,
            )
            rewards.append(transition.reward_vnd)
            done = transition.done

        self.assertTrue(done)
        self.assertTrue(np.isfinite(rewards).all())
        grid = recorder.grid_import_days[0]
        soc = recorder.state_of_charge_days[0]
        self.assertTrue((grid >= 0.0).all())
        self.assertGreaterEqual(soc.min(), cfg.SOC_min - 1e-12)
        self.assertLessEqual(soc.max(), cfg.SOC_max + 1e-12)
        fixed_block_averages = grid.reshape(-1, 30).mean(axis=1)
        self.assertAlmostEqual(
            env.bess_world.meter_state.monthly_peak_kw,
            float(fixed_block_averages.max()),
            places=9,
        )

    def test_policy_rollout_starts_at_soc_min_without_terminal_soc_target(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1000.0, 500.0, base.P_target_user)
        cfg.dt = 0.25
        steps = 96
        day = DayData(
            load=np.full(steps, 700.0, dtype=np.float64),
            pv=np.zeros(steps, dtype=np.float64),
            day_type="working",
            weather="test",
            day_index=0,
            date_iso="2026-01-01",
        )
        month = MonthData(days=[day], source="soc-min-start-test")

        class AlwaysChargeAgent:
            def __init__(self):
                self.meta = {
                    "obs_dim": OBSERVATION_DIM,
                    "control_dt_minutes": 15.0,
                    "battery_wear_cost": 0.0,
                }

            @staticmethod
            def predict_action(_observation):
                return -1.0

        result = run_drl_policy(
            month,
            cfg,
            AlwaysChargeAgent(),
            p_ref_kw=1000.0,
        )

        self.assertAlmostEqual(float(result["soc_days"][0][0]), cfg.SOC_min)
        self.assertAlmostEqual(float(result["soc_days"][-1][-1]), cfg.SOC_max)
        self.assertGreater(result["blocked_action_pct"], 0.0)

    def test_seeded_update_stays_finite_and_checkpoint_is_compatible(self):
        agent = PPOAgent(obs_dim=OBSERVATION_DIM, seed=11, epochs=1, minibatch=8)
        buffer = RolloutBuffer(16, OBSERVATION_DIM)
        rng = np.random.default_rng(11)
        obs = rng.normal(size=OBSERVATION_DIM).astype(np.float32)
        for index in range(buffer.size):
            action, logp, latent, value = agent.act_with_latent(obs)
            buffer.add(
                obs,
                action,
                logp,
                float(index) / 100.0,
                value,
                0.0,
                latent,
            )
            obs = rng.normal(size=OBSERVATION_DIM).astype(np.float32)
        agent.update(buffer, 0.0)
        self.assertTrue(
            all(torch.isfinite(parameter).all() for parameter in agent.net.parameters())
        )

        agent.meta = {
            "obs_dim": OBSERVATION_DIM,
            "native_dt_minutes": 1.0,
            "control_dt_minutes": 1.0,
            "native_steps_per_action": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            agent.save(path)
            loaded = PPOAgent(obs_dim=OBSERVATION_DIM)
            loaded.load(path)
            self.assertEqual(loaded.meta, agent.meta)
            probe = np.zeros(OBSERVATION_DIM, dtype=np.float32)
            self.assertEqual(loaded.predict_action(probe), agent.predict_action(probe))


if __name__ == "__main__":
    unittest.main()
