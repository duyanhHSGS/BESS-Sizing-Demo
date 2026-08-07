from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bess.core.bess_env import BESSEnv, OBS_DIM
from bess.core.common import load_system_config, make_bess_config
from bess.agents.grepo_agent import GREPOAgent
from bess.core.scenario_gen import DayData, MonthData


def _case(minutes=15):
    steps = 1440 // minutes
    x = np.arange(steps, dtype=np.float64)
    base = load_system_config()
    cfg = make_bess_config(base, 1000.0, 500.0, base.P_target_user)
    cfg.dt = minutes / 60.0
    day = DayData(
        load=650.0 + 120.0 * np.sin(2.0 * np.pi * x / steps),
        pv=np.maximum(0.0, 350.0 * np.sin(np.pi * x / steps)),
        day_type="working",
        weather="test",
        day_index=0,
        date_iso="2026-01-01",
    )
    return cfg, MonthData(days=[day], source="grepo_test")


class GREPOInferenceAndMathTests(unittest.TestCase):
    def test_observation_and_network_contracts_are_unchanged(self):
        agent = GREPOAgent(OBS_DIM, device="cpu")
        self.assertEqual(OBS_DIM, 13)
        self.assertEqual(agent.actor[0].in_features, 13)
        self.assertEqual(agent.actor[0].out_features, 256)
        self.assertEqual(agent.actor[2].out_features, 128)
        self.assertEqual(agent.critic[0].in_features, 13)

    def test_predict_action_matches_deterministic_act(self):
        agent = GREPOAgent(OBS_DIM, seed=7, device="cpu")
        observation = np.linspace(-1.0, 1.0, OBS_DIM, dtype=np.float32)
        expected, _, _ = agent.act(observation, deterministic=True)
        self.assertAlmostEqual(
            agent.predict_action(observation), expected, places=7
        )

    def test_vectorized_discounted_returns_match_scalar_reference(self):
        agent = GREPOAgent(OBS_DIM, gamma=0.97, device="cpu")
        rewards = np.random.default_rng(3).normal(
            size=(4, 37)
        ).astype(np.float32)
        expected = np.empty_like(rewards)
        for group_index in range(rewards.shape[0]):
            accumulator = 0.0
            for t in range(rewards.shape[1] - 1, -1, -1):
                accumulator = (
                    rewards[group_index, t] + agent.gamma * accumulator
                )
                expected[group_index, t] = accumulator
        np.testing.assert_allclose(
            agent._discounted_returns(rewards),
            expected,
            rtol=0.0,
            atol=1e-6,
        )

    def test_normal_log_probability_matches_analytical_formula(self):
        agent = GREPOAgent(OBS_DIM, std=0.30, device="cpu")
        action = torch.tensor([[-0.7], [0.1], [1.2]], dtype=torch.float32)
        mean = torch.tensor([[-0.2], [0.0], [0.8]], dtype=torch.float32)
        expected = (
            -0.5 * ((action - mean) / agent.std) ** 2
            - math.log(agent.std)
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(-1)
        torch.testing.assert_close(
            agent._logp(action, mean), expected, rtol=1e-6, atol=1e-7
        )


class GREPOCollectorParityTests(unittest.TestCase):
    def test_lockstep_collector_matches_scalar_with_shared_noise(self):
        cfg, month = _case(minutes=15)

        def make_env():
            return BESSEnv(
                cfg,
                p_ref_kw=1000.0,
                d_run_init_kw=300.0,
                gamma=0.995,
                control_dt_minutes=15.0,
            )

        agent = GREPOAgent(
            OBS_DIM, n_group=3, seed=11, epochs=1, device="cpu"
        )
        decisions = len(month.days[0].load)
        noise = np.random.default_rng(11).normal(
            size=(agent.n_group, decisions)
        ).astype(np.float32)
        optimized = agent.collect_group(
            make_env, month, soc_init=0.55, d_run_init=300.0,
            noise_g=noise,
        )
        scalar = agent.collect_group_scalar_reference(
            make_env, month, soc_init=0.55, d_run_init=300.0,
            noise_g=noise,
        )
        for actual, expected in zip(optimized, scalar):
            np.testing.assert_allclose(
                actual, expected, rtol=0.0, atol=1e-6
            )
        for actual_env, expected_env in zip(
            agent._group_envs, agent._scalar_reference_envs
        ):
            self.assertAlmostEqual(actual_env.soc, expected_env.soc, places=6)
            self.assertAlmostEqual(
                actual_env.d_run, expected_env.d_run, places=6
            )
            np.testing.assert_allclose(
                actual_env.log_grid[0],
                expected_env.log_grid[0],
                rtol=0.0,
                atol=1e-6,
            )

    def test_logging_disabled_preserves_physics_and_rewards(self):
        cfg, month = _case(minutes=15)
        recorded = BESSEnv(
            cfg, p_ref_kw=1000.0, d_run_init_kw=300.0,
            control_dt_minutes=15.0, record_trajectory=True,
        )
        unrecorded = BESSEnv(
            cfg, p_ref_kw=1000.0, d_run_init_kw=300.0,
            control_dt_minutes=15.0, record_trajectory=False,
        )
        obs_recorded = recorded.reset(month, soc_init=0.55)
        obs_unrecorded = unrecorded.reset(month, soc_init=0.55)
        np.testing.assert_array_equal(obs_recorded, obs_unrecorded)
        actions = np.sin(
            np.linspace(0.0, 4.0 * np.pi, len(month.days[0].load))
        )
        for action in actions:
            result_recorded = recorded.step(float(action))
            result_unrecorded = unrecorded.step(float(action))
            self.assertAlmostEqual(
                result_recorded[1], result_unrecorded[1], places=12
            )
            self.assertEqual(result_recorded[2], result_unrecorded[2])
            for key in (
                "grid_kw", "energy_delta", "peak_delta", "deg_cost",
                "shaping", "d_run",
            ):
                self.assertAlmostEqual(
                    result_recorded[3][key],
                    result_unrecorded[3][key],
                    places=12,
                )
            if not result_recorded[2]:
                np.testing.assert_array_equal(
                    result_recorded[0], result_unrecorded[0]
                )
        self.assertEqual(unrecorded.log_grid, [])
        self.assertEqual(unrecorded.log_soc, [])
        self.assertEqual(unrecorded.log_pbess, [])
        self.assertAlmostEqual(recorded.soc, unrecorded.soc, places=12)
        self.assertAlmostEqual(recorded.d_run, unrecorded.d_run, places=12)


class GREPOUpdateAndCheckpointTests(unittest.TestCase):
    def _finite_update_and_checkpoint(self, device):
        rng = np.random.default_rng(17)
        agent = GREPOAgent(
            OBS_DIM, n_group=2, seed=17, epochs=1, minibatch=16,
            device=device,
        )
        obs = rng.normal(size=(2, 24, OBS_DIM)).astype(np.float32)
        action = rng.normal(size=(2, 24)).astype(np.float32)
        mean = torch.zeros((48, 1), dtype=torch.float32)
        action_tensor = torch.from_numpy(action.reshape(48, 1))
        logp = agent._logp(action_tensor, mean).numpy().reshape(2, 24)
        rewards = rng.normal(size=(2, 24)).astype(np.float32)
        losses = agent.update(obs, action, logp, rewards)
        self.assertTrue(np.isfinite(list(losses.values())).all())
        self.assertTrue(all(
            torch.isfinite(parameter).all()
            for parameter in agent._params
        ))

        agent.meta = {"contract": "unchanged"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grepo.pt"
            agent.save(path)
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(
                set(checkpoint),
                {"actor", "critic", "obs_dim", "std", "algo", "meta"},
            )
            self.assertTrue(all(
                tensor.device.type == "cpu"
                for tensor in checkpoint["actor"].values()
            ))
            loaded = GREPOAgent(OBS_DIM, device="cpu")
            loaded.load(path)
            probe = np.zeros(OBS_DIM, dtype=np.float32)
            self.assertEqual(
                loaded.predict_action(probe), agent.predict_action(probe)
            )

    def test_cpu_update_is_finite_and_checkpoint_is_portable(self):
        self._finite_update_and_checkpoint("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_update_is_finite_and_checkpoint_is_portable(self):
        self._finite_update_and_checkpoint("cuda")


if __name__ == "__main__":
    unittest.main()
