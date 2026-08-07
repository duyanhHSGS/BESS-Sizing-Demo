from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ppo2_agent import PPO2Agent, RolloutBuffer, resolve_ppo2_device
from scenario_gen import DayData, MonthData
from training_common import augment_month, build_training_bess_config


class SharedTrainingHelpersTests(unittest.TestCase):
    def test_augmentation_is_seeded_and_handles_pv(self):
        day = DayData(
            load=np.full(8, 100.0, dtype=np.float64),
            pv=np.full(8, 25.0, dtype=np.float64),
            day_type="working",
            weather="test",
            day_index=1,
            date_iso="2026-01-01",
        )
        month = MonthData(days=[day], source="test")

        first = augment_month(month, np.random.default_rng(7))
        second = augment_month(month, np.random.default_rng(7))

        np.testing.assert_allclose(first.days[0].load, second.days[0].load)
        np.testing.assert_allclose(first.days[0].pv, second.days[0].pv)
        self.assertTrue(np.isfinite(first.days[0].pv).all())
        self.assertTrue((first.days[0].pv >= 0.0).all())

    def test_shared_config_builds_windows_at_dataset_resolution(self):
        tariff = {
            "price_peak": 3000.0,
            "price_mid": 1500.0,
            "price_off": 900.0,
            "t_cap": 200000.0,
            "billing_mode": "tou",
            "peak_windows": "17:00-18:00",
            "off_windows": "00:00-01:00",
            "sunday_no_peak": False,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
            "minimum_soc": 0.10,
            "maximum_soc": 0.90,
            "required_final_soc": 0.50,
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "training_config.json"
            config_path.write_text(json.dumps(tariff), encoding="utf-8")
            cfg, billing = build_training_bess_config(
                1000.0,
                500.0,
                1.0 / 60.0,
                config_path,
            )

        self.assertEqual(billing, "tou")
        self.assertAlmostEqual(cfg.dt, 1.0 / 60.0)
        self.assertEqual(len(cfg.OFF), 60)
        self.assertEqual(len(cfg.W2), 60)
        self.assertEqual(cfg.OFF_PEAK_END_STEP, 60)
        self.assertEqual(cfg.T_cap, 0.0)


class PPO2DeviceTests(unittest.TestCase):
    def test_cpu_device_path_updates_and_syncs_collector(self):
        self.assertEqual(resolve_ppo2_device("cpu"), "cpu")
        agent = PPO2Agent(
            obs_dim=4,
            seed=3,
            device="cpu",
            epochs=1,
            minibatch=4,
        )
        buffer = RolloutBuffer(size=4, obs_dim=4)
        rng = np.random.default_rng(3)
        for index in range(4):
            obs = rng.normal(size=4).astype(np.float32)
            action, logp, latent, value_energy, value_peak = agent.act(obs)
            buffer.add(
                obs,
                action,
                latent,
                logp,
                0.01 * index,
                -0.005 * index,
                value_energy,
                value_peak,
                0.0,
            )

        agent.update(buffer, 0.0, 0.0)

        self.assertEqual(agent.device.type, "cpu")
        for learner, collector in zip(
            agent.net.parameters(), agent.collector_net.parameters(), strict=True
        ):
            torch.testing.assert_close(learner.detach().cpu(), collector.detach())
        probe = np.zeros(4, dtype=np.float32)
        self.assertTrue(np.isfinite(agent.predict_action(probe)))


if __name__ == "__main__":
    unittest.main()
