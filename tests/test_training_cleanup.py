from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bess.agents.ppo2_agent import PPO2Agent, RolloutBuffer, resolve_ppo2_device
from bess.core.scenario_gen import DayData, MonthData
from bess.paths import PROJECT_ROOT
from bess.training.training_common import augment_month, build_training_bess_config
from bess.training.training_launcher import (
    TrainingLaunchError,
    build_training_command,
    write_training_config,
)


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
            "battery_wear_cost": 500.0,
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
        self.assertEqual(cfg.battery_wear_cost_vnd_per_kwh, 500.0)


class PPO2LauncherTests(unittest.TestCase):
    def test_reference_command_uses_internal_oracle_and_senior_step_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            csv_path = base / "site.csv"
            config_path = base / "training_config.json"
            oracle_path = base / "old_oracle.json"
            rows = ["date_iso,day_index,step,day_type,P_load_kW,P_pv_kW"]
            rows.extend(
                f"2026-01-01,1,{step},working,100,0" for step in range(96)
            )
            csv_path.write_text("\n".join(rows), encoding="utf-8")
            config_path.write_text("{}", encoding="utf-8")
            payload = {
                "algo": "ppo2",
                "dataset_id": "test",
                "e_cap_kwh": 1000.0,
                "p_rated_kw": 500.0,
                "obs_variant": "base",
                "device": "cpu",
            }
            spec = build_training_command(
                payload,
                csv_path,
                config_path,
                oracle_path,
                python_executable="python",
                base_dir=base,
            )
        command = spec["cmd"]
        self.assertNotIn("--oracle-cache", command)
        steps_index = command.index("--steps")
        self.assertEqual(command[steps_index + 1], "1500000")
        self.assertEqual(command[command.index("--control-dt-minutes") + 1], "15")

    def test_reference_command_forwards_custom_ppo2_knobs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            csv_path = base / "site.csv"
            config_path = base / "training_config.json"
            oracle_path = base / "old_oracle.json"
            rows = ["date_iso,day_index,step,day_type,P_load_kW,P_pv_kW"]
            rows.extend(f"2026-01-01,1,{step},working,100,0" for step in range(96))
            csv_path.write_text("\n".join(rows), encoding="utf-8")
            config_path.write_text("{}", encoding="utf-8")
            payload = {
                "algo": "ppo2", "dataset_id": "test", "e_cap_kwh": 1000.0,
                "p_rated_kw": 500.0, "obs_variant": "base", "device": "cpu",
                "ppo2_rollout": 960, "ppo2_epochs": 4, "ppo2_minibatch": 128,
                "ppo2_lam_peak": "0.7,0.97", "ppo2_torch_threads": 3,
            }
            command = build_training_command(
                payload, csv_path, config_path, oracle_path,
                python_executable="python", base_dir=base,
            )["cmd"]
        self.assertEqual(command[command.index("--rollout") + 1], "960")
        self.assertEqual(command[command.index("--ppo-epochs") + 1], "4")
        self.assertEqual(command[command.index("--minibatch") + 1], "128")
        self.assertEqual(command[command.index("--lambda-peak") + 1], "0.7,0.97")
        self.assertEqual(command[command.index("--torch-threads") + 1], "3")

    def test_reference_command_rejects_non_reference_gamma(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            csv_path = base / "site.csv"
            config_path = base / "training_config.json"
            oracle_path = base / "old_oracle.json"
            rows = ["date_iso,day_index,step,day_type,P_load_kW,P_pv_kW"]
            rows.extend(f"2026-01-01,1,{step},working,100,0" for step in range(96))
            csv_path.write_text("\n".join(rows), encoding="utf-8")
            config_path.write_text("{}", encoding="utf-8")
            payload = {
                "algo": "ppo2", "dataset_id": "test", "e_cap_kwh": 1000.0,
                "p_rated_kw": 500.0, "obs_variant": "base", "device": "cpu",
                "ppo2_gamma": 0.99,
            }
            with self.assertRaisesRegex(TrainingLaunchError, "gamma=1.0"):
                build_training_command(
                    payload, csv_path, config_path, oracle_path,
                    python_executable="python", base_dir=base,
                )

    def test_training_config_carries_battery_wear_cost(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = write_training_config(
                {"battery_wear_cost": "500"}, Path(directory)
            )
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["battery_wear_cost"], 500.0)


class PPO2DeviceTests(unittest.TestCase):
    def test_reference_cpu_device_path_updates_single_network(self):
        self.assertEqual(resolve_ppo2_device("cpu"), "cpu")
        self.assertEqual(resolve_ppo2_device("auto"), "cpu")
        with self.assertRaisesRegex(RuntimeError, "CPU-only"):
            resolve_ppo2_device("cuda")
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
        self.assertFalse(hasattr(agent, "collector_net"))
        self.assertTrue(np.isfinite(agent.diagnostics["approx_kl"]))
        probe = np.zeros(4, dtype=np.float32)
        self.assertTrue(np.isfinite(agent.predict_action(probe)))


if __name__ == "__main__":
    unittest.main()
