from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from baselines import run_drl_policy, validate_dispatch_sampling
from common import load_system_config, make_bess_config
from dispatch_runner import DispatchRunWarning, run_policy_dispatch
from scenario_gen import DayData, MonthData
from settings import DEFAULT_PARAMETERS


class CountingPolicy:
    def __init__(self):
        self.calls = 0
        self.meta = {
            "algo": "ppo",
            "native_dt_minutes": 15.0,
            "control_dt_minutes": 15.0,
            "native_steps_per_action": 1,
            "e_cap_kwh": 1000.0,
            "p_rated_kw": 500.0,
            "p_ref_kw": 1500.0,
        }

    def predict_action(self, _obs):
        self.calls += 1
        return -1.0


def dense_month(load_kw=0.0, pv_kw=1000.0):
    steps = 1440
    return MonthData(
        days=[
            DayData(
                load=np.full(steps, load_kw, dtype=np.float64),
                pv=np.full(steps, pv_kw, dtype=np.float64),
                day_type="working",
                weather="test",
                day_index=1,
                date_iso="2026-01-01",
            )
        ],
        source="test",
    )


def dense_config():
    base = load_system_config()
    cfg = make_bess_config(base, 1000.0, 500.0, base.P_target_user)
    cfg.dt = 1.0 / 60.0
    return cfg


class CrossResolutionDispatchTests(unittest.TestCase):
    def test_legacy_policy_uses_96_decisions_and_1440_native_updates(self):
        policy = CountingPolicy()
        cfg = dense_config()
        initial_soc = cfg.SOC_eod

        result = run_drl_policy(dense_month(), cfg, policy, p_ref_kw=1500.0)

        self.assertEqual(policy.calls, 96)
        self.assertEqual(len(result["p_grid_days"][0]), 1440)
        self.assertEqual(len(result["p_bess_days"][0]), 1440)
        self.assertEqual(len(result["soc_days"][0]), 1441)
        expected_soc = initial_soc + 500.0 * cfg.eta_ch * (15.0 / 60.0) / cfg.E_cap
        self.assertAlmostEqual(result["soc_days"][0][15], expected_soc, places=12)
        np.testing.assert_allclose(result["p_bess_days"][0][:15], -500.0)

    def test_dense_demand_window_uses_30_native_samples(self):
        policy = CountingPolicy()
        policy.predict_action = lambda _obs: 0.0
        cfg = dense_config()
        cfg.T_cap = 1.0

        result = run_drl_policy(
            dense_month(load_kw=300.0, pv_kw=0.0),
            cfg,
            policy,
            p_ref_kw=1500.0,
        )

        rolling = np.convolve(result["p_grid_days"][0], np.ones(30) / 30.0, mode="valid")
        self.assertEqual(len(rolling), 1411)
        np.testing.assert_allclose(rolling, 300.0)

    def test_incompatible_sampling_ratios_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not an exact multiple"):
            validate_dispatch_sampling({"control_dt_minutes": 15.0}, 4.0)
        with self.assertRaisesRegex(ValueError, "data is coarser"):
            validate_dispatch_sampling({"control_dt_minutes": 15.0}, 30.0)

    def test_dispatch_result_warns_but_succeeds_cross_resolution(self):
        policy = CountingPolicy()
        parameters = {
            **DEFAULT_PARAMETERS,
            "dt": str(1.0 / 60.0),
            "billing_mode": "2tc",
        }

        with patch(
            "dispatch_runner.load_policy",
            return_value=(policy, "ppo", policy.meta),
        ):
            result = run_policy_dispatch(
                "policy_old_deprecated.pt",
                parameters,
                month=dense_month(),
            )

        self.assertEqual(policy.calls, 96)
        self.assertEqual(len(result["days"][0]["grid"]), 1440)
        self.assertIn("15-minute decisions and 1-minute physics", result["warnings"][0])

    def test_dispatch_rejects_non_divisible_selected_data(self):
        policy = CountingPolicy()
        parameters = {
            **DEFAULT_PARAMETERS,
            "dt": str(4.0 / 60.0),
            "billing_mode": "2tc",
        }

        with patch(
            "dispatch_runner.load_policy",
            return_value=(policy, "ppo", policy.meta),
        ):
            with self.assertRaisesRegex(DispatchRunWarning, "not an exact multiple"):
                run_policy_dispatch(
                    "policy_old_deprecated.pt",
                    parameters,
                    month=dense_month(),
                )


if __name__ == "__main__":
    unittest.main()
