from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from bess.core.common import load_system_config, make_bess_config
from bess.core.scenario_gen import DayData, MonthData
from bess.core.settings import DEFAULT_PARAMETERS
from bess.dispatch.dispatch_runner import DispatchRunWarning, run_policy_dispatch
from bess.evaluation.baselines import run_drl_policy, validate_dispatch_sampling


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


class ChargeFirstDayPolicy(CountingPolicy):
    def predict_action(self, _obs):
        action = -1.0 if self.calls < 96 else 0.0
        self.calls += 1
        return action


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
    def test_iq66_checkpoint_metadata_enables_peak_guard_without_changing_legacy_policy(self):
        cfg = load_system_config()
        cfg = make_bess_config(cfg, 1000.0, 500.0, cfg.P_target_user)
        cfg.dt = 0.25
        day_two_load = np.full(96, 100.0, dtype=np.float64)
        day_two_load[:2] = 800.0
        days = [
            DayData(
                load=np.full(96, 100.0, dtype=np.float64),
                pv=np.zeros(96, dtype=np.float64),
                day_type="working",
                weather="test",
                day_index=1,
                date_iso="2026-01-01",
            ),
            DayData(
                load=day_two_load,
                pv=np.zeros(96, dtype=np.float64),
                day_type="working",
                weather="test",
                day_index=2,
                date_iso="2026-01-02",
            ),
        ]
        month = MonthData(days=days, source="iq66-meta-test")

        legacy = ChargeFirstDayPolicy()
        guarded = ChargeFirstDayPolicy()
        guarded.meta.update({
            "peak_guard_enabled": True,
            "peak_guard_min_completed_days": 1,
            "peak_guard_deadband_kw": 1.0,
        })
        legacy_result = run_drl_policy(month, cfg, legacy, p_ref_kw=1500.0)
        guarded_result = run_drl_policy(month, cfg, guarded, p_ref_kw=1500.0)

        legacy_day_two_peak = float(np.max(legacy_result["p_grid_days"][1]))
        guarded_day_two_peak = float(np.max(guarded_result["p_grid_days"][1]))
        day_one_peak = float(np.max(guarded_result["p_grid_days"][0].reshape(-1, 2).mean(axis=1)))
        self.assertAlmostEqual(legacy_day_two_peak, 800.0)
        self.assertLessEqual(guarded_day_two_peak, day_one_peak + 1e-9)
        self.assertEqual(legacy_result["peak_guard_trigger_steps"], 0)
        self.assertGreater(guarded_result["peak_guard_trigger_steps"], 0)
        self.assertGreater(guarded_result["peak_guard_override_steps"], 0)
        self.assertEqual(guarded_result["peak_guard_unmet_steps"], 0)

    def test_legacy_policy_uses_96_decisions_and_1440_native_updates(self):
        policy = CountingPolicy()
        cfg = dense_config()
        initial_soc = cfg.SOC_min

        result = run_drl_policy(dense_month(), cfg, policy, p_ref_kw=1500.0)

        self.assertEqual(policy.calls, 96)
        self.assertEqual(len(result["p_grid_days"][0]), 1440)
        self.assertEqual(len(result["p_bess_days"][0]), 1440)
        self.assertEqual(len(result["soc_days"][0]), 1441)
        # BrainEnv action power is battery-side power, so SOC integrates it
        # directly; charge efficiency only affects the outside/grid power.
        expected_soc = initial_soc + 500.0 * (15.0 / 60.0) / cfg.E_cap
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

    def test_eye6_trace_is_exact_pre_action_running_peak_and_opt_in(self):
        policy = CountingPolicy()
        policy.meta["control_dt_minutes"] = 30.0
        policy.meta["native_steps_per_action"] = 2
        policy.predict_action = lambda _obs: 0.0
        cfg = load_system_config()
        cfg = make_bess_config(cfg, 1000.0, 500.0, cfg.P_target_user)
        cfg.dt = 0.25
        steps = 96
        load = np.full(steps, 200.0, dtype=np.float64)
        load[:2] = 100.0
        month = MonthData(days=[DayData(
            load=load,
            pv=np.zeros(steps, dtype=np.float64),
            day_type="working",
            weather="test",
            day_index=1,
            date_iso="2026-01-01",
        )], source="eye6-test")

        normal = run_drl_policy(month, cfg, policy, p_ref_kw=1000.0)
        self.assertNotIn("brain_eye6_running_peak_days", normal)

        visible = run_drl_policy(
            month,
            cfg,
            policy,
            p_ref_kw=1000.0,
            record_brain_eye6=True,
        )
        eye6 = visible["brain_eye6_running_peak_days"][0]
        self.assertEqual(len(eye6), steps)
        np.testing.assert_allclose(eye6[:2], [0.0, 0.0])
        np.testing.assert_allclose(eye6[2:4], [100.0, 100.0])
        np.testing.assert_allclose(eye6[4:6], [200.0, 200.0])

    def test_eye6_and_policy_episode_reset_at_fixed_month_boundary(self):
        policy = CountingPolicy()
        policy.meta["control_dt_minutes"] = 30.0
        policy.meta["native_steps_per_action"] = 2
        policy.predict_action = lambda _obs: 0.0
        reset_count = 0

        def reset_recurrent_state():
            nonlocal reset_count
            reset_count += 1

        policy.reset_recurrent_state = reset_recurrent_state
        cfg = load_system_config()
        cfg = make_bess_config(cfg, 1000.0, 500.0, cfg.P_target_user)
        cfg.dt = 0.25
        days = [
            DayData(
                load=np.full(96, 1000.0 if day_index == 1 else 100.0),
                pv=np.zeros(96),
                day_type="working",
                weather="test",
                day_index=day_index,
                date_iso=f"2026-01-{min(day_index, 31):02d}",
            )
            for day_index in range(1, 32)
        ]

        result = run_drl_policy(
            MonthData(days=days, source="eye6-boundary-test"),
            cfg,
            policy,
            p_ref_kw=1500.0,
            record_brain_eye6=True,
        )

        eye6_days = result["brain_eye6_running_peak_days"]
        self.assertEqual(len(eye6_days), 31)
        self.assertAlmostEqual(eye6_days[29][-1], 1000.0, places=3)
        self.assertEqual(eye6_days[30][0], 0.0)
        np.testing.assert_allclose(eye6_days[30][2:], 100.0, rtol=0.0, atol=1e-3)
        self.assertEqual(reset_count, 4)

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
            "bess.dispatch.dispatch_runner.load_policy",
            return_value=(policy, "ppo", policy.meta),
        ):
            result = run_policy_dispatch(
                "policy_old_deprecated.pt",
                parameters,
                month=dense_month(),
            )

        self.assertEqual(policy.calls, 96)
        self.assertEqual(len(result["days"][0]["grid"]), 1440)
        self.assertEqual(len(result["days"][0]["ppo_eye6_running_peak_kw"]), 1440)
        self.assertIn("15-minute decisions and 1-minute physics", result["warnings"][0])

    def test_dispatch_rejects_non_divisible_selected_data(self):
        policy = CountingPolicy()
        parameters = {
            **DEFAULT_PARAMETERS,
            "dt": str(4.0 / 60.0),
            "billing_mode": "2tc",
        }

        with patch(
            "bess.dispatch.dispatch_runner.load_policy",
            return_value=(policy, "ppo", policy.meta),
        ), self.assertRaisesRegex(DispatchRunWarning, "not an exact multiple"):
            run_policy_dispatch(
                "policy_old_deprecated.pt",
                parameters,
                month=dense_month(),
            )


if __name__ == "__main__":
    unittest.main()
