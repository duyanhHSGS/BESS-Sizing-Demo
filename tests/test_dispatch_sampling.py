from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from bess.core.common import load_system_config, make_bess_config
from bess.core.scenario_gen import DayData, MonthData
from bess.core.settings import DEFAULT_PARAMETERS, PPO_PEAK_GUARD_FIRST_DAY_ARM_HOUR
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


class AlwaysDischargePolicy(CountingPolicy):
    def predict_action(self, _obs):
        self.calls += 1
        return 1.0


class IdlePolicy(CountingPolicy):
    def predict_action(self, _obs):
        self.calls += 1
        return 0.0


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
    def test_iq67_checkpoint_metadata_enforces_0600_full_soc_and_keeps_legacy_off(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1250.0, 450.0, base.P_target_user)
        cfg.dt = 0.25
        month = MonthData(
            days=[
                DayData(
                    load=np.full(96, 40.0, dtype=np.float64),
                    pv=np.zeros(96, dtype=np.float64),
                    day_type="working",
                    weather="test",
                    day_index=1,
                    date_iso="2026-01-01",
                )
            ],
            source="iq67-checkpoint-meta-test",
        )
        legacy = AlwaysDischargePolicy()
        guarded = AlwaysDischargePolicy()
        guarded.meta.update({
            "soc_deadline_enabled": True,
            "soc_deadline_hour": 6.0,
            "soc_deadline_shortfall_penalty_vnd": 128_250_000.0,
        })

        legacy_result = run_drl_policy(month, cfg, legacy, p_ref_kw=1500.0)
        guarded_result = run_drl_policy(month, cfg, guarded, p_ref_kw=1500.0)

        self.assertAlmostEqual(legacy_result["soc_days"][0][24], cfg.SOC_min)
        self.assertAlmostEqual(guarded_result["soc_days"][0][24], cfg.SOC_max, places=10)
        self.assertEqual(legacy_result["soc_deadline_trigger_steps"], 0)
        self.assertGreater(guarded_result["soc_deadline_override_steps"], 0)
        self.assertEqual(guarded_result["soc_deadline_unmet_count"], 0)

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

    def test_iq68_checkpoint_wakes_peak_guard_at_configured_cheap_window_end(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1250.0, 450.0, base.P_target_user)
        cfg.dt = 0.25
        load = np.full(96, 100.0, dtype=np.float64)
        load[24:26] = 400.0
        month = MonthData(
            days=[
                DayData(
                    load=load,
                    pv=np.zeros(96, dtype=np.float64),
                    day_type="working",
                    weather="test",
                    day_index=31,
                    date_iso="2026-02-01",
                )
            ],
            source="iq68-first-day-meta-test",
        )
        common_meta = {
            "peak_guard_enabled": True,
            "peak_guard_min_completed_days": 1,
            "peak_guard_deadband_kw": 1.0,
            "soc_deadline_enabled": True,
            "soc_deadline_hour": 6.0,
            "soc_deadline_shortfall_penalty_vnd": 128_250_000.0,
        }
        iq67 = IdlePolicy()
        iq67.meta.update(common_meta)
        iq68 = IdlePolicy()
        iq68.meta.update(common_meta)
        iq68.meta["peak_guard_first_day_arm_at_cheap_end"] = True

        iq67_result = run_drl_policy(month, cfg, iq67, p_ref_kw=1500.0)
        iq68_result = run_drl_policy(month, cfg, iq68, p_ref_kw=1500.0)

        iq67_danger_block = np.mean(iq67_result["p_grid_days"][0][24:26])
        iq68_danger_block = np.mean(iq68_result["p_grid_days"][0][24:26])
        iq68_cheap_peak = np.max(
            iq68_result["p_grid_days"][0][:24].reshape(-1, 2).mean(axis=1)
        )
        self.assertAlmostEqual(iq67_danger_block, 400.0)
        self.assertLessEqual(iq68_danger_block, iq68_cheap_peak + 1.0 + 1e-9)
        self.assertEqual(iq67_result["peak_guard_trigger_steps"], 0)
        self.assertGreater(iq68_result["peak_guard_trigger_steps"], 0)
        self.assertAlmostEqual(iq68_result["soc_days"][0][24], cfg.SOC_max, places=10)

    def test_iq68_wake_time_follows_a_nonstandard_cheap_window(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1250.0, 450.0, base.P_target_user)
        cfg.off_windows = "00:00-05:00"
        cfg.SOC_min = 0.89
        cfg.set_dt(0.25)
        load = np.full(96, 100.0, dtype=np.float64)
        load[20:22] = 110.0
        month = MonthData(
            days=[
                DayData(
                    load=load,
                    pv=np.zeros(96, dtype=np.float64),
                    day_type="working",
                    weather="test",
                    day_index=31,
                    date_iso="2026-02-01",
                )
            ],
            source="iq68-custom-cheap-window-test",
        )
        policy = IdlePolicy()
        policy.meta.update({
            "peak_guard_enabled": True,
            "peak_guard_min_completed_days": 1,
            "peak_guard_first_day_arm_at_cheap_end": True,
            "peak_guard_deadband_kw": 1.0,
            "soc_deadline_enabled": True,
            "soc_deadline_hour": cfg.OFF_PEAK_END_STEP * cfg.dt,
            "soc_deadline_shortfall_penalty_vnd": 128_250_000.0,
        })

        result = run_drl_policy(month, cfg, policy, p_ref_kw=1500.0)
        danger_block = np.mean(result["p_grid_days"][0][20:22])
        cheap_peak = np.max(
            result["p_grid_days"][0][:20].reshape(-1, 2).mean(axis=1)
        )

        self.assertEqual(cfg.OFF_PEAK_END_STEP, 20)
        self.assertGreater(result["peak_guard_trigger_steps"], 0)
        self.assertLessEqual(danger_block, cheap_peak + 1.0 + 1e-9)

    def test_iq71_checkpoint_sleeps_before_noon_and_wakes_exactly_at_noon(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1250.0, 450.0, base.P_target_user)
        cfg.set_dt(0.25)
        load = np.full(96, 100.0, dtype=np.float64)
        load[40:42] = 350.0  # 10:00: police must still sleep.
        load[48:50] = 450.0  # 12:00: police must wake immediately.
        month = MonthData(
            days=[DayData(
                load=load,
                pv=np.zeros(96, dtype=np.float64),
                day_type="working",
                weather="test",
                day_index=1,
                date_iso="2026-01-01",
            )],
            source="iq71-noon-wake-test",
        )
        policy = IdlePolicy()
        policy.meta.update({
            "peak_guard_enabled": True,
            "peak_guard_min_completed_days": 1,
            "peak_guard_first_day_arm_hour": PPO_PEAK_GUARD_FIRST_DAY_ARM_HOUR,
            "peak_guard_deadband_kw": 1.0,
            "soc_deadline_enabled": True,
            "soc_deadline_hour": 6.0,
            "soc_deadline_shortfall_penalty_vnd": 128_250_000.0,
        })

        result = run_drl_policy(month, cfg, policy, p_ref_kw=1500.0)
        ten_am_block = float(np.mean(result["p_grid_days"][0][40:42]))
        noon_block = float(np.mean(result["p_grid_days"][0][48:50]))

        self.assertEqual(PPO_PEAK_GUARD_FIRST_DAY_ARM_HOUR, 12.0)
        self.assertAlmostEqual(ten_am_block, 350.0)
        self.assertLessEqual(noon_block, 351.0 + 1e-9)
        self.assertGreater(result["peak_guard_trigger_steps"], 0)
        self.assertGreater(result["peak_guard_override_steps"], 0)

    def test_iq71_noon_metadata_overrides_legacy_cheap_end_wake_flag(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1250.0, 450.0, base.P_target_user)
        cfg.set_dt(0.25)
        load = np.full(96, 100.0, dtype=np.float64)
        load[24:26] = 350.0  # 06:00 would be guarded if the legacy flag won.
        load[48:50] = 450.0
        month = MonthData(
            days=[DayData(
                load=load,
                pv=np.zeros(96, dtype=np.float64),
                day_type="working",
                weather="test",
                day_index=1,
                date_iso="2026-01-01",
            )],
            source="iq71-metadata-priority-test",
        )
        policy = IdlePolicy()
        policy.meta.update({
            "peak_guard_enabled": True,
            "peak_guard_min_completed_days": 1,
            "peak_guard_first_day_arm_hour": 12.0,
            "peak_guard_first_day_arm_at_cheap_end": True,
            "peak_guard_deadband_kw": 1.0,
            "soc_deadline_enabled": True,
            "soc_deadline_hour": 6.0,
            "soc_deadline_shortfall_penalty_vnd": 128_250_000.0,
        })

        result = run_drl_policy(month, cfg, policy, p_ref_kw=1500.0)

        self.assertAlmostEqual(float(np.mean(result["p_grid_days"][0][24:26])), 350.0)
        self.assertLessEqual(float(np.mean(result["p_grid_days"][0][48:50])), 351.0 + 1e-9)

    def test_iq71_noon_wake_restarts_on_each_fixed_30_day_bucket(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1250.0, 450.0, base.P_target_user)
        cfg.set_dt(0.25)

        def bucket_day(day_index: int, date_iso: str) -> DayData:
            load = np.full(96, 100.0, dtype=np.float64)
            load[40:42] = 350.0
            load[48:50] = 450.0
            return DayData(
                load=load,
                pv=np.zeros(96, dtype=np.float64),
                day_type="working",
                weather="test",
                day_index=day_index,
                date_iso=date_iso,
            )

        month = MonthData(
            days=[bucket_day(1, "2026-01-01"), bucket_day(31, "2026-01-31")],
            source="iq71-fixed-bucket-reset-test",
        )
        policy = IdlePolicy()
        policy.meta.update({
            "peak_guard_enabled": True,
            "peak_guard_min_completed_days": 1,
            "peak_guard_first_day_arm_hour": 12.0,
            "peak_guard_deadband_kw": 1.0,
            "soc_deadline_enabled": True,
            "soc_deadline_hour": 6.0,
            "soc_deadline_shortfall_penalty_vnd": 128_250_000.0,
        })

        result = run_drl_policy(month, cfg, policy, p_ref_kw=1500.0)

        self.assertEqual(len(result["p_grid_days"]), 2)
        for day_grid in result["p_grid_days"]:
            self.assertAlmostEqual(float(np.mean(day_grid[40:42])), 350.0)
            self.assertLessEqual(float(np.mean(day_grid[48:50])), 351.0 + 1e-9)
        self.assertGreaterEqual(result["peak_guard_trigger_steps"], 2)

    def test_iq71_rejects_invalid_checkpoint_wake_hours(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1250.0, 450.0, base.P_target_user)
        cfg.set_dt(0.25)
        month = MonthData(
            days=[DayData(
                load=np.full(96, 100.0, dtype=np.float64),
                pv=np.zeros(96, dtype=np.float64),
                day_type="working",
                weather="test",
                day_index=1,
                date_iso="2026-01-01",
            )],
            source="iq71-invalid-hour-test",
        )

        for bad_hour in (-1.0, 24.0, float("nan"), 12.1):
            with self.subTest(bad_hour=bad_hour):
                policy = IdlePolicy()
                policy.meta.update({
                    "peak_guard_enabled": True,
                    "peak_guard_first_day_arm_hour": bad_hour,
                })
                with self.assertRaises(ValueError):
                    run_drl_policy(month, cfg, policy, p_ref_kw=1500.0)

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
