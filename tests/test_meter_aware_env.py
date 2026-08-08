from __future__ import annotations

import unittest

import numpy as np

from bess.core.bess_env import (
    BESSEnv,
    FORECAST_OBSERVATION_DIM,
    NORMAL_OBSERVATION_SCHEMA,
    REACTIVE_OBSERVATION_DIM,
    normal_observation_compatibility_error,
)
from bess.core.common import (
    TOU_RULES,
    load_system_config,
    make_bess_config,
    score_month,
    score_operating_month,
)
from bess.core.scenario_gen import DayData, MonthData
from bess.core.settings import PPO_GAMMA
from bess.evaluation.policy_diagnostics import monthly_policy_diagnostics, run_cheap_window_acceptance
from bess.training.training_common import heldout_calendar_split


def _case(minutes: float, *, forecast: bool = False):
    steps = int(round(1440.0 / minutes))
    base = load_system_config()
    cfg = make_bess_config(base, 1250.0, 450.0, base.P_target_user)
    cfg.set_dt(minutes / 60.0)
    day = DayData(
        load=np.full(steps, 200.0),
        pv=np.zeros(steps),
        day_type="working",
        weather="test",
        day_index=0,
        date_iso="2026-01-05",
    )
    if forecast:
        day.forecast = np.zeros((steps, 4), dtype=np.float32)
    return cfg, MonthData(days=[day], source="meter_test")


class MeterObservationTests(unittest.TestCase):
    def test_regular_ppo_default_uses_undiscounted_money(self):
        self.assertEqual(PPO_GAMMA, 1.0)

    def test_phase_and_partial_contribution_are_resolution_independent(self):
        for minutes, last_phase in ((15.0, 1.0), (1.0, 1.0 / 29.0), (0.5, 1.0 / 59.0), (30.0, 0.0)):
            with self.subTest(minutes=minutes):
                cfg, month = _case(minutes)
                env = BESSEnv(cfg, reference_power_kw=1000.0)
                observation = env.reset(month, initial_state_of_charge=0.5)
                self.assertEqual(len(observation), REACTIVE_OBSERVATION_DIM)
                self.assertEqual(observation[13], 0.0)
                self.assertEqual(observation[14], 0.0)
                observation, _, _, _ = env.step(0.0)
                if minutes == 30.0:
                    self.assertEqual(observation[13], 0.0)
                    self.assertEqual(observation[14], 0.0)
                else:
                    self.assertAlmostEqual(float(observation[13]), last_phase, places=6)
                    self.assertAlmostEqual(float(observation[14]), 200.0 / env.samples_per_demand_block / 1000.0, places=6)

    def test_block_reset_control_hold_forecast_and_extra_dimensions(self):
        cfg, month = _case(15.0, forecast=True)
        env = BESSEnv(
            cfg,
            reference_power_kw=1000.0,
            forecast_enabled=True,
            control_interval_minutes=30.0,
            extra_observation_dimensions=1,
        )
        observation = env.reset(month, initial_state_of_charge=0.5)
        self.assertEqual(len(observation), FORECAST_OBSERVATION_DIM + 1)
        observation, _, _, info = env.step(0.0)
        self.assertEqual(info["native_rows"], 2)
        self.assertEqual(observation[13], 0.0)
        self.assertEqual(observation[14], 0.0)
        np.testing.assert_array_equal(observation[15:19], np.zeros(4))

    def test_legacy_normal_checkpoint_is_rejected_without_padding(self):
        error = normal_observation_compatibility_error("ppo", {"obs_dim": 13})
        self.assertIn("Retrain", error)
        self.assertIsNone(normal_observation_compatibility_error("ppo2", {"obs_dim": 17}))
        self.assertIsNone(normal_observation_compatibility_error("ppo", {
            "observation_schema": NORMAL_OBSERVATION_SCHEMA,
            "obs_variant": "base",
            "obs_dim": REACTIVE_OBSERVATION_DIM,
            "battery_wear_cost": 500.0,
            "deployment_acceptance_passed": True,
            "test_saving_pct": 1.0,
        }))
        self.assertIsNone(normal_observation_compatibility_error("ppo", {
            "observation_schema": NORMAL_OBSERVATION_SCHEMA,
            "obs_variant": "base",
            "obs_dim": REACTIVE_OBSERVATION_DIM,
            "battery_wear_cost": 500.0,
            "cheap_window_acceptance_passed": False,
            "deployment_acceptance_passed": True,
            "test_saving_pct": 1.0,
        }))
        self.assertIn("does not beat No-BESS", normal_observation_compatibility_error("ppo", {
            "observation_schema": NORMAL_OBSERVATION_SCHEMA,
            "obs_variant": "base",
            "obs_dim": REACTIVE_OBSERVATION_DIM,
            "battery_wear_cost": 500.0,
            "deployment_acceptance_passed": True,
            "test_saving_pct": -0.1,
        }))

    def test_wear_cost_is_exact_and_non_negative(self):
        cfg, month = _case(15.0)
        cfg.battery_wear_cost_vnd_per_kwh = 500.0
        env = BESSEnv(cfg)
        self.assertEqual(env.degradation_cost_vnd_per_kwh, 500.0)
        power = np.full(96, -100.0)
        score = score_operating_month(
            [np.full(96, 200.0)], [power], cfg, days=month.days
        )
        self.assertEqual(score["throughput_kwh"], 2400.0)
        self.assertEqual(score["wear_cost_vnd"], 1_200_000.0)
        with self.assertRaisesRegex(ValueError, "finite and >= 0"):
            BESSEnv(cfg, degradation_cost_vnd_per_kwh=-1.0)

    def test_potential_shaping_uses_constant_mid_reference_and_zero_terminal_potential(self):
        cfg, first_month = _case(15.0)
        _, second_month = _case(15.0)
        second_month.days[0].day_index = 1
        second_month.days[0].date_iso = "2026-01-06"
        month = MonthData(days=[first_month.days[0], second_month.days[0]], source="boundary")
        env = BESSEnv(cfg, reference_power_kw=1000.0, discount_factor=0.9)
        env.reset(month, initial_state_of_charge=0.5)
        for _ in range(95):
            env.step(0.0)
        _, _, done, info = env.step(0.0)
        self.assertFalse(done)
        stored = (0.5 - cfg.SOC_min) * cfg.E_cap * cfg.eta_dis
        expected_nonterminal = stored * cfg.price_mid * (0.9 - 1.0)
        self.assertAlmostEqual(info["shaping"], expected_nonterminal, places=6)

        terminal = BESSEnv(cfg, reference_power_kw=1000.0, discount_factor=0.9)
        terminal.reset(first_month, initial_state_of_charge=0.5)
        for _ in range(95):
            terminal.step(0.0)
        _, _, done, info = terminal.step(0.0)
        self.assertTrue(done)
        self.assertAlmostEqual(info["shaping"], -stored * cfg.price_mid, places=6)

    def test_two_day_lab_constants_and_relative_gate(self):
        class IdleAgent:
            meta = {"obs_variant": "base", "control_dt_minutes": 15.0, "gamma": 0.995}

            @staticmethod
            def predict_action(_observation):
                return 0.0

        cfg, _ = _case(15.0)
        result = run_cheap_window_acceptance(IdleAgent(), cfg, reference_power_kw=1000.0)
        self.assertAlmostEqual(result["required_cheap_energy_kwh"], 972.222222, places=5)
        self.assertAlmostEqual(result["required_average_charge_kw"], 162.037037, places=5)
        self.assertEqual(result["peak_safe_charge_kw"], 250.0)
        self.assertAlmostEqual(result["avoidable_normal_charge_limit_kwh"], 48.611111, places=5)
        self.assertTrue(result["passed"])
        self.assertFalse(result["deployment_gate"])

    def test_calendar_month_billing_and_environment_peak_reset(self):
        cfg, first = _case(15.0)
        _, second = _case(15.0)
        first_day = first.days[0]
        second_day = second.days[0]
        first_day.day_index = 1
        first_day.date_iso = "2026-04-30"
        first_day.load[:] = 400.0
        second_day.day_index = 2
        second_day.date_iso = "2026-05-01"
        second_day.load[:] = 100.0
        month = MonthData(days=[first_day, second_day], source="calendar_boundary")

        scored = score_month(
            [np.full(96, 400.0), np.full(96, 100.0)],
            cfg,
            days=month.days,
        )
        self.assertEqual(scored["month_count"], 2)
        self.assertAlmostEqual(scored["demand_cost_vnd"], 500.0 * cfg.T_cap, places=6)

        env = BESSEnv(cfg, reference_power_kw=1000.0, initial_running_peak_kw=50.0)
        observation = env.reset(month, initial_state_of_charge=cfg.SOC_eod)
        for _ in range(96):
            observation, _, done, _ = env.step(0.0)
        self.assertFalse(done)
        self.assertEqual(env.current_day_index, 1)
        self.assertAlmostEqual(env.running_monthly_peak_kw, 50.0, places=6)
        self.assertAlmostEqual(float(observation[8]), 0.05, places=6)
        self.assertAlmostEqual(float(observation[11]), 0.0, places=6)

    def test_final_soc_reachability_constraint_prevents_free_terminal_depletion(self):
        cfg, month = _case(15.0)
        cfg.SOC_eod = 0.50
        env = BESSEnv(cfg, reference_power_kw=1000.0)
        env.reset(month, initial_state_of_charge=0.20)
        forced_kwh = 0.0
        done = False
        while not done:
            _, _, done, info = env.step(0.0)
            forced_kwh += info["final_soc_forced_charge_kwh"]
        self.assertGreater(forced_kwh, 0.0)
        self.assertGreaterEqual(env.state_of_charge, cfg.SOC_eod - 1e-9)

    def test_heldout_split_prefers_complete_calendar_months(self):
        days = []
        start = np.datetime64("2026-04-01")
        end = np.datetime64("2026-07-02")
        day_index = 0
        for stamp in np.arange(start, end):
            text = np.datetime_as_string(stamp, unit="D")
            if text in {"2026-05-15", "2026-06-15"}:
                continue
            day_index += 1
            days.append(DayData(
                load=np.full(96, 100.0),
                pv=np.zeros(96),
                day_type="working",
                weather="test",
                day_index=day_index,
                date_iso=text,
            ))
        train, validation, test, meta = heldout_calendar_split(days, 30, 30)
        self.assertEqual(meta["mode"], "calendar_month_buckets")
        self.assertEqual(meta["validation_months"], ["2026-05"])
        self.assertEqual(meta["test_months"], ["2026-06"])
        self.assertEqual(validation[0].date_iso, "2026-05-01")
        self.assertEqual(validation[-1].date_iso, "2026-05-31")
        self.assertEqual(test[0].date_iso, "2026-06-01")
        self.assertEqual(test[-1].date_iso, "2026-06-30")
        self.assertEqual(train[-1].date_iso, "2026-04-30")
        self.assertEqual(meta["ignored_edge_days"], 1)

    def test_diagnostics_separate_pv_and_apply_sunday_rule(self):
        cfg, month = _case(15.0)
        day = month.days[0]
        day.date_iso = "2026-01-04"
        expensive_step = min(set(cfg.W1) | set(cfg.W2))
        pv_step = 48
        power = np.zeros(96)
        power[expensive_step] = -100.0
        power[pv_step] = -50.0
        day.pv[pv_step] = 250.0
        grid = np.maximum(day.load - day.pv - power, 0.0)
        soc = np.full(97, 0.5)
        old_rule = TOU_RULES["sunday_no_peak"]
        TOU_RULES["sunday_no_peak"] = True
        try:
            diagnostic = monthly_policy_diagnostics(month, {
                "p_grid_days": [grid], "p_bess_days": [power], "soc_days": [soc],
            }, cfg)[0]
        finally:
            TOU_RULES["sunday_no_peak"] = old_rule
        self.assertEqual(diagnostic["expensive_grid_charge_kwh"], 0.0)
        self.assertEqual(diagnostic["normal_grid_charge_kwh"], 25.0)
        self.assertEqual(diagnostic["pv_charge_kwh"], 12.5)

    def test_avoidable_charge_moves_only_into_later_cheap_capacity(self):
        cfg, first = _case(15.0)
        _, second = _case(15.0)
        second.days[0].day_index = 1
        second.days[0].date_iso = "2026-01-06"
        month = MonthData(days=[first.days[0], second.days[0]], source="causal_shift")
        powers = [np.zeros(96), np.zeros(96)]
        powers[0][92] = -100.0
        grids = [np.maximum(day.load - day.pv - power, 0.0) for day, power in zip(month.days, powers)]
        socs = [np.full(97, 0.5), np.full(97, 0.5)]
        diagnostic = monthly_policy_diagnostics(month, {
            "p_grid_days": grids, "p_bess_days": powers, "soc_days": socs,
        }, cfg, initial_running_peak_kw=350.0)[0]
        self.assertEqual(diagnostic["normal_grid_charge_kwh"], 25.0)
        self.assertEqual(diagnostic["avoidable_normal_charge_kwh"], 25.0)


if __name__ == "__main__":
    unittest.main()
