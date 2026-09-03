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
    constrain_charge_to_cheap_window,
    enforce_seen_peak_guard,
    make_brain_env,
    observation_array,
    step_brain_control,
)
from bess.core.common import load_system_config, make_bess_config
from bess.core.scenario_gen import DayData, MonthData
from bess.evaluation.baselines import run_drl_policy
from bess.evaluation.benchmark import _rolling_30_minute_average
from bess.training.runners.train_ppo_dataset import _collect_oracle_teacher_samples


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
    @staticmethod
    def _peak_guard(action, **overrides):
        arguments = {
            "net_load_kw": 140.0,
            "monthly_peak_kw": 100.0,
            "block_energy_kwh": 20.0,
            "block_elapsed_hours": 0.25,
            "timestep_hours": 0.25,
            "battery_power_kw": 100.0,
            "charge_efficiency": 0.8,
            "discharge_efficiency": 0.8,
            "enabled": True,
            "armed": True,
            "deadband_kw": 1.0,
        }
        arguments.update(overrides)
        return enforce_seen_peak_guard(action, **arguments)

    def test_iq66_peak_guard_raises_weak_discharge_to_meter_budget(self):
        decision = self._peak_guard(0.10)

        # First half used 20 kWh, leaving 30 kWh for the last 15 minutes:
        # allowed grid = 120 kW. Removing the 20 kW excess through 80%
        # efficiency requires 25 battery kW, or action 0.25.
        self.assertTrue(decision.triggered)
        self.assertTrue(decision.adjusted)
        self.assertAlmostEqual(decision.allowed_grid_kw, 120.0)
        self.assertAlmostEqual(decision.action, 0.25)

    def test_iq66_peak_guard_leaves_stronger_and_safe_discharge_alone(self):
        stronger = self._peak_guard(0.50)
        safe = self._peak_guard(
            0.10,
            net_load_kw=80.0,
            block_energy_kwh=0.0,
            block_elapsed_hours=0.0,
        )

        self.assertFalse(stronger.triggered)
        self.assertFalse(stronger.adjusted)
        self.assertEqual(stronger.action, 0.50)
        self.assertFalse(safe.triggered)
        self.assertFalse(safe.adjusted)
        self.assertEqual(safe.action, 0.10)

    def test_iq66_peak_guard_limits_charging_that_would_create_peak(self):
        decision = self._peak_guard(
            -0.50,
            net_load_kw=80.0,
            block_energy_kwh=0.0,
            block_elapsed_hours=0.0,
        )

        # There is 20 kW of outside charging headroom. At 80% charge
        # efficiency that is -16 battery kW, or action -0.16.
        self.assertTrue(decision.triggered)
        self.assertTrue(decision.adjusted)
        self.assertAlmostEqual(decision.action, -0.16)

    def test_iq66_peak_guard_stays_off_until_armed_with_nonzero_peak(self):
        disabled = self._peak_guard(-0.50, enabled=False)
        unarmed = self._peak_guard(-0.50, armed=False)
        empty_peak = self._peak_guard(-0.50, monthly_peak_kw=0.0)

        for decision in (disabled, unarmed, empty_peak):
            self.assertFalse(decision.triggered)
            self.assertFalse(decision.adjusted)
            self.assertEqual(decision.action, -0.50)
            self.assertIsNone(decision.allowed_grid_kw)

    def test_iq66_peak_guard_clamps_impossible_request_to_action_limit(self):
        decision = self._peak_guard(
            0.0,
            net_load_kw=1000.0,
            battery_power_kw=100.0,
        )

        self.assertTrue(decision.triggered)
        self.assertTrue(decision.adjusted)
        self.assertEqual(decision.action, 1.0)

    def test_iq66_peak_guard_rejects_bad_meter_and_physics_inputs(self):
        bad_cases = (
            {"monthly_peak_kw": -1.0},
            {"block_energy_kwh": -1.0},
            {"block_elapsed_hours": 0.5},
            {"timestep_hours": 0.3},
            {"battery_power_kw": 0.0},
            {"charge_efficiency": 0.0},
            {"discharge_efficiency": 1.1},
            {"deadband_kw": -1.0},
        )
        for overrides in bad_cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self._peak_guard(0.0, **overrides)

    def test_iq66_native_guard_rescues_second_half_of_held_action(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1000.0, 500.0, base.P_target_user)
        cfg.dt = 0.25
        day_one = DayData(
            load=np.full(96, 100.0, dtype=np.float64),
            pv=np.zeros(96, dtype=np.float64),
            day_type="working",
            weather="test",
            day_index=1,
            date_iso="2026-01-01",
        )
        day_two_load = np.full(96, 80.0, dtype=np.float64)
        day_two_load[1] = 140.0
        day_two = DayData(
            load=day_two_load,
            pv=np.zeros(96, dtype=np.float64),
            day_type="working",
            weather="test",
            day_index=2,
            date_iso="2026-01-02",
        )
        month = MonthData(days=[day_one, day_two], source="iq66-native-guard-test")
        env = make_brain_env(
            month,
            cfg,
            power_scale_kw=1000.0,
            initial_state_of_charge=0.50,
        )
        env.reset()

        first_day = step_brain_control(
            env,
            0.0,
            native_steps=96,
            peak_guard_enabled=True,
            peak_guard_min_completed_days=1,
            peak_guard_deadband_kw=1.0,
        )
        danger_block = step_brain_control(
            env,
            0.0,
            native_steps=2,
            peak_guard_enabled=True,
            peak_guard_min_completed_days=1,
            peak_guard_deadband_kw=1.0,
        )

        self.assertEqual(first_day.peak_guard_trigger_steps, 0)
        self.assertAlmostEqual(env.bess_world.meter_state.monthly_peak_kw, 100.0)
        self.assertAlmostEqual(danger_block.native_results[0].bess.physics.grid_import_kw, 80.0)
        self.assertAlmostEqual(danger_block.native_results[1].bess.physics.grid_import_kw, 120.0)
        self.assertEqual(danger_block.peak_guard_trigger_steps, 1)
        self.assertEqual(danger_block.peak_guard_override_steps, 1)
        self.assertEqual(danger_block.peak_guard_unmet_steps, 0)
        self.assertEqual(danger_block.requested_policy_action, 0.0)
        self.assertEqual(danger_block.applied_native_actions[0], 0.0)
        self.assertGreater(danger_block.applied_native_actions[1], 0.0)

    def test_iq66_native_guard_reports_soc_limited_unmet_peak(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1000.0, 500.0, base.P_target_user)
        cfg.dt = 0.25
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
                load=np.concatenate((np.asarray([80.0, 140.0]), np.full(94, 80.0))),
                pv=np.zeros(96, dtype=np.float64),
                day_type="working",
                weather="test",
                day_index=2,
                date_iso="2026-01-02",
            ),
        ]
        env = make_brain_env(MonthData(days=days, source="iq66-unmet-test"), cfg, power_scale_kw=1000.0)
        env.reset()
        step_brain_control(
            env,
            0.0,
            native_steps=96,
            peak_guard_enabled=True,
            peak_guard_min_completed_days=1,
        )

        danger_block = step_brain_control(
            env,
            0.0,
            native_steps=2,
            peak_guard_enabled=True,
            peak_guard_min_completed_days=1,
        )

        self.assertEqual(danger_block.peak_guard_trigger_steps, 1)
        self.assertEqual(danger_block.peak_guard_override_steps, 1)
        self.assertEqual(danger_block.peak_guard_unmet_steps, 1)
        self.assertAlmostEqual(env.bess_world.meter_state.monthly_peak_kw, 110.0)

    def test_iq66_native_guard_reports_unmet_when_strong_policy_hits_empty_battery(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1000.0, 500.0, base.P_target_user)
        cfg.dt = 0.25
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
                load=np.concatenate((np.asarray([80.0, 140.0]), np.full(94, 80.0))),
                pv=np.zeros(96, dtype=np.float64),
                day_type="working",
                weather="test",
                day_index=2,
                date_iso="2026-01-02",
            ),
        ]
        env = make_brain_env(MonthData(days=days, source="iq66-strong-unmet-test"), cfg, power_scale_kw=1000.0)
        env.reset()
        step_brain_control(
            env,
            0.0,
            native_steps=96,
            peak_guard_enabled=True,
            peak_guard_min_completed_days=1,
        )

        danger_block = step_brain_control(
            env,
            1.0,
            native_steps=2,
            peak_guard_enabled=True,
            peak_guard_min_completed_days=1,
        )

        self.assertEqual(danger_block.peak_guard_trigger_steps, 0)
        self.assertEqual(danger_block.peak_guard_override_steps, 0)
        self.assertEqual(danger_block.peak_guard_unmet_steps, 1)
        self.assertAlmostEqual(env.bess_world.meter_state.monthly_peak_kw, 110.0)

    def test_cheap_tariff_charge_constraint_blocks_only_noncheap_charging(self):
        cheap_steps = frozenset({0, 1, 2, 3})
        self.assertEqual(
            constrain_charge_to_cheap_window(
                -0.75,
                native_step_in_day=2,
                cheap_tariff_steps=cheap_steps,
                enabled=True,
            ),
            -0.75,
        )
        self.assertEqual(
            constrain_charge_to_cheap_window(
                -0.75,
                native_step_in_day=20,
                cheap_tariff_steps=cheap_steps,
                enabled=True,
            ),
            0.0,
        )
        self.assertEqual(
            constrain_charge_to_cheap_window(
                0.75,
                native_step_in_day=20,
                cheap_tariff_steps=cheap_steps,
                enabled=True,
            ),
            0.75,
        )
        self.assertEqual(
            constrain_charge_to_cheap_window(
                -0.75,
                native_step_in_day=20,
                cheap_tariff_steps=cheap_steps,
                enabled=False,
            ),
            -0.75,
        )

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
            day_index=1,
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
            day_index=1,
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

    def test_iq57_checkpoint_charges_only_during_cheap_tariff(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1000.0, 20.0, base.P_target_user)
        cfg.set_dt(0.25)
        steps = 96
        day = DayData(
            load=np.full(steps, 700.0, dtype=np.float64),
            pv=np.zeros(steps, dtype=np.float64),
            day_type="working",
            weather="test",
            day_index=1,
            date_iso="2026-01-01",
        )
        month = MonthData(days=[day], source="iq57-cheap-charge-test")

        class AlwaysChargeAgent:
            def __init__(self):
                self.meta = {
                    "obs_dim": OBSERVATION_DIM,
                    "control_dt_minutes": 15.0,
                    "battery_wear_cost": 0.0,
                    "charge_only_during_cheap_tariff": True,
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

        battery_power = np.asarray(result["p_bess_days"][0], dtype=np.float64)
        cheap_steps = np.asarray(cfg.OFF, dtype=np.int64)
        noncheap_mask = np.ones(steps, dtype=bool)
        noncheap_mask[cheap_steps] = False
        self.assertTrue((battery_power[cheap_steps] < 0.0).all())
        self.assertTrue((battery_power[noncheap_mask] == 0.0).all())
        self.assertEqual(result["tariff_blocked_charge_steps"], int(noncheap_mask.sum()))
        self.assertGreater(result["blocked_action_pct"], 0.0)

    def test_iq57_oracle_teacher_removes_noncheap_charge_lessons(self):
        base = load_system_config()
        cfg = make_bess_config(base, 10000.0, 20.0, base.P_target_user)
        cfg.set_dt(0.25)
        steps = 96
        day = DayData(
            load=np.full(steps, 700.0, dtype=np.float64),
            pv=np.zeros(steps, dtype=np.float64),
            day_type="working",
            weather="test",
            day_index=1,
            date_iso="2026-01-01",
        )
        month = MonthData(days=[day], source="iq57-oracle-filter-test")
        oracle_dispatch = [{
            "discharge": [0.0] * steps,
            "grid_charge": [10.0] * steps,
            "solar_charge": [0.0] * steps,
        }]

        observations, targets, rewards = _collect_oracle_teacher_samples(
            month,
            oracle_dispatch,
            cfg,
            power_scale_kw=1000.0,
            battery_wear_cost=0.0,
            native_steps=2,
        )

        self.assertEqual(observations.shape, (48, OBSERVATION_DIM))
        self.assertEqual(rewards.shape, (48, 3))
        self.assertTrue(np.isfinite(rewards).all())
        self.assertTrue((targets[:12] < 0.0).all())
        self.assertTrue((targets[12:] == 0.0).all())

    def test_pre_iq57_checkpoint_keeps_legacy_anytime_charging(self):
        base = load_system_config()
        cfg = make_bess_config(base, 1000.0, 20.0, base.P_target_user)
        cfg.set_dt(0.25)
        steps = 96
        day = DayData(
            load=np.full(steps, 700.0, dtype=np.float64),
            pv=np.zeros(steps, dtype=np.float64),
            day_type="working",
            weather="test",
            day_index=1,
            date_iso="2026-01-01",
        )
        month = MonthData(days=[day], source="legacy-charge-test")

        class LegacyAlwaysChargeAgent:
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
            LegacyAlwaysChargeAgent(),
            p_ref_kw=1000.0,
        )

        battery_power = np.asarray(result["p_bess_days"][0], dtype=np.float64)
        noncheap_mask = np.ones(steps, dtype=bool)
        noncheap_mask[np.asarray(cfg.OFF, dtype=np.int64)] = False
        self.assertTrue((battery_power[noncheap_mask] < 0.0).any())
        self.assertEqual(result["tariff_blocked_charge_steps"], 0)

    def test_seeded_update_stays_finite_and_checkpoint_is_compatible(self):
        agent = PPOAgent(
            obs_dim=OBSERVATION_DIM,
            seed=11,
            epochs=1,
            minibatch=8,
            hidden_size=96,
            initial_log_std=-0.8,
            actor_grad_clip=0.4,
            critic_grad_clip=0.8,
        )
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
            self.assertEqual(loaded.hidden_size, 96)
            self.assertEqual(loaded.meta, agent.meta)
            probe = np.zeros(OBSERVATION_DIM, dtype=np.float32)
            self.assertEqual(loaded.predict_action(probe), agent.predict_action(probe))


if __name__ == "__main__":
    unittest.main()
