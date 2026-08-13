"""Causal/declared-noisy SADRBC planning and baseline rollouts.

SADRBC never receives the exact future trajectory here. When portable real
weather forecasts are attached to DayData, the planner consumes their causal
next-1h/following-2h outputs and replans hourly. The battery rollout itself is
always executed by the canonical seven-eye ``BrainEnv``.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np

from bess.agents.sadrbc import phase3_strategic_plan
from bess.core.brain_runtime import (
    BrainTrajectoryRecorder,
    make_brain_env,
    native_steps_per_action,
    step_brain_control,
)
from bess.core.scenario_gen import MonthData


DEFAULT_FORECAST_SEED = 13_0013
DEFAULT_LOAD_SIGMA = 0.05
DEFAULT_PV_SIGMA = 0.15
DEFAULT_RHO = 0.90
DEFAULT_RESIDUAL_LIMIT = 0.20


@dataclass
class SADRBCForecastSpec:
    seed: int = DEFAULT_FORECAST_SEED
    load_sigma: float = DEFAULT_LOAD_SIGMA
    pv_sigma: float = DEFAULT_PV_SIGMA
    rho: float = DEFAULT_RHO
    replan_minutes: int = 60

    def public(self, mode: str) -> dict:
        return {
            "mode": mode,
            "seed": int(self.seed),
            "load_sigma": float(self.load_sigma),
            "pv_sigma": float(self.pv_sigma),
            "rho": float(self.rho),
            "replan_minutes": int(self.replan_minutes),
            "exact_future_actuals_passed_to_sadrbc": False,
            "synthetic_forecast_uses_actual_as_latent_truth": "declared_noisy" in mode,
        }


@dataclass
class BaselineRollout:
    actions: list[np.ndarray]
    decision_costs: list[float]
    p_grid_days: list[np.ndarray]
    soc_days: list[np.ndarray]
    p_bess_days: list[np.ndarray]
    mode: str
    spec: SADRBCForecastSpec


def _stable_rng(seed: int, day) -> np.random.Generator:
    identity = f"{int(seed)}|{day.date_iso or ''}|{int(day.day_index)}".encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _ar1_noisy(actual: np.ndarray, sigma: float, rho: float,
               rng: np.random.Generator) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    if sigma <= 0.0:
        return actual.copy()
    errors = np.empty(len(actual), dtype=np.float64)
    errors[0] = sigma * rng.standard_normal()
    innovation = sigma * np.sqrt(max(0.0, 1.0 - rho * rho))
    for index in range(1, len(actual)):
        errors[index] = rho * errors[index - 1] + innovation * rng.standard_normal()
    return np.maximum(0.0, actual * (1.0 + errors))


def _declared_noisy_days(month: MonthData, spec: SADRBCForecastSpec):
    rows = []
    for day in month.days:
        rng = _stable_rng(spec.seed, day)
        rows.append((
            _ar1_noisy(day.load, spec.load_sigma, spec.rho, rng),
            _ar1_noisy(day.pv, spec.pv_sigma, spec.rho, rng),
        ))
    return rows


def _forecast_mode(month: MonthData) -> str:
    return (
        "real_weather_causal_plus_declared_noisy_tail"
        if month.days and all(getattr(day, "forecast", None) is not None for day in month.days)
        else "declared_noisy_ar1"
    )


def _forecast_at(day, noisy_load: np.ndarray, noisy_pv: np.ndarray,
                 origin: int, p_ref_kw: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the information set available at ``origin`` without future truth."""
    load = noisy_load.copy()
    pv = noisy_pv.copy()
    load[: origin + 1] = day.load[: origin + 1]
    pv[: origin + 1] = day.pv[: origin + 1]
    causal = getattr(day, "forecast", None)
    if causal is None:
        return load, pv

    causal = np.asarray(causal, dtype=np.float64)
    if causal.shape != (len(day.load), 4):
        raise ValueError(
            f"real forecast for {day.date_iso} must have shape {(len(day.load), 4)}"
        )
    per_hour = max(1, round(len(day.load) / 24))
    eff_1h, pv_1h, eff_2h, pv_2h = causal[origin] * float(p_ref_kw)
    first_end = min(len(load), origin + 1 + per_hour)
    second_end = min(len(load), first_end + 2 * per_hour)
    pv[origin + 1:first_end] = pv_1h
    load[origin + 1:first_end] = eff_1h + pv_1h
    pv[first_end:second_end] = pv_2h
    load[first_end:second_end] = eff_2h + pv_2h
    return load, pv


def build_sadrbc_forecast_baseline(
    month: MonthData,
    cfg,
    *,
    p_ref_kw: float,
    control_dt_minutes: float | None = None,
    soc_init: float | None = None,
    spec: SADRBCForecastSpec | None = None,
) -> BaselineRollout:
    """Plan with SADRBC forecasts and execute every battery step in ``BrainEnv``."""
    spec = spec or SADRBCForecastSpec()
    cfg_copy = copy.copy(cfg)
    initial_soc = float(cfg_copy.SOC_eod if soc_init is None else soc_init)
    env = make_brain_env(
        month,
        cfg_copy,
        power_scale_kw=p_ref_kw,
        initial_state_of_charge=initial_soc,
    )
    env.reset()
    recorder = BrainTrajectoryRecorder(month, initial_soc)
    native_hold_steps = native_steps_per_action(cfg_copy.dt, control_dt_minutes)

    noisy_days = _declared_noisy_days(month, spec)
    actions = [np.zeros(len(day.load), dtype=np.float64) for day in month.days]
    decision_costs: list[float] = []
    schedule: np.ndarray | None = None
    done = False
    replan_steps = max(1, round(spec.replan_minutes / (cfg_copy.dt * 60.0)))
    steps_per_day = len(month.days[0].load)

    while not done:
        flat_index = env.bess_world.timestep_index
        day_index = flat_index // steps_per_day
        native_step = flat_index % steps_per_day
        day = month.days[day_index]

        if schedule is None or native_step % replan_steps == 0:
            fc_load, fc_pv = _forecast_at(
                day, *noisy_days[day_index], native_step, p_ref_kw
            )
            next_type = (
                month.days[day_index + 1].day_type
                if day_index + 1 < len(month.days)
                else "working"
            )
            plan = phase3_strategic_plan(
                fc_load,
                fc_pv,
                day.day_type,
                next_type,
                PMax_running_month=env.bess_world.meter_state.monthly_peak_kw,
                plan_timestamp=day.date_iso,
                cfg=cfg_copy,
                t_start=native_step,
                SOC_start=env.bess_world.state_of_charge,
            )
            schedule = np.asarray(plan["execution"]["p_plan"], dtype=np.float64)

        action = float(np.clip(
            schedule[native_step] / max(cfg_copy.P_rated_nominal, 1e-9),
            -1.0,
            1.0,
        ))
        end = min(len(actions[day_index]), native_step + native_hold_steps)
        actions[day_index][native_step:end] = action

        transition = step_brain_control(
            env,
            action,
            native_steps=native_hold_steps,
            recorder=recorder,
        )
        decision_costs.append(float(sum(
            result.bess.cost.operating_cost_vnd for result in transition.native_results
        )))
        done = transition.done

        if done:
            schedule = None
        else:
            next_flat_index = env.bess_world.timestep_index
            next_day_index = next_flat_index // steps_per_day
            if next_day_index != day_index:
                schedule = None

    return BaselineRollout(
        actions=actions,
        decision_costs=decision_costs,
        p_grid_days=[row.copy() for row in recorder.grid_import_days],
        soc_days=[row.copy() for row in recorder.state_of_charge_days],
        p_bess_days=[row.copy() for row in recorder.battery_power_days],
        mode=_forecast_mode(month),
        spec=spec,
    )


def rollout_activity(result: dict, dt_hours: float) -> dict:
    rows = [np.asarray(row, dtype=np.float64) for row in result["p_bess_days"]]
    throughput = float(sum(np.sum(np.abs(row)) * dt_hours for row in rows))
    total_rows = sum(len(row) for row in rows)
    mean_abs = throughput / max(dt_hours * total_rows, 1e-12)
    soc_values = np.concatenate([
        np.asarray(row, dtype=np.float64) for row in result["soc_days"]
    ]) if result["soc_days"] else np.asarray([0.0])
    return {
        "throughput_kwh": throughput,
        "mean_abs_p_bess_kw": mean_abs,
        "soc_span_pct": float((soc_values.max() - soc_values.min()) * 100.0),
    }
