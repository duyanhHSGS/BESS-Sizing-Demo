"""Causal/declared-noisy SADRBC baseline and GrePRO residual environment.

SADRBC never receives the exact future trajectory here.  When portable real
weather forecasts are attached to DayData, the planner consumes their causal
next-1h/following-2h outputs and replans hourly.  The remaining horizon (or
the entire horizon when no real artifact exists) uses explicitly declared,
deterministic AR(1) forecast errors.
"""
from __future__ import annotations

import copy
import hashlib
import threading
from dataclasses import dataclass

import numpy as np

from bess.core.bess_env import BESSEnv
from bess.core.scenario_gen import MonthData
from bess.agents.sadrbc import phase3_strategic_plan


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
    d_run_init_kw: float | None = None,
    spec: SADRBCForecastSpec | None = None,
) -> BaselineRollout:
    spec = spec or SADRBCForecastSpec()
    cfg_copy = copy.copy(cfg)
    env = BESSEnv(
        cfg_copy,
        p_ref_kw=p_ref_kw,
        control_dt_minutes=control_dt_minutes,
        record_trajectory=True,
    )
    env.d_run_init = float(
        d_run_init_kw if d_run_init_kw is not None else cfg_copy.P_target_user
    )
    observation = env.reset(month, soc_init=soc_init)
    del observation
    noisy_days = _declared_noisy_days(month, spec)
    actions = [np.zeros(len(day.load), dtype=np.float64) for day in month.days]
    decision_costs: list[float] = []
    schedule = None
    done = False
    replan_steps = max(1, round(spec.replan_minutes / (cfg_copy.dt * 60.0)))

    while not done:
        day_index = env.day
        native_step = env.t
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
                PMax_running_month=env.d_run,
                plan_timestamp=day.date_iso,
                cfg=cfg_copy,
                t_start=native_step,
                SOC_start=env.soc,
            )
            schedule = np.asarray(plan["execution"]["p_plan"], dtype=np.float64)
        action = float(np.clip(
            schedule[native_step] / max(cfg_copy.P_rated_nominal, 1e-9), -1.0, 1.0
        ))
        end = min(len(actions[day_index]), native_step + env.native_steps_per_action)
        actions[day_index][native_step:end] = action
        _, _, done, info = env.step(action)
        decision_costs.append(float(
            info["energy_cost"] + info["peak_pen"] + info["deg_cost"]
        ))
        if done or env.day != day_index:
            schedule = None
            if (
                not done
                and (month.days[env.day].day_index - 1) % 30 == 0
            ):
                env.d_run = env.d_run_init
                env.d_run_nb = env.d_run_init

    return BaselineRollout(
        actions=actions,
        decision_costs=decision_costs,
        p_grid_days=[row.copy() for row in env.log_grid],
        soc_days=[row.copy() for row in env.log_soc],
        p_bess_days=[row.copy() for row in env.log_pbess],
        mode=_forecast_mode(month),
        spec=spec,
    )


_BASELINE_CACHE: dict[tuple, BaselineRollout] = {}
_BASELINE_CACHE_LOCK = threading.Lock()


class SADRBCResidualEnv(BESSEnv):
    """GrePRO learns a bounded correction around a causal SADRBC action."""

    def __init__(self, *args, residual_limit: float = DEFAULT_RESIDUAL_LIMIT,
                 forecast_spec: SADRBCForecastSpec | None = None, **kwargs):
        super().__init__(*args, extra_obs_dim=1, **kwargs)
        self.residual_limit = float(residual_limit)
        self.forecast_spec = forecast_spec or SADRBCForecastSpec()
        self._baseline: BaselineRollout | None = None
        self._baseline_decision = 0

    def reset(self, month: MonthData, soc_init: float | None = None,
              static_observation_cache: np.ndarray | None = None) -> np.ndarray:
        initial_soc = float(soc_init if soc_init is not None else self.cfg.SOC_eod)
        key = (
            id(month), round(initial_soc, 8), round(float(self.d_run_init), 8),
            self.n_steps, self.control_dt_minutes, round(self.p_ref, 8),
            round(float(self.cfg.E_cap), 8),
            round(float(self.cfg.P_rated_nominal), 8),
            self.forecast_spec.seed,
            self.forecast_spec.load_sigma, self.forecast_spec.pv_sigma,
            self.forecast_spec.rho, self.forecast_spec.replan_minutes,
            self.use_forecast,
        )
        with _BASELINE_CACHE_LOCK:
            baseline = _BASELINE_CACHE.get(key)
        if baseline is None:
            baseline = build_sadrbc_forecast_baseline(
                month,
                self.cfg,
                p_ref_kw=self.p_ref,
                control_dt_minutes=self.control_dt_minutes,
                soc_init=initial_soc,
                d_run_init_kw=self.d_run_init,
                spec=self.forecast_spec,
            )
            with _BASELINE_CACHE_LOCK:
                if len(_BASELINE_CACHE) >= 32:
                    _BASELINE_CACHE.pop(next(iter(_BASELINE_CACHE)))
                _BASELINE_CACHE[key] = baseline
        self._baseline = baseline
        self._baseline_decision = 0
        return super().reset(
            month,
            soc_init=soc_init,
            static_observation_cache=static_observation_cache,
        )

    def _baseline_action(self) -> float:
        if self._baseline is None:
            return 0.0
        return float(self._baseline.actions[self.day][self.t])

    def _fill_extra_observation(self, observation: np.ndarray, t: int) -> None:
        if self._baseline is not None:
            observation[-1] = self._baseline.actions[self.day][t]

    def step(self, residual_action: float):
        baseline_action = self._baseline_action()
        residual = float(np.clip(residual_action, -1.0, 1.0))
        final_action = float(np.clip(
            baseline_action + self.residual_limit * residual, -1.0, 1.0
        ))
        day_before = self.day
        observation, _, done, info = super().step(final_action)
        baseline_cost = self._baseline.decision_costs[self._baseline_decision]
        hybrid_cost = float(info["energy_cost"] + info["peak_pen"] + info["deg_cost"])
        reward = (baseline_cost - hybrid_cost) / 1e6
        if done or self.day != day_before:
            baseline_soc_end = float(self._baseline.soc_days[day_before][-1])
            reward -= (
                (abs(self.soc - self.cfg.SOC_eod)
                 - abs(baseline_soc_end - self.cfg.SOC_eod))
                * self.cfg.E_cap * self.cfg.price_mid / 1e6
            )
            if (
                not done
                and (self.month.days[self.day].day_index - 1) % 30 == 0
            ):
                self.d_run = self.d_run_init
                self.d_run_nb = self.d_run_init
                observation = self._obs()
        requested_kw = abs(final_action) * self.cfg.P_rated_nominal
        blocked = requested_kw > 1.0 and info["mean_abs_p_bess_kw"] < 1e-3
        info.update({
            "baseline_action": baseline_action,
            "residual_action": residual,
            "final_action": final_action,
            "blocked_action": bool(blocked),
            "baseline_cost": baseline_cost,
            "hybrid_cost": hybrid_cost,
            "residual_limit": self.residual_limit,
        })
        self._baseline_decision += 1
        return observation, reward, done, info


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
