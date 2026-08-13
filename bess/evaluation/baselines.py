"""bess.evaluation.baselines.py  the benchmark set required by the evaluation protocol.

  NO-BESS      : grid = max(0, load - pv). Lower reference for savings.
  SADRBC v13   : the production rule-based controller (algorithm/sadrbc.py),
                 fed causal real-weather forecasts when attached, otherwise
                 declared deterministic noisy forecasts. Future truth is
                 never passed to the planner.
  ORACLE       : supplied separately from the cached, month-wide Oracle LP.

Runtime baselines return the same structure so common.score_month can score
each method on the identical cost model.
"""
from __future__ import annotations

import numpy as np

from bess.core.bess_env import OBSERVATION_DIM
from bess.core.brain_runtime import (
    BrainTrajectoryRecorder,
    make_brain_env,
    native_steps_per_action,
    observation_array,
    step_brain_control,
)
from bess.core.common import validate_control_interval_minutes
from bess.core.scenario_gen import MonthData


def _result(p_grid_days, soc_days, p_bess_days):
    return {"p_grid_days": p_grid_days, "soc_days": soc_days,
            "p_bess_days": p_bess_days}


# ---------------------------------------------------------------------------
def run_no_bess(month: MonthData, cfg) -> dict:
    grids, socs, pbs = [], [], []
    for day in month.days:
        g = np.maximum(0.0, day.load - day.pv)
        n_steps = len(g)
        grids.append(g)
        socs.append(np.full(n_steps + 1, cfg.SOC_eod))
        pbs.append(np.zeros(n_steps))
    return _result(grids, socs, pbs)


# ---------------------------------------------------------------------------
def _cfg_to_dict(cfg) -> dict:
    return {
        "E_cap_kWh": cfg.E_cap, "P_rated_kW": cfg.P_rated_nominal,
        "eta_ch": cfg.eta_ch, "eta_dis": cfg.eta_dis,
        "soc_min": cfg.SOC_min, "soc_max": cfg.SOC_max,
        "soc_safety_buffer": cfg.SOC_safety,
        "soc_eod": cfg.SOC_eod,
        "soc_min_emergency": cfg.SOC_min_emergency,
        "dt_hours": cfg.dt,
        "price_peak": cfg.price_peak, "price_mid": cfg.price_mid,
        "price_off": cfg.price_off, "T_cap": cfg.T_cap,
        "FIT_PRICE": cfg.FIT_PRICE, "ENABLE_EXPORT": cfg.ENABLE_EXPORT,
        "P_target_user_kW": cfg.P_target_user,
        "W1": list(cfg.W1), "W2": list(cfg.W2),
        "INTER": list(cfg.INTER), "OFF": list(cfg.OFF),
        "W1_START": cfg.W1_START, "W2_START": cfg.W2_START,
        "OFF_PEAK_END_STEP": cfg.OFF_PEAK_END_STEP,
    }


def run_sadrbc(month: MonthData, cfg, *, forecast_spec=None,
               p_ref_kw: float | None = None) -> dict:
    from bess.forecasting.sadrbc_forecast import build_sadrbc_forecast_baseline

    peak = max((float(np.max(day.load)) for day in month.days), default=500.0)
    p_ref = p_ref_kw or max(500.0, np.ceil(peak / 500.0) * 500.0)
    rollout = build_sadrbc_forecast_baseline(
        month, cfg, p_ref_kw=p_ref, spec=forecast_spec
    )
    return _result(rollout.p_grid_days, rollout.soc_days, rollout.p_bess_days)


# ---------------------------------------------------------------------------
def validate_dispatch_sampling(meta: dict, native_dt_minutes: float) -> float:
    """Return a compatible policy control interval for native dispatch data."""
    control_dt_minutes = float(meta.get("control_dt_minutes", native_dt_minutes))
    return validate_control_interval_minutes(native_dt_minutes, control_dt_minutes)


# ---------------------------------------------------------------------------
def run_drl_policy(month: MonthData, cfg, agent, p_ref_kw: float = 500.0,
                   measure_latency: bool = False,
                   deterministic: bool = True) -> dict:
    """Roll a trained policy through its native environment contract.

    PPO2 keeps its independent senior-reference environment. Every other policy
    now consumes the canonical seven-eye ``BrainEnv`` directly. Legacy 13/17-eye
    checkpoints are intentionally rejected instead of being silently adapted.
    """
    import time

    meta = getattr(agent, "meta", {}) or {}
    native_dt_minutes = cfg.dt * 60.0
    control_dt_minutes = validate_dispatch_sampling(meta, native_dt_minutes)

    if meta.get("reference_env") == "ppo2_senior_15m_v1":
        from bess.core.ppo2_env import PPO2Env

        env = PPO2Env(
            cfg,
            p_ref_kw=p_ref_kw,
            degradation_cost_per_kwh_discharged=float(
                meta.get("degradation_cost_per_kwh_discharged", 0.0)
            ),
        )
        obs = env.reset(month)
        done = False
        lat = []
        decisions = 0
        while not done:
            t0 = time.perf_counter()
            raw_action = agent.act(obs, deterministic=deterministic)
            action = raw_action[0] if isinstance(raw_action, tuple) else raw_action
            if measure_latency:
                lat.append((time.perf_counter() - t0) * 1e3)
            obs, _, done, _ = env.step(action)
            decisions += 1
        out = _result(env.log_grid, env.log_soc, env.log_pbess)
        out["blocked_action_count"] = 0
        out["decision_count"] = decisions
        out["blocked_action_pct"] = 0.0
        if measure_latency:
            out["latency_ms_mean"] = float(np.mean(lat))
            out["latency_ms_max"] = float(np.max(lat))
        return out

    checkpoint_obs_dim = meta.get("obs_dim")
    if checkpoint_obs_dim is not None and int(checkpoint_obs_dim) != OBSERVATION_DIM:
        raise ValueError(
            f"checkpoint uses legacy observation dimension {checkpoint_obs_dim}; "
            f"current BrainEnv requires {OBSERVATION_DIM}. Retrain this policy."
        )

    wear_cost = float(meta.get("battery_wear_cost", 0.0))
    env = make_brain_env(
        month,
        cfg,
        power_scale_kw=p_ref_kw,
        battery_wear_vnd_per_kwh=wear_cost,
    )
    observation = observation_array(env.reset())
    recorder = BrainTrajectoryRecorder(month, env.bess_world.state_of_charge)
    native_steps = native_steps_per_action(cfg.dt, control_dt_minutes)

    residual_baseline_actions: list[float] | None = None
    residual_limit = float(meta.get("residual_limit", 0.20))
    if meta.get("controller") == "sadrbc_residual":
        from bess.forecasting.sadrbc_forecast import (
            SADRBCForecastSpec,
            build_sadrbc_forecast_baseline,
        )

        forecast = meta.get("sadrbc_forecast", {}) or {}
        spec = SADRBCForecastSpec(
            seed=int(forecast.get("seed", 13_0013)),
            load_sigma=float(forecast.get("load_sigma", 0.05)),
            pv_sigma=float(forecast.get("pv_sigma", 0.15)),
            rho=float(forecast.get("rho", 0.90)),
            replan_minutes=int(forecast.get("replan_minutes", 60)),
        )
        baseline = build_sadrbc_forecast_baseline(
            month,
            cfg,
            p_ref_kw=p_ref_kw,
            control_dt_minutes=control_dt_minutes,
            soc_init=float(cfg.SOC_eod),
            spec=spec,
        )
        residual_baseline_actions = [
            float(action)
            for day_actions in baseline.actions
            for action in np.asarray(day_actions, dtype=np.float64)[::native_steps]
        ]

    lat: list[float] = []
    blocked_actions = 0
    decisions = 0
    done = False
    while not done:
        t0 = time.perf_counter()
        if deterministic and hasattr(agent, "predict_action"):
            policy_action = float(agent.predict_action(observation))
        else:
            raw_action = agent.act(observation, deterministic=deterministic)
            policy_action = float(raw_action[0] if isinstance(raw_action, tuple) else raw_action)

        if residual_baseline_actions is not None:
            if decisions >= len(residual_baseline_actions):
                raise RuntimeError("SADRBC residual baseline ended before BrainEnv")
            action = float(np.clip(
                residual_baseline_actions[decisions]
                + residual_limit * np.clip(policy_action, -1.0, 1.0),
                -1.0,
                1.0,
            ))
        else:
            action = float(np.clip(policy_action, -1.0, 1.0))

        if measure_latency:
            lat.append((time.perf_counter() - t0) * 1e3)

        transition = step_brain_control(
            env,
            action,
            native_steps=native_steps,
            recorder=recorder,
        )
        blocked_actions += int(transition.adjusted_action)
        decisions += 1
        done = transition.done
        if not done:
            if transition.next_observation is None:
                raise RuntimeError("BrainEnv omitted the next observation before episode end")
            observation = observation_array(transition.next_observation)

    if residual_baseline_actions is not None and decisions != len(residual_baseline_actions):
        raise RuntimeError("SADRBC residual baseline and BrainEnv decision horizons disagree")

    out = _result(
        recorder.grid_import_days,
        recorder.state_of_charge_days,
        recorder.battery_power_days,
    )
    out["blocked_action_count"] = blocked_actions
    out["decision_count"] = decisions
    out["blocked_action_pct"] = 100.0 * blocked_actions / max(1, decisions)
    if measure_latency:
        out["latency_ms_mean"] = float(np.mean(lat))
        out["latency_ms_max"] = float(np.max(lat))
    return out
