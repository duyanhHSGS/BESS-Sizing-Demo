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

from bess.core.common import validate_control_interval_minutes
from bess.core.scenario_gen import MonthData
from bess.core.settings import PPO_GAMMA


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
    """Roll a trained policy through the environment declared by its checkpoint."""
    import time
    from bess.core.bess_env import BESSEnv
    from bess.forecasting.sadrbc_forecast import SADRBCForecastSpec, SADRBCResidualEnv

    meta = getattr(agent, "meta", {}) or {}
    native_dt_minutes = cfg.dt * 60.0
    validate_dispatch_sampling(meta, native_dt_minutes)

    if meta.get("reference_env") == "ppo2_senior_15m_v1":
        from bess.core.ppo2_env import PPO2Env

        env = PPO2Env(
            cfg,
            p_ref_kw=p_ref_kw,
            degradation_cost_per_kwh_discharged=float(
                meta.get("degradation_cost_per_kwh_discharged", 0.0)
            ),
        )
    else:
        use_fc = meta.get("obs_variant") == "fc"
        control_dt_minutes = validate_dispatch_sampling(meta, native_dt_minutes)
        env_class = (
            SADRBCResidualEnv
            if meta.get("controller") == "sadrbc_residual"
            else BESSEnv
        )
        env_kwargs = {}
        if env_class is SADRBCResidualEnv:
            forecast = meta.get("sadrbc_forecast", {}) or {}
            env_kwargs = {
                "residual_limit": float(meta.get("residual_limit", 0.20)),
                "forecast_spec": SADRBCForecastSpec(
                    seed=int(forecast.get("seed", 13_0013)),
                    load_sigma=float(forecast.get("load_sigma", 0.05)),
                    pv_sigma=float(forecast.get("pv_sigma", 0.15)),
                    rho=float(forecast.get("rho", 0.90)),
                    replan_minutes=int(forecast.get("replan_minutes", 60)),
                ),
            }
        env = env_class(
            cfg,
            reference_power_kw=p_ref_kw,
            forecast_enabled=use_fc,
            initial_running_peak_kw=meta.get("d_run_init_kw"),
            discount_factor=float(meta.get("gamma", PPO_GAMMA)),
            control_interval_minutes=control_dt_minutes,
            **env_kwargs,
        )

    obs = env.reset(month)
    done = False
    lat = []
    blocked_actions = 0
    decisions = 0
    while not done:
        t0 = time.perf_counter()
        if meta.get("reference_env") == "ppo2_senior_15m_v1":
            raw_action = agent.act(obs, deterministic=deterministic)
            a = raw_action[0] if isinstance(raw_action, tuple) else raw_action
        elif deterministic and hasattr(agent, "predict_action"):
            a = agent.predict_action(obs)
        else:
            raw_action = agent.act(obs, deterministic=deterministic)
            a = raw_action[0] if isinstance(raw_action, tuple) else raw_action
        if measure_latency:
            lat.append((time.perf_counter() - t0) * 1e3)
        obs, _, done, info = env.step(a)
        blocked_actions += int(info.get("blocked_action", False))
        decisions += 1
    if meta.get("reference_env") == "ppo2_senior_15m_v1":
        out = _result(env.log_grid, env.log_soc, env.log_pbess)
    else:
        out = _result(
            env.grid_import_history,
            env.state_of_charge_history,
            env.battery_power_history,
        )
    out["blocked_action_count"] = blocked_actions
    out["decision_count"] = decisions
    out["blocked_action_pct"] = 100.0 * blocked_actions / max(1, decisions)
    if measure_latency:
        out["latency_ms_mean"] = float(np.mean(lat))
        out["latency_ms_max"] = float(np.max(lat))
    return out
