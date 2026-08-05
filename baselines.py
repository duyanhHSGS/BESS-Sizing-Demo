"""baselines.py  the benchmark set required by the evaluation protocol.

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

from scenario_gen import MonthData
from settings import PPO_GAMMA


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
    from sadrbc_forecast import build_sadrbc_forecast_baseline

    peak = max((float(np.max(day.load)) for day in month.days), default=500.0)
    p_ref = p_ref_kw or max(500.0, np.ceil(peak / 500.0) * 500.0)
    rollout = build_sadrbc_forecast_baseline(
        month, cfg, p_ref_kw=p_ref, spec=forecast_spec
    )
    return _result(rollout.p_grid_days, rollout.soc_days, rollout.p_bess_days)


# ---------------------------------------------------------------------------
def validate_dispatch_sampling(meta: dict, native_dt_minutes: float) -> float:
    """Return a compatible policy control interval for native dispatch data."""
    control_dt_minutes = float(
        meta.get("control_dt_minutes", native_dt_minutes)
    )
    if not np.isfinite(native_dt_minutes) or native_dt_minutes <= 0.0:
        raise ValueError("Dispatch data resolution must be a positive number of minutes")
    if not np.isfinite(control_dt_minutes) or control_dt_minutes <= 0.0:
        raise ValueError("Policy control interval must be a positive number of minutes")

    ratio = control_dt_minutes / native_dt_minutes
    if control_dt_minutes < native_dt_minutes - 1e-9:
        raise ValueError(
            f"Policy control interval is {control_dt_minutes:g} minutes, "
            f"but Dispatch data is coarser at {native_dt_minutes:g} minutes"
        )
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"Policy control interval of {control_dt_minutes:g} minutes is not "
            f"an exact multiple of the {native_dt_minutes:g}-minute Dispatch data"
        )
    if (
        abs(30.0 / control_dt_minutes - round(30.0 / control_dt_minutes)) > 1e-9
        or abs(1440.0 / control_dt_minutes - round(1440.0 / control_dt_minutes)) > 1e-9
    ):
        raise ValueError(
            f"Policy control interval of {control_dt_minutes:g} minutes must "
            "divide both 30 minutes and 24 hours"
        )
    return control_dt_minutes


# ---------------------------------------------------------------------------
def run_drl_policy(month: MonthData, cfg, agent, p_ref_kw: float = 500.0,
                   measure_latency: bool = False) -> dict:
    """Deterministic rollout of a trained policy. The env variant
    (forecast-informed or not) follows the checkpoint's meta."""
    import time
    from bess_env import BESSEnv
    from sadrbc_forecast import SADRBCForecastSpec, SADRBCResidualEnv
    meta = getattr(agent, "meta", {}) or {}
    use_fc = meta.get("obs_variant") == "fc"
    native_dt_minutes = cfg.dt * 60.0
    control_dt_minutes = validate_dispatch_sampling(meta, native_dt_minutes)
    env_class = SADRBCResidualEnv if meta.get("controller") == "sadrbc_residual" else BESSEnv
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
    env = env_class(cfg, p_ref_kw=p_ref_kw, use_forecast=use_fc,
                    use_tariff_lookahead=int(meta.get("obs_schema_version", 1)) >= 2,
                    d_run_init_kw=meta.get("d_run_init_kw"),
                    gamma=float(meta.get("gamma", PPO_GAMMA)),
                    control_dt_minutes=control_dt_minutes,
                    **env_kwargs)
    obs = env.reset(month)
    done = False
    lat = []
    blocked_actions = 0
    decisions = 0
    while not done:
        t0 = time.perf_counter()
        if hasattr(agent, "predict_action"):
            a = agent.predict_action(obs)
        else:
            a, _, _ = agent.act(obs, deterministic=True)
        if measure_latency:
            lat.append((time.perf_counter() - t0) * 1e3)
        obs, _, done, info = env.step(a)
        blocked_actions += int(info.get("blocked_action", False))
        decisions += 1
    out = _result(env.log_grid, env.log_soc, env.log_pbess)
    out["blocked_action_count"] = blocked_actions
    out["decision_count"] = decisions
    out["blocked_action_pct"] = 100.0 * blocked_actions / max(1, decisions)
    if measure_latency:
        out["latency_ms_mean"] = float(np.mean(lat))
        out["latency_ms_max"] = float(np.max(lat))
    return out
