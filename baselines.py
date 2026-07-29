"""baselines.py  the benchmark set required by the evaluation protocol.

  NO-BESS      : grid = max(0, load - pv). Lower reference for savings.
  SADRBC v13   : the production rule-based controller (algorithm/sadrbc.py),
                 fed the day's actuals as its forecast (its best case).
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


def run_sadrbc(month: MonthData, cfg) -> dict:
    from sadrbc import SADRBCRunner
    runner = SADRBCRunner(_cfg_to_dict(cfg))
    grids, socs, pbs = [], [], []
    days = month.days
    for i, day in enumerate(days):
        if i and (day.day_index - 1) % 30 == 0:
            runner.reset_monthly(reason="30_day_billing_boundary")
        nxt = days[i + 1].day_type if i + 1 < len(days) else "working"
        _, soc, pb, pg, _ = runner.step_day(
            list(day.load), list(day.pv), day.day_type, day_type_next=nxt)
        grids.append(np.asarray(pg, dtype=float))
        socs.append(np.asarray(soc, dtype=float))
        pbs.append(np.asarray(pb, dtype=float))
    return _result(grids, socs, pbs)


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
                   measure_latency: bool = False,
                   fc_seed: int = 12345) -> dict:
    """Deterministic rollout of a trained policy. The env variant
    (forecast-informed or not) follows the checkpoint's meta."""
    import time
    from bess_env import BESSEnv
    meta = getattr(agent, "meta", {}) or {}
    use_fc = meta.get("obs_variant") == "fc"
    native_dt_minutes = cfg.dt * 60.0
    control_dt_minutes = validate_dispatch_sampling(meta, native_dt_minutes)
    env = BESSEnv(cfg, p_ref_kw=p_ref_kw, use_forecast=use_fc,
                  fc_seed=fc_seed,
                  d_run_init_kw=meta.get("d_run_init_kw"),
                  gamma=float(meta.get("gamma", PPO_GAMMA)),
                  control_dt_minutes=control_dt_minutes)
    obs = env.reset(month)
    done = False
    lat = []
    while not done:
        t0 = time.perf_counter()
        if hasattr(agent, "predict_action"):
            a = agent.predict_action(obs)
        else:
            a, _, _ = agent.act(obs, deterministic=True)
        if measure_latency:
            lat.append((time.perf_counter() - t0) * 1e3)
        obs, _, done, _ = env.step(a)
    out = _result(env.log_grid, env.log_soc, env.log_pbess)
    if measure_latency:
        out["latency_ms_mean"] = float(np.mean(lat))
        out["latency_ms_max"] = float(np.max(lat))
    return out
