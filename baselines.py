"""baselines.py — the benchmark set required by the evaluation protocol.

  NO-BESS      : grid = max(0, load - pv). Lower reference for savings.
  SADRBC v13   : the production rule-based controller (algorithm/sadrbc.py),
                 fed the day's actuals as its forecast (its best case).
  ORACLE LP-PF : perfect-foresight LP (benchmark_compare/lp_core.py) chained
                 day-by-day in MONTHLY billing mode — each day pays only the
                 marginal increment of the running monthly peak. This is the
                 optimistic performance estimator of the planning layer.

All three return the same structure so drl/evaluate.py can score every
method with common.score_month on the identical cost model.
"""
from __future__ import annotations

import numpy as np

from common import STEPS_PER_DAY, DT_HOURS
from scenario_gen import MonthData
from lp_core import build_eff_load, solve_day_lp


def _result(p_grid_days, soc_days, p_bess_days):
    return {"p_grid_days": p_grid_days, "soc_days": soc_days,
            "p_bess_days": p_bess_days}


# ---------------------------------------------------------------------------
def run_no_bess(month: MonthData, cfg) -> dict:
    grids, socs, pbs = [], [], []
    for day in month.days:
        g = np.maximum(0.0, day.load - day.pv)
        grids.append(g)
        socs.append(np.full(STEPS_PER_DAY + 1, cfg.SOC_eod))
        pbs.append(np.zeros(STEPS_PER_DAY))
    return _result(grids, socs, pbs)


# ---------------------------------------------------------------------------
def _cfg_to_dict(cfg) -> dict:
    return {
        "E_cap_kWh": cfg.E_cap, "P_rated_kW": cfg.P_rated_nominal,
        "eta_ch": cfg.eta_ch, "eta_dis": cfg.eta_dis,
        "soc_min": cfg.SOC_min, "soc_max": cfg.SOC_max,
        "soc_safety_buffer": cfg.SOC_safety,
        "price_peak": cfg.price_peak, "price_mid": cfg.price_mid,
        "price_off": cfg.price_off, "T_cap": cfg.T_cap,
        "FIT_PRICE": cfg.FIT_PRICE, "ENABLE_EXPORT": cfg.ENABLE_EXPORT,
        "P_target_user_kW": cfg.P_target_user,
    }


def run_sadrbc(month: MonthData, cfg) -> dict:
    from sadrbc import SADRBCRunner
    runner = SADRBCRunner(_cfg_to_dict(cfg))
    grids, socs, pbs = [], [], []
    days = month.days
    for i, day in enumerate(days):
        nxt = days[i + 1].day_type if i + 1 < len(days) else "working"
        _, soc, pb, pg, _ = runner.step_day(
            list(day.load), list(day.pv), day.day_type, day_type_next=nxt)
        grids.append(np.asarray(pg, dtype=float))
        socs.append(np.asarray(soc, dtype=float))
        pbs.append(np.asarray(pb, dtype=float))
    return _result(grids, socs, pbs)


# ---------------------------------------------------------------------------
def run_oracle(month: MonthData, cfg,
               soc_floor_frac: float = 0.0) -> dict:
    """Day-chained perfect-foresight LP under true monthly demand billing."""
    from common import TOU_RULES, is_sunday, cfg_no_peak
    grids, socs, pbs = [], [], []
    soc = cfg.SOC_eod
    peak_floor = 0.0
    cfg_sun = (cfg_no_peak(cfg)
               if TOU_RULES.get("sunday_no_peak") else cfg)
    for day in month.days:
        cfg_day = cfg_sun if (TOU_RULES.get("sunday_no_peak")
                              and is_sunday(day)) else cfg
        eff, sur = build_eff_load(list(day.load), list(day.pv))
        res = solve_day_lp(
            eff, sur, cfg_day,
            demand_rate_per_kW=cfg.T_cap,
            soc_init=soc, soc_end_min=cfg.SOC_eod,
            soc_floor_frac=soc_floor_frac,
            peak_floor=peak_floor)
        if res["status"] != "OK":                  # defensive: idle fallback
            g = np.maximum(0.0, day.load - day.pv)
            grids.append(g)
            socs.append(np.full(STEPS_PER_DAY + 1, soc))
            pbs.append(np.zeros(STEPS_PER_DAY))
            continue
        g = np.maximum(0.0, np.asarray(res["p_grid"]))
        grids.append(g)
        socs.append(np.asarray(res["soc"]))
        pbs.append(np.asarray(res["p_bess"]))
        soc = float(res["soc"][-1])
        # running monthly peak on the same 30-min rolling metric
        roll = np.convolve(g, np.ones(2) / 2, mode="valid")
        peak_floor = max(peak_floor, float(roll.max(initial=0.0)))
    return _result(grids, socs, pbs)


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
    env = BESSEnv(cfg, p_ref_kw=p_ref_kw, use_forecast=use_fc,
                  fc_seed=fc_seed,
                  d_run_init_kw=meta.get("d_run_init_kw"))
    obs = env.reset(month)
    done = False
    lat = []
    while not done:
        t0 = time.perf_counter()
        a, _, _ = agent.act(obs, deterministic=True)
        if measure_latency:
            lat.append((time.perf_counter() - t0) * 1e3)
        obs, _, done, _ = env.step(a)
    out = _result(env.log_grid, env.log_soc, env.log_pbess)
    if measure_latency:
        out["latency_ms_mean"] = float(np.mean(lat))
        out["latency_ms_max"] = float(np.max(lat))
    return out
