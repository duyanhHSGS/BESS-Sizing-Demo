"""common.py  shared paths, config mapping, tariff helpers and the
monthly-billing scorer used by EVERY method (DRL, SADRBC, Oracle, No-BESS)
so comparisons are scored on one identical cost model.

Cost model (EVN 2-part tariff, TT16/2014 params from settings.SYSTEM_CONFIG):
  energy_cost  = sum_t tariff[t] * max(0, grid[t]) * dt        (dense)
  demand_cost  = T_cap * PMax_month
  PMax_month   = max over days of the 30-min ROLLING-AVERAGE grid import
                 (same convention as sadrbc.compute_PMax_30min_rolling)
  zero export  = grid[t] >= 0 for all t  (hard, TT05/2025)
"""
from __future__ import annotations

from pathlib import Path

from settings import SYSTEM_CONFIG

import numpy as np

DRL_DIR = Path(__file__).resolve().parent
ROOT = DRL_DIR
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "checkpoints"

from sadrbc import SADRBCConfig, tariff_for_step  # noqa: E402

STEPS_PER_DAY = 96
DT_HOURS = 0.25
DEMAND_WINDOW_HOURS = 0.5
def steps_per_day_from_dt(dt_hours: float) -> int:
    dt = float(dt_hours)
    if dt <= 0.0:
        raise ValueError(f"dt_hours must be positive, got {dt_hours!r}")
    steps = int(round(24.0 / dt))
    if steps <= 0 or abs(steps * dt - 24.0) > 1e-6:
        raise ValueError(f"dt_hours={dt_hours!r} does not divide a 24-hour day")
    return steps


def dt_from_steps_per_day(steps_per_day: int) -> float:
    steps = int(steps_per_day)
    if steps <= 0:
        raise ValueError(f"steps_per_day must be positive, got {steps_per_day!r}")
    return 24.0 / steps


def rolling_window_steps(dt_hours: float, window_hours: float = DEMAND_WINDOW_HOURS) -> int:
    return max(1, int(round(float(window_hours) / max(float(dt_hours), 1e-9))))

# ---------------------------------------------------------------------------
# 2026 Vietnam market CAPEX (same source as sizing/sizing_matrix_2026.py)
# ---------------------------------------------------------------------------
CAPEX_BESS_PER_KWH = 5_000_000.0   # VND/kWh  LFP container class
CAPEX_BESS_PER_KW = 4_000_000.0    # VND/kW   PCS / installation
OPEX_PCT_PER_YEAR = 0.02           # of CAPEX, maintenance + insurance
DISCOUNT_RATE = 0.08
PROJECT_YEARS = 10

# Tham s ti chnh GHI  C t UI (Tool C /api/config/tariff).
# realization: t l hin thc ha  tit kim THC THI (DRL/v13 qua plant
# tht) so vi c lng Oracle perfect-foresight dng trong sizing.
# o thc nghim: Crystal ~0.86, Tande ~0.54, Namduoc ~0.75  mc nh
# thn trng 0.6; nn cp nht theo benchmark tng site.
FIN = {
    "capex_per_kwh": CAPEX_BESS_PER_KWH,
    "capex_per_kw": CAPEX_BESS_PER_KW,
    "opex_pct": OPEX_PCT_PER_YEAR,
    "discount": DISCOUNT_RATE,
    "years": PROJECT_YEARS,
    "realization": 0.6,
}


def load_system_config() -> SADRBCConfig:
    """Map settings.SYSTEM_CONFIG onto a SADRBCConfig (no file I/O)."""
    b = SYSTEM_CONFIG["BESS"]
    tr = SYSTEM_CONFIG["Tariff"]
    op = SYSTEM_CONFIG["Operation"]
    cfg = {
        "E_cap_kWh": b["E_cap_kWh"],
        "P_rated_kW": b["P_rated_kW"],
        "eta_ch": b["eta_ch"],
        "eta_dis": b["eta_dis"],
        "soc_min": b["soc_min"],
        "soc_max": b["soc_max"],
        "soc_safety_buffer": b["soc_safety_buffer"],
        "soc_eod": b.get("soc_eod"),
        "soc_min_emergency": b["soc_min_emergency"],
        "price_peak": tr["price_peak_VND_per_kWh"],
        "price_mid": tr["price_mid_VND_per_kWh"],
        "price_off": tr["price_off_VND_per_kWh"],
        "T_cap": tr["T_cap_VND_per_kW_per_month"],
        "FIT_PRICE": tr["FIT_PRICE_VND_per_kWh"],
        "ENABLE_EXPORT": tr["ENABLE_EXPORT"],
        "P_target_user_kW": op["P_target_user_kW"],
    }
    return SADRBCConfig(cfg)


def make_bess_config(base: SADRBCConfig, e_cap_kwh: float,
                     p_rated_kw: float, p_target_kw: float) -> SADRBCConfig:
    """Clone `base` tariff/SOC/TOU-window settings with a different BESS
    size. Windows PHI c copy  nu khng, biu gi ty chnh (VD khung
    cao im mi 17:3022:30) s m thm quay v mc nh TT16."""
    return SADRBCConfig({
        "E_cap_kWh": e_cap_kwh,
        "P_rated_kW": p_rated_kw,
        "eta_ch": base.eta_ch,
        "eta_dis": base.eta_dis,
        "soc_min": base.SOC_min,
        "soc_max": base.SOC_max,
        "soc_safety_buffer": base.SOC_safety,
        "soc_eod": base.SOC_eod,
        "soc_min_emergency": base.SOC_min_emergency,
        "dt_hours": base.dt,
        "price_peak": base.price_peak,
        "price_mid": base.price_mid,
        "price_off": base.price_off,
        "T_cap": base.T_cap,
        "FIT_PRICE": base.FIT_PRICE,
        "ENABLE_EXPORT": base.ENABLE_EXPORT,
        "P_target_user_kW": p_target_kw,
        "W1": list(base.W1), "W2": list(base.W2),
        "INTER": list(base.INTER), "OFF": list(base.OFF),
        "W1_START": base.W1_START, "W2_START": base.W2_START,
        "OFF_PEAK_END_STEP": base.OFF_PEAK_END_STEP,
    })


def build_tariff_windows(peak_ranges: str, off_ranges: str, dt_hours: float = DT_HOURS) -> dict:
    """i chui khung gi "HH:MM-HH:MM,..." thnh step-lists 15 cho
    SADRBCConfig. Ti a 2 khung cao im (sngW1, chiu/tiW2);
    1 khung  W1 rng. VD kch bn mi:
        peak "17:30-22:30", off "00:00-06:00"."""
    steps_per_day = steps_per_day_from_dt(dt_hours)

    def to_steps(rng: str) -> list[int]:
        a, b = rng.strip().split("-")
        h1, m1 = map(int, a.split(":"))
        h2, m2 = map(int, b.split(":"))
        s1 = round((h1 + m1 / 60.0) / dt_hours)
        s2 = round((h2 + m2 / 60.0) / dt_hours)
        if s2 <= s1:
            s2 += steps_per_day
        return [s % steps_per_day for s in range(s1, s2)]

    peaks = [to_steps(r) for r in peak_ranges.split(",") if r.strip()]
    offs: list[int] = []
    for r in off_ranges.split(","):
        if r.strip():
            offs += to_steps(r)
    peaks.sort(key=lambda w: w[0])
    if len(peaks) == 1:
        w1, w2 = [], peaks[0]
    else:
        w1, w2 = peaks[0], peaks[-1]
    all_peak = set(w1) | set(w2)
    inter = [t for t in range(steps_per_day) if t not in all_peak and t not in set(offs)]
    # bc kt thc khi OFF bui sng (v13 sc m ti mc ny)
    morning = sorted(t for t in offs if t < steps_per_day // 2)
    off_end = (morning[-1] + 1) if morning else round(4.0 / dt_hours)
    return {"W1": w1, "W2": w2, "OFF": sorted(set(offs)), "INTER": inter,
            "W1_START": (w1[0] if w1 else w2[0]), "W2_START": w2[0],
            "OFF_PEAK_END_STEP": off_end}


def tariff_vector(cfg: SADRBCConfig) -> np.ndarray:
    steps_per_day = steps_per_day_from_dt(cfg.dt)
    return np.array([tariff_for_step(t, cfg) for t in range(steps_per_day)],
                    dtype=np.float64)


# Quy tc TOU theo lch (EVN: Ch nht KHNG c gi cao im  gi 
# tnh gi bnh thng). Bt/tt t cu hnh biu gi Tool C.
TOU_RULES = {"sunday_no_peak": False}


def is_sunday(day) -> bool:
    """DayData  c phi Ch nht khng (da date_iso; thiu ngy  False,
    d liu simulator khng c lch tht nn khng p quy tc)."""
    iso = getattr(day, "date_iso", None)
    if not iso:
        return False
    try:
        from datetime import date as _d
        return _d.fromisoformat(str(iso)).weekday() == 6
    except ValueError:
        return False


def cfg_no_peak(cfg: SADRBCConfig) -> SADRBCConfig:
    """Clone cfg vi khung cao im rng (gi cao im  gi bnh thng)
     dng cho ngy Ch nht khi TOU_RULES['sunday_no_peak'] bt."""
    c = make_bess_config(cfg, cfg.E_cap, cfg.P_rated_nominal,
                         cfg.P_target_user)
    old_peak = set(c.W1) | set(c.W2)
    c.W1, c.W2 = [], []
    c.INTER = sorted(set(c.INTER) | old_peak)
    return c


def tariff_vector_day(cfg: SADRBCConfig, day) -> np.ndarray:
    if TOU_RULES.get("sunday_no_peak") and is_sunday(day):
        return tariff_vector(cfg_no_peak(cfg))
    return tariff_vector(cfg)


def rolling_pmax_day(p_grid_day: np.ndarray, dt_hours: float = DT_HOURS) -> float:
    """Max 30-min rolling average of one day's grid import (kW)."""
    g = np.maximum(0.0, np.asarray(p_grid_day, dtype=np.float64))
    roll_win = rolling_window_steps(dt_hours)
    if len(g) < roll_win:
        return float(g.max(initial=0.0))
    roll = np.convolve(g, np.ones(roll_win) / roll_win, mode="valid")
    return float(roll.max(initial=0.0))


def score_month(p_grid_days: list[np.ndarray], cfg: SADRBCConfig,
                days: list | None = None) -> dict:
    """Monthly bill on the shared cost model. Input: list of daily grids.
    `days` (list DayData, cng th t vi grids): cho quy tc lch 
    Ch nht khng cao im khi TOU_RULES['sunday_no_peak'] bt."""
    tar = tariff_vector(cfg)
    tar_sun = (tariff_vector(cfg_no_peak(cfg))
               if TOU_RULES.get("sunday_no_peak") and days else tar)
    energy = 0.0
    pmax = 0.0
    for i, g in enumerate(p_grid_days):
        g = np.maximum(0.0, np.asarray(g, dtype=np.float64))
        t = tar_sun if (days is not None and i < len(days)
                        and is_sunday(days[i])) else tar
        if len(t) != len(g):
            raise ValueError(
                f"Tariff length {len(t)} does not match grid length {len(g)}; "
                "set cfg.dt from the selected data resolution."
            )
        energy += float(np.sum(g * t) * cfg.dt)
        pmax = max(pmax, rolling_pmax_day(g, cfg.dt))
    demand = pmax * cfg.T_cap
    return {
        "energy_cost_vnd": energy,
        "demand_cost_vnd": demand,
        "total_cost_vnd": energy + demand,
        "pmax_month_kw": pmax,
    }


def check_hard_constraints(p_grid_days, soc_days, cfg: SADRBCConfig,
                           tol: float = 1e-6) -> dict:
    """Count violations of the two hard constraints (must both be zero)."""
    export_viol = sum(int(np.any(np.asarray(g) < -tol)) for g in p_grid_days)
    soc_viol = 0
    for s in soc_days:
        s = np.asarray(s)
        # soc_min_emergency is the absolute floor allowed during blackout;
        # normal operation must stay within [SOC_min, SOC_max].
        if np.any(s < cfg.SOC_min - 1e-4) or np.any(s > cfg.SOC_max + 1e-4):
            soc_viol += 1
    return {"zero_export_violation_days": export_viol,
            "soc_violation_days": soc_viol}
