"""common.py — shared paths, config mapping, tariff helpers and the
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
ROLL_WIN = 2          # 30-min billing window = 2 x 15-min slots

# ---------------------------------------------------------------------------
# 2026 Vietnam market CAPEX (same source as sizing/sizing_matrix_2026.py)
# ---------------------------------------------------------------------------
CAPEX_BESS_PER_KWH = 5_000_000.0   # VND/kWh — LFP container class
CAPEX_BESS_PER_KW = 4_000_000.0    # VND/kW  — PCS / installation
OPEX_PCT_PER_YEAR = 0.02           # of CAPEX, maintenance + insurance
DISCOUNT_RATE = 0.08
PROJECT_YEARS = 10

# Tham số tài chính GHI ĐÈ ĐƯỢC từ UI (Tool C /api/config/tariff).
# realization: tỷ lệ hiện thực hóa — tiết kiệm THỰC THI (DRL/v13 qua plant
# thật) so với ước lượng Oracle perfect-foresight dùng trong sizing.
# Đo thực nghiệm: Crystal ~0.86, Tande ~0.54, Namduoc ~0.75 → mặc định
# thận trọng 0.6; nên cập nhật theo benchmark từng site.
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
    size. Windows PHẢI được copy — nếu không, biểu giá tùy chỉnh (VD khung
    cao điểm mới 17:30–22:30) sẽ âm thầm quay về mặc định TT16."""
    return SADRBCConfig({
        "E_cap_kWh": e_cap_kwh,
        "P_rated_kW": p_rated_kw,
        "eta_ch": base.eta_ch,
        "eta_dis": base.eta_dis,
        "soc_min": base.SOC_min,
        "soc_max": base.SOC_max,
        "soc_safety_buffer": base.SOC_safety,
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


def build_tariff_windows(peak_ranges: str, off_ranges: str) -> dict:
    """Đổi chuỗi khung giờ "HH:MM-HH:MM,..." thành step-lists 15′ cho
    SADRBCConfig. Tối đa 2 khung cao điểm (sáng→W1, chiều/tối→W2);
    1 khung → W1 rỗng. VD kịch bản mới:
        peak "17:30-22:30", off "00:00-06:00"."""
    def to_steps(rng: str) -> list[int]:
        a, b = rng.strip().split("-")
        h1, m1 = map(int, a.split(":"))
        h2, m2 = map(int, b.split(":"))
        s1 = h1 * 4 + m1 // 15
        s2 = h2 * 4 + m2 // 15
        if s2 <= s1:
            s2 += 96                       # qua nửa đêm
        return [s % 96 for s in range(s1, s2)]

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
    inter = [t for t in range(96) if t not in all_peak and t not in set(offs)]
    # bước kết thúc khối OFF buổi sáng (v13 sạc đêm tới mốc này)
    morning = sorted(t for t in offs if t < 48)
    off_end = (morning[-1] + 1) if morning else 16
    return {"W1": w1, "W2": w2, "OFF": sorted(set(offs)), "INTER": inter,
            "W1_START": (w1[0] if w1 else w2[0]), "W2_START": w2[0],
            "OFF_PEAK_END_STEP": off_end}


def tariff_vector(cfg: SADRBCConfig) -> np.ndarray:
    return np.array([tariff_for_step(t, cfg) for t in range(STEPS_PER_DAY)],
                    dtype=np.float64)


# Quy tắc TOU theo lịch (EVN: Chủ nhật KHÔNG có giờ cao điểm — giờ đó
# tính giá bình thường). Bật/tắt từ cấu hình biểu giá Tool C.
TOU_RULES = {"sunday_no_peak": False}


def is_sunday(day) -> bool:
    """DayData → có phải Chủ nhật không (dựa date_iso; thiếu ngày → False,
    dữ liệu simulator không có lịch thật nên không áp quy tắc)."""
    iso = getattr(day, "date_iso", None)
    if not iso:
        return False
    try:
        from datetime import date as _d
        return _d.fromisoformat(str(iso)).weekday() == 6
    except ValueError:
        return False


def cfg_no_peak(cfg: SADRBCConfig) -> SADRBCConfig:
    """Clone cfg với khung cao điểm rỗng (giờ cao điểm → giá bình thường)
    — dùng cho ngày Chủ nhật khi TOU_RULES['sunday_no_peak'] bật."""
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


def rolling_pmax_day(p_grid_day: np.ndarray) -> float:
    """Max 30-min rolling average of one day's grid import (kW)."""
    g = np.maximum(0.0, np.asarray(p_grid_day, dtype=np.float64))
    if len(g) < ROLL_WIN:
        return float(g.max(initial=0.0))
    roll = np.convolve(g, np.ones(ROLL_WIN) / ROLL_WIN, mode="valid")
    return float(roll.max(initial=0.0))


def score_month(p_grid_days: list[np.ndarray], cfg: SADRBCConfig,
                days: list | None = None) -> dict:
    """Monthly bill on the shared cost model. Input: list of 96-step grids.
    `days` (list DayData, cùng thứ tự với grids): cho quy tắc lịch —
    Chủ nhật không cao điểm khi TOU_RULES['sunday_no_peak'] bật."""
    tar = tariff_vector(cfg)
    tar_sun = (tariff_vector(cfg_no_peak(cfg))
               if TOU_RULES.get("sunday_no_peak") and days else tar)
    energy = 0.0
    pmax = 0.0
    for i, g in enumerate(p_grid_days):
        g = np.maximum(0.0, np.asarray(g, dtype=np.float64))
        t = tar_sun if (days is not None and i < len(days)
                        and is_sunday(days[i])) else tar
        energy += float(np.sum(g * t) * DT_HOURS)
        pmax = max(pmax, rolling_pmax_day(g))
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
