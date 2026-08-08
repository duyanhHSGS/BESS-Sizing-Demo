"""bess.core.common.py  shared paths, config mapping, tariff helpers and the
monthly-billing scorer used by EVERY method (DRL, SADRBC, Oracle, No-BESS)
so comparisons are scored on one identical cost model.

Cost model (EVN 2-part tariff params from settings.SYSTEM_CONFIG):
  energy_cost  = sum_t tariff[t] * max(0, grid[t]) * dt        (dense)
  demand_cost  = T_cap * PMax_month
  PMax_month   = max over clock-aligned, fixed/non-overlapping 30-minute
                 meter integration intervals
  zero export  = grid[t] >= 0 for all t  (hard)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from bess.paths import PROJECT_ROOT

from bess.core.settings import DEFAULT_PARAMETERS, SYSTEM_CONFIG
from bess.core.timebase import fixed_demand_block_averages, steps_per_day_from_dt

import numpy as np

DRL_DIR = PROJECT_ROOT
ROOT = PROJECT_ROOT
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "checkpoints"

from bess.agents.sadrbc import SADRBCConfig, tariff_for_step  # noqa: E402


def ensure_inside_directory(path: Path, base_dir: Path, *, label: str = "project") -> Path:
    """Resolve a path and reject attempts to escape the requested base directory."""
    resolved = Path(path).resolve()
    base = Path(base_dir).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"path escapes {label}: {resolved}")
    return resolved


def validate_control_interval_minutes(
    native_dt_minutes: float,
    control_dt_minutes: float,
) -> float:
    """Validate one control interval against native data and billing windows."""
    native = float(native_dt_minutes)
    control = float(control_dt_minutes)
    if not np.isfinite(native) or native <= 0.0:
        raise ValueError("Dispatch data resolution must be a positive number of minutes")
    if not np.isfinite(control) or control <= 0.0:
        raise ValueError("Policy control interval must be a positive number of minutes")
    if control < native - 1e-9:
        raise ValueError(
            f"Policy control interval is {control:g} minutes, "
            f"but Dispatch data is coarser at {native:g} minutes"
        )
    ratio = control / native
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"Policy control interval of {control:g} minutes is not "
            f"an exact multiple of the {native:g}-minute Dispatch data"
        )
    if (
        abs(30.0 / control - round(30.0 / control)) > 1e-9
        or abs(1440.0 / control - round(1440.0 / control)) > 1e-9
    ):
        raise ValueError(
            f"Policy control interval of {control:g} minutes must "
            "divide both 30 minutes and 24 hours"
        )
    return control


# Financial defaults come from the same canonical UI/default parameter source.
FIN = {
    "capex_per_kwh": float(DEFAULT_PARAMETERS["billing_battery_per_kWh"]),
    "capex_per_kw": float(DEFAULT_PARAMETERS["billing_battery_per_kW"]),
    "opex_pct": float(DEFAULT_PARAMETERS["billing_yearly_maintain_percentage"]),
    "discount": float(DEFAULT_PARAMETERS["billing_discount_rate"]),
    "years": int(DEFAULT_PARAMETERS["billing_years"]),
    "realization": float(DEFAULT_PARAMETERS["billing_real_saving_factor"]),
}


def load_system_config() -> SADRBCConfig:
    """Build the fallback controller config from canonical project defaults."""
    battery = SYSTEM_CONFIG["BESS"]
    tariff = SYSTEM_CONFIG["Tariff"]
    time_config = SYSTEM_CONFIG["Time"]
    operation = SYSTEM_CONFIG["Operation"]
    safety = SYSTEM_CONFIG["Safety"]
    dt_hours = float(time_config["dt_hours"])
    cfg = {
        "E_cap_kWh": battery["E_cap_kWh"],
        "P_rated_kW": battery["P_rated_kW"],
        "eta_ch": battery["eta_ch"],
        "eta_dis": battery["eta_dis"],
        "soc_min": battery["soc_min"],
        "soc_max": battery["soc_max"],
        "soc_safety_buffer": battery["soc_safety_buffer"],
        "soc_eod": battery["soc_eod"],
        "soc_min_emergency": battery["soc_min_emergency"],
        "dt_hours": dt_hours,
        "price_peak": tariff["price_peak_VND_per_kWh"],
        "price_mid": tariff["price_mid_VND_per_kWh"],
        "price_off": tariff["price_off_VND_per_kWh"],
        "T_cap": tariff["T_cap_VND_per_kW_per_month"],
        "FIT_PRICE": tariff["FIT_PRICE_VND_per_kWh"],
        "ENABLE_EXPORT": tariff["ENABLE_EXPORT"],
        "P_target_user_kW": operation["P_target_user_kW"],
        "V_NOMINAL": safety["V_NOMINAL"],
        "V_BLACKOUT_TH": safety["V_BLACKOUT_TH"],
        "T_DERATE": list(safety["T_DERATE"]),
        "peak_windows": tariff["peak_windows"],
        "off_windows": tariff["off_windows"],
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
        "battery_wear_cost_vnd_per_kwh": base.battery_wear_cost_vnd_per_kwh,
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
        "V_NOMINAL": base.V_NOMINAL,
        "V_BLACKOUT_TH": base.V_BLACKOUT_TH,
        "T_DERATE": list(base.T_DERATE),
        "peak_windows": base.peak_windows,
        "off_windows": base.off_windows,
    })




def tariff_vector(cfg: SADRBCConfig) -> np.ndarray:
    steps_per_day = steps_per_day_from_dt(cfg.dt)
    return np.array([tariff_for_step(t, cfg) for t in range(steps_per_day)],
                    dtype=np.float64)


# Quy tc TOU theo lch (EVN: Ch nht KHNG c gi cao im  gi 
# tnh gi bnh thng). Bt/tt t cu hnh biu gi Tool C.
TOU_RULES = {"sunday_no_peak": bool(SYSTEM_CONFIG["Tariff"]["sunday_no_peak"])}


def billing_month_key(day) -> tuple[str, str | int]:
    """Return the real calendar month when dated, else a legacy 30-day bucket."""
    iso = day.get("date_iso") if isinstance(day, dict) else getattr(day, "date_iso", None)
    if iso:
        try:
            parsed = date.fromisoformat(str(iso))
            return ("calendar", f"{parsed.year:04d}-{parsed.month:02d}")
        except ValueError:
            pass
    day_index = day.get("day_index") if isinstance(day, dict) else getattr(day, "day_index", None)
    if day_index is None:
        # A scorer caller with neither calendar date nor index is still grouped
        # deterministically as one legacy month instead of inventing a date.
        return ("legacy30", 1)
    return ("legacy30", ((int(day_index) - 1) // 30) * 30 + 1)


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


def fixed_pmax_day(p_grid_day: np.ndarray, dt_hours: float) -> float:
    """Max fixed, clock-aligned 30-minute meter-interval average."""
    g = np.maximum(0.0, np.asarray(p_grid_day, dtype=np.float64))
    block_averages = fixed_demand_block_averages(g, dt_hours)
    return float(max(block_averages, default=0.0))


# Backward-compatible name for older callers; semantics are now fixed blocks.
def rolling_pmax_day(p_grid_day: np.ndarray, dt_hours: float) -> float:
    return fixed_pmax_day(p_grid_day, dt_hours)


def score_month(p_grid_days: list[np.ndarray], cfg: SADRBCConfig,
                days: list | None = None) -> dict:
    """Score utility energy plus one PMax demand charge per billing month.

    Dated inputs use their real ``date_iso`` calendar month. Undated legacy
    inputs fall back to sequential 30-day buckets by ``day_index``. If ``days``
    is omitted entirely, all provided grids are treated as one billing month.
    """
    if days is not None and len(days) != len(p_grid_days):
        raise ValueError("days and grid day counts must match for calendar billing")
    tar = tariff_vector(cfg)
    tar_sun = (tariff_vector(cfg_no_peak(cfg))
               if TOU_RULES.get("sunday_no_peak") and days else tar)
    energy = 0.0
    month_peaks: dict[tuple[str, str | int], float] = {}
    for i, g in enumerate(p_grid_days):
        g = np.maximum(0.0, np.asarray(g, dtype=np.float64))
        day = days[i] if days is not None and i < len(days) else None
        t = tar_sun if (day is not None and is_sunday(day)) else tar
        if len(t) != len(g):
            raise ValueError(
                f"Tariff length {len(t)} does not match grid length {len(g)}; "
                "set cfg.dt from the selected data resolution."
            )
        energy += float(np.sum(g * t) * cfg.dt)
        month_key = billing_month_key(day) if day is not None else ("implicit", 1)
        month_peaks[month_key] = max(
            month_peaks.get(month_key, 0.0),
            fixed_pmax_day(g, cfg.dt),
        )
    demand = sum(peak * cfg.T_cap for peak in month_peaks.values())
    pmax = max(month_peaks.values(), default=0.0)
    return {
        "energy_cost_vnd": energy,
        "demand_cost_vnd": demand,
        "total_cost_vnd": energy + demand,
        "pmax_month_kw": pmax,
        "month_count": len(month_peaks),
        "monthly_pmax_kw": {str(key[1]): peak for key, peak in month_peaks.items()},
    }


def score_operating_month(
    p_grid_days: list[np.ndarray],
    p_bess_days: list[np.ndarray],
    cfg: SADRBCConfig,
    days: list | None = None,
) -> dict:
    """Score utility cost plus the configured symmetric throughput wear cost."""
    if len(p_grid_days) != len(p_bess_days):
        raise ValueError("grid and battery-power day counts must match")
    utility = score_month(p_grid_days, cfg, days=days)
    throughput_kwh = sum(
        float(np.sum(np.abs(np.asarray(values, dtype=np.float64))) * cfg.dt)
        for values in p_bess_days
    )
    wear_cost_vnd = throughput_kwh * cfg.battery_wear_cost_vnd_per_kwh
    return {
        **utility,
        "throughput_kwh": throughput_kwh,
        "wear_cost_vnd": wear_cost_vnd,
        "total_operating_cost_vnd": utility["total_cost_vnd"] + wear_cost_vnd,
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
