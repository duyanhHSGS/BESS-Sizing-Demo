from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from bess.core.common import TOU_RULES, build_tariff_windows, load_system_config, make_bess_config
from bess.core.scenario_gen import DayData, MonthData


def load_training_days(path: str | Path, *, weather: str = "csv") -> list[DayData]:
    """Load the canonical training CSV into ordered DayData objects."""
    csv_path = Path(path)
    by_day: dict[str, dict] = {}
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            day = by_day.setdefault(
                row["date_iso"],
                {
                    "load": [],
                    "pv": [],
                    "day_type": row["day_type"],
                    "day_index": int(row["day_index"]),
                },
            )
            step = int(row["step"])
            while len(day["load"]) <= step:
                day["load"].append(0.0)
                day["pv"].append(0.0)
            day["load"][step] = float(row["P_load_kW"])
            day["pv"][step] = float(row["P_pv_kW"])

    days: list[DayData] = []
    for date_iso in sorted(by_day):
        day = by_day[date_iso]
        days.append(
            DayData(
                load=np.asarray(day["load"], dtype=np.float64),
                pv=np.asarray(day["pv"], dtype=np.float64),
                day_type=day["day_type"],
                weather=weather,
                day_index=day["day_index"],
                date_iso=date_iso,
            )
        )
    return days


def month_blocks(days: list[DayData], *, minimum_days: int = 15) -> list[MonthData]:
    """Group chronological training days into usable calendar-month blocks."""
    blocks: dict[str, MonthData] = {}
    for day in days:
        month_key = str(day.date_iso)[:7]
        blocks.setdefault(
            month_key,
            MonthData(source=f"csv:{month_key}"),
        ).days.append(day)
    months = [
        month
        for _, month in sorted(blocks.items())
        if len(month.days) >= minimum_days
    ]
    if months or not days:
        return months
    return [MonthData(days=days, source="csv:train_short")]


def augment_month(
    month: MonthData,
    rng: np.random.Generator,
    sigma_load: float = 0.04,
    sigma_pv: float = 0.08,
    rho: float = 0.9,
) -> MonthData:
    """Apply deterministic-seeded AR(1) multiplicative noise to load and PV."""
    out = MonthData(source=month.source + ":aug")
    white_scale = np.sqrt(1.0 - rho**2)

    def ar1_error(n_steps: int, sigma: float) -> np.ndarray:
        error = np.zeros(n_steps)
        innovations = (
            sigma
            * white_scale
            * rng.standard_normal(max(0, n_steps - 1))
        )
        for step in range(1, n_steps):
            error[step] = rho * error[step - 1] + innovations[step - 1]
        return error

    for day in month.days:
        n_steps = len(day.load)
        load = np.maximum(
            0.0,
            day.load * (1.0 + ar1_error(n_steps, sigma_load)),
        )
        pv = np.maximum(
            0.0,
            day.pv * (1.0 + ar1_error(n_steps, sigma_pv)),
        )
        out.days.append(
            DayData(
                load=load,
                pv=pv,
                day_type=day.day_type,
                weather=day.weather,
                day_index=day.day_index,
                date_iso=day.date_iso,
            )
        )
    return out


def build_training_bess_config(
    e_cap_kwh: float,
    p_rated_kw: float,
    dt_hours: float,
    training_config_path: str | Path,
    *,
    default_billing: str = "2tc",
):
    """Build one canonical BESS/tariff config for every training algorithm."""
    base = load_system_config()
    cfg = make_bess_config(base, e_cap_kwh, p_rated_kw, base.P_target_user)
    cfg.dt = float(dt_hours)

    tariff = json.loads(Path(training_config_path).read_text(encoding="utf-8"))
    cfg.price_peak = float(tariff.get("price_peak", cfg.price_peak))
    cfg.price_mid = float(tariff.get("price_mid", cfg.price_mid))
    cfg.price_off = float(tariff.get("price_off", cfg.price_off))
    cfg.T_cap = float(tariff.get("t_cap", cfg.T_cap))
    cfg.eta_ch = float(tariff.get("charge_efficiency", cfg.eta_ch))
    cfg.eta_dis = float(tariff.get("discharge_efficiency", cfg.eta_dis))
    cfg.SOC_min = float(tariff.get("minimum_soc", cfg.SOC_min))
    cfg.SOC_max = float(tariff.get("maximum_soc", cfg.SOC_max))
    cfg.SOC_eod = float(tariff.get("required_final_soc", cfg.SOC_eod))

    for key, value in build_tariff_windows(
        tariff.get("peak_windows", ""),
        tariff.get("off_windows", ""),
        cfg.dt,
    ).items():
        setattr(cfg, key, value)

    billing = str(tariff.get("billing_mode", default_billing))
    TOU_RULES["sunday_no_peak"] = bool(tariff.get("sunday_no_peak", False))
    if billing == "tou":
        cfg.T_cap = 0.0
    return cfg, billing
