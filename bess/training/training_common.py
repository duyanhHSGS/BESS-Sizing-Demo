from __future__ import annotations

import calendar
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

from bess.agents.sadrbc import SADRBCConfig
from bess.core.common import TOU_RULES, load_system_config, score_month
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


def heldout_calendar_split(
    days: list[DayData],
    validation_days_requested: int,
    test_days_requested: int,
) -> tuple[list[DayData], list[DayData], list[DayData], dict]:
    """Prefer whole observed calendar-month buckets for validation and test.

    Demand charges are monthly, so a 30-row-count split can accidentally create
    a one-day "month" at either edge. Dated telemetry may miss a few source days;
    near-full buckets that span the calendar-month edges are still kept intact.
    Truly partial edge months are ignored. Undated or very sparse datasets fall
    back to the legacy exact-day-count split.
    """
    requested_total = int(validation_days_requested) + int(test_days_requested)
    if validation_days_requested < 1 or test_days_requested < 1:
        raise ValueError("validation and test day requests must both be at least 1")
    if len(days) <= requested_total:
        raise ValueError(
            f"Need more than {requested_total} days for train/validation/test; found {len(days)}"
        )

    grouped: dict[str, list[tuple[date, DayData]]] = {}
    dated = True
    for day in days:
        try:
            parsed = date.fromisoformat(str(day.date_iso))
        except (TypeError, ValueError):
            dated = False
            break
        grouped.setdefault(f"{parsed.year:04d}-{parsed.month:02d}", []).append((parsed, day))

    calendar_months: list[tuple[str, list[DayData], int]] = []
    if dated:
        for key in sorted(grouped):
            entries = sorted(grouped[key], key=lambda item: item[0])
            first_date = entries[0][0]
            expected_days = calendar.monthrange(first_date.year, first_date.month)[1]
            actual_dates = {item[0] for item in entries}
            observed_days = len(actual_dates)
            # Real telemetry can miss a few dates. A held-out month is still a
            # valid calendar bucket when it spans essentially the whole month
            # and contains enough observations; do not fall back to fake 30-row
            # billing just because one or two source dates are absent.
            covers_month_edges = (
                min(actual_dates).day <= 2
                and max(actual_dates).day >= expected_days - 1
            )
            has_substantial_coverage = observed_days >= max(
                15, int(np.ceil(0.75 * expected_days))
            )
            if covers_month_edges and has_substantial_coverage:
                calendar_months.append(
                    (key, [item[1] for item in entries], expected_days)
                )

    def take_from_end(months, minimum_nominal_days):
        chosen = []
        nominal_count = 0
        observed_count = 0
        for key, month_days, nominal_days in reversed(months):
            chosen.append((key, month_days, nominal_days))
            nominal_count += nominal_days
            observed_count += len(month_days)
            if nominal_count >= minimum_nominal_days:
                break
        return list(reversed(chosen)), observed_count, nominal_count

    if len(calendar_months) >= 2:
        test_months, test_count, test_nominal = take_from_end(
            calendar_months, test_days_requested
        )
        remaining = calendar_months[: len(calendar_months) - len(test_months)]
        val_months, val_count, val_nominal = take_from_end(
            remaining, validation_days_requested
        )
        if (
            val_months
            and test_months
            and val_nominal >= validation_days_requested
            and test_nominal >= test_days_requested
        ):
            val_days = [day for _, block, _ in val_months for day in block]
            test_days = [day for _, block, _ in test_months for day in block]
            validation_start = date.fromisoformat(val_days[0].date_iso)
            train_days = [
                day for day in days
                if date.fromisoformat(str(day.date_iso)) < validation_start
            ]
            if train_days:
                used_indexes = {id(day) for day in (*train_days, *val_days, *test_days)}
                ignored_days = [day for day in days if id(day) not in used_indexes]
                return train_days, val_days, test_days, {
                    "mode": "calendar_month_buckets",
                    "validation_months": [key for key, _, _ in val_months],
                    "test_months": [key for key, _, _ in test_months],
                    "validation_days": val_count,
                    "test_days": test_count,
                    "validation_nominal_calendar_days": val_nominal,
                    "test_nominal_calendar_days": test_nominal,
                    "ignored_edge_days": len(ignored_days),
                }

    split_days = requested_total
    train_days = days[:-split_days]
    val_days = days[-split_days:-test_days_requested]
    test_days = days[-test_days_requested:]
    return train_days, val_days, test_days, {
        "mode": "legacy_day_count",
        "validation_days": len(val_days),
        "test_days": len(test_days),
        "ignored_edge_days": 0,
    }


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
    tariff = json.loads(Path(training_config_path).read_text(encoding="utf-8"))
    if "battery_wear_cost" not in tariff:
        raise ValueError("training config requires battery_wear_cost")
    battery_wear_cost = float(tariff["battery_wear_cost"])
    if not np.isfinite(battery_wear_cost) or battery_wear_cost < 0.0:
        raise ValueError("battery_wear_cost must be finite and >= 0")
    cfg = SADRBCConfig({
        "E_cap_kWh": e_cap_kwh,
        "P_rated_kW": p_rated_kw,
        "battery_wear_cost_vnd_per_kwh": battery_wear_cost,
        "eta_ch": float(tariff.get("charge_efficiency", base.eta_ch)),
        "eta_dis": float(tariff.get("discharge_efficiency", base.eta_dis)),
        "soc_min": float(tariff.get("minimum_soc", base.SOC_min)),
        "soc_max": float(tariff.get("maximum_soc", base.SOC_max)),
        "soc_safety_buffer": base.SOC_safety,
        "soc_eod": float(tariff.get("required_final_soc", base.SOC_eod)),
        "soc_min_emergency": base.SOC_min_emergency,
        "dt_hours": float(dt_hours),
        "price_peak": float(tariff.get("price_peak", base.price_peak)),
        "price_mid": float(tariff.get("price_mid", base.price_mid)),
        "price_off": float(tariff.get("price_off", base.price_off)),
        "T_cap": float(tariff.get("t_cap", base.T_cap)),
        "FIT_PRICE": base.FIT_PRICE,
        "ENABLE_EXPORT": base.ENABLE_EXPORT,
        "P_target_user_kW": base.P_target_user,
        "V_NOMINAL": base.V_NOMINAL,
        "V_BLACKOUT_TH": base.V_BLACKOUT_TH,
        "T_DERATE": list(base.T_DERATE),
        "peak_windows": str(tariff.get("peak_windows", base.peak_windows)),
        "off_windows": str(tariff.get("off_windows", base.off_windows)),
    })

    billing = str(tariff.get("billing_mode", default_billing))
    TOU_RULES["sunday_no_peak"] = bool(tariff.get("sunday_no_peak", False))
    if billing == "tou":
        cfg.T_cap = 0.0
    return cfg, billing


def score_cached_oracle(
    cache_path: str | Path,
    day_indexes: list[int],
    cfg,
    source_days: list[DayData],
) -> dict:
    """Score cached Oracle grid plus its saved UI-priced throughput wear."""
    from bess.evaluation.oracle.oracle_cache import load_cached_training_days

    oracle_days = load_cached_training_days(cache_path, day_indexes)
    utility = score_month([day["grid"] for day in oracle_days], cfg, days=source_days)
    wear_cost = sum(float(day.get("wear_cost_vnd", 0.0)) for day in oracle_days)
    return {
        **utility,
        "wear_cost_vnd": wear_cost,
        "total_operating_cost_vnd": utility["total_cost_vnd"] + wear_cost,
        "days": oracle_days,
    }
