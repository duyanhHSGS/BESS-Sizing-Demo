import csv
from datetime import date
from pathlib import Path

import numpy as np

from bess.core.timebase import (
    demand_window_steps,
    dispatch_month_start_day,
    fixed_demand_block_averages,
    fixed_demand_windows,
)
from bess.paths import PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA_FILENAME = "offline_data_Youngone.csv"
DATA_PATH = DATA_DIR / DEFAULT_DATA_FILENAME
DEMAND_WINDOW_HOURS = 0.5
FLOAT_EPSILON = 1e-9


def list_data_csvs():
    return sorted(path.name for path in DATA_DIR.glob("*.csv") if path.is_file())


def selected_data_filename(parameters):
    csv_files = list_data_csvs()
    requested = Path(str(parameters.get("selected_data_csv") or DEFAULT_DATA_FILENAME)).name
    if requested in csv_files:
        return requested
    if DEFAULT_DATA_FILENAME in csv_files:
        return DEFAULT_DATA_FILENAME
    return csv_files[0] if csv_files else DEFAULT_DATA_FILENAME


def selected_data_path(parameters):
    filename = selected_data_filename(parameters)
    path = (DATA_DIR / filename).resolve()
    if DATA_DIR.resolve() not in path.parents:
        raise ValueError(f"CSV path escapes data folder: {filename}")
    return path


def detect_dt_hours(path):
    rows = _load_rows(path)
    return _detect_dt_from_rows(rows)


def build_benchmark(parameters):
    csv_path = selected_data_path(parameters)
    rows = _load_rows(csv_path)
    dt = _detect_dt_from_rows(rows)
    days = _group_days(rows, dt)
    total_load_kWh = sum(day["load_kWh"] for day in days)
    total_pv_kWh = sum(day["pv_kWh"] for day in days)
    total_grid_kWh = sum(day["grid_kWh"] for day in days)
    total_surplus_kWh = sum(day["surplus_kWh"] for day in days)
    battery_cost_vnd = (
        _to_float(parameters.get("battery_capacity_kWh"), 0.0)
        * _to_float(parameters.get("billing_battery_per_kWh"), 0.0)
        + _to_float(parameters.get("battery_power_limit_kW"), 0.0)
        * _to_float(parameters.get("billing_battery_per_kW"), 0.0)
    )
    month_peaks = _month_peaks(days, dt)
    for day in days:
        day["month_peak"] = month_peaks.get(_month_start_day(day["day_index"]))
    _annotate_day_billing(days, parameters, dt)

    monthly_peak = max(
        month_peaks.values(),
        key=lambda item: item["value_kW"],
        default={"value_kW": 0.0, "day_index": None, "step": None, "time": "00:00", "month_start_day_index": None, "month_end_day_index": None},
    )
    peak_grid_kW = monthly_peak["value_kW"]
    energy_cost_vnd = sum(_day_energy_cost(day, parameters, dt) for day in days)
    demand_charge_vnd = sum(
        _demand_charge(parameters, peak["value_kW"])
        for peak in month_peaks.values()
    )
    monthly_peaks = [
        {
            **peak,
            "value_kW": round(peak["value_kW"], 2),
            "demand_charge_vnd": round(_demand_charge(parameters, peak["value_kW"])),
        }
        for _, peak in sorted(month_peaks.items())
    ]

    return {
        "dt": dt,
        "csv_filename": csv_path.name,
        "time_labels": _time_labels(_max_step_count(days), dt),
        "days": days,
        "summary": {
            "day_count": len(days),
            "month_start_day_index": monthly_peak["month_start_day_index"],
            "month_end_day_index": monthly_peak["month_end_day_index"],
            "month_count": len(month_peaks),
            "monthly_peaks": monthly_peaks,
            "total_load_kWh": round(total_load_kWh, 2),
            "total_pv_kWh": round(total_pv_kWh, 2),
            "total_grid_kWh": round(total_grid_kWh, 2),
            "total_surplus_kWh": round(total_surplus_kWh, 2),
            "battery_cost_vnd": round(battery_cost_vnd),
            "peak_grid_kW": round(peak_grid_kW, 2),
            "peak_day_index": monthly_peak["day_index"],
            "peak_step": monthly_peak["step"],
            "peak_time": monthly_peak["time"],
            "energy_cost_vnd": round(energy_cost_vnd),
            "annualized_energy_cost_vnd": round(energy_cost_vnd * 365.0 / len(days)) if days else 0,
            "demand_charge_vnd": round(demand_charge_vnd),
            "total_bill_vnd": round(energy_cost_vnd + demand_charge_vnd),
        },
    }


def _load_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            {
                "day_index": int(row["day_index"]),
                "step": int(row["step"]),
                "load_kW": float(row["P_load_kW"]),
                "pv_kW": float(row["P_pv_kW"]),
                "day_type": row["day_type"],
                "date_iso": row.get("date_iso") or None,
            }
            for row in reader
        ]


def _detect_dt_from_rows(rows):
    steps_by_day = {}
    for row in rows:
        steps_by_day.setdefault(row["day_index"], set()).add(row["step"])
    step_counts = [len(steps) for steps in steps_by_day.values() if steps]
    if not step_counts:
        raise ValueError("Cannot detect dt: CSV has no day/step rows")
    steps_per_day = max(set(step_counts), key=step_counts.count)
    if not steps_per_day:
        raise ValueError("Cannot detect dt: steps_per_day is zero")
    return round(24.0 / steps_per_day, 6)


def _group_days(rows, dt):
    grouped = {}
    for row in rows:
        grouped.setdefault(
            row["day_index"],
            {"day_index": row["day_index"], "day_type": row["day_type"], "points": []},
        )["points"].append(row)

    days = []
    for day_index in sorted(grouped):
        source_day = grouped[day_index]
        points = sorted(source_day["points"], key=lambda item: item["step"])
        load = [point["load_kW"] for point in points]
        pv = [point["pv_kW"] for point in points]
        grid = [max(0.0, point["load_kW"] - point["pv_kW"]) for point in points]
        surplus = [max(0.0, point["pv_kW"] - point["load_kW"]) for point in points]
        rolling_grid = _rolling_30_minute_average(grid, dt)

        days.append(
            {
                "day_index": day_index,
                "day_type": source_day["day_type"],
                "date_iso": points[0].get("date_iso"),
                "load": _rounded_series(load),
                "pv": _rounded_series(pv),
                "grid": _rounded_series(grid),
                "surplus": _rounded_series(surplus),
                "rolling_grid": _rounded_series(rolling_grid),
                "load_kWh": round(sum(load) * dt, 2),
                "pv_kWh": round(sum(pv) * dt, 2),
                "grid_kWh": round(sum(grid) * dt, 2),
                "surplus_kWh": round(sum(surplus) * dt, 2),
                "peak_grid_kW": round(max(rolling_grid, default=0.0), 2),
            }
        )
    return days


def _day_energy_cost(day, parameters, dt):
    costs = 0.0
    prices = _prices_for_day(day, parameters, dt)
    for grid_kW, price in zip(day["grid"], prices):
        costs += grid_kW * dt * price
    return costs


def _prices_for_day(day, parameters, dt):
    expensive = _to_float(parameters.get("billing_expensive"), 0.0)
    normal = _to_float(parameters.get("billing_normal"), 0.0)
    cheap = _to_float(parameters.get("billing_cheap"), 0.0)
    expensive_windows = _parse_windows(parameters.get("billing_windows_expensive", ""))
    cheap_windows = _parse_windows(parameters.get("billing_windows_cheap", ""))
    sunday_is_normal = bool(parameters.get("billing_sunday"))
    is_sunday = _is_sunday(day.get("date_iso"))

    prices = []
    for step in range(len(day["grid"])):
        hour = step * dt
        if _inside_windows(hour, cheap_windows):
            prices.append(cheap)
        elif _inside_windows(hour, expensive_windows) and not (sunday_is_normal and is_sunday):
            prices.append(expensive)
        else:
            prices.append(normal)
    return prices


def _demand_charge(parameters, peak_grid_kW):
    if parameters.get("billing_mode") != "2tc":
        return 0.0
    return peak_grid_kW * _to_float(parameters.get("billing_peak_penalty"), 0.0)


def _annotate_day_billing(days, parameters, dt):
    month_counts = {}
    for day in days:
        month_start = _month_start_day(day["day_index"])
        month_counts[month_start] = month_counts.get(month_start, 0) + 1

    for day in days:
        month_start = _month_start_day(day["day_index"])
        month_peak = day.get("month_peak")
        monthly_demand = _demand_charge(parameters, month_peak["value_kW"]) if month_peak else 0.0
        full_peak_on_owner = (
            monthly_demand
            if month_peak and day["day_index"] == month_peak.get("day_index")
            else 0.0
        )
        prorated_peak = monthly_demand / max(1, month_counts.get(month_start, 1))
        energy_bill = _day_energy_cost(day, parameters, dt)
        day["energy_bill_vnd"] = round(energy_bill)
        day["peak_bill_owner_vnd"] = round(full_peak_on_owner)
        day["peak_bill_prorated_vnd"] = round(prorated_peak)
        day["bill_with_owner_peak_vnd"] = round(energy_bill + full_peak_on_owner)
        day["bill_with_prorated_peak_vnd"] = round(energy_bill + prorated_peak)


def _parse_windows(raw_windows):
    windows = []
    for raw_window in str(raw_windows).split(","):
        if "-" not in raw_window:
            continue
        start_raw, end_raw = raw_window.strip().split("-", 1)
        windows.append((_time_to_hour(start_raw), _time_to_hour(end_raw)))
    return windows


def _inside_windows(hour, windows):
    for start, end in windows:
        if start <= end and start <= hour < end:
            return True
        if start > end and (hour >= start or hour < end):
            return True
    return False


def _time_to_hour(value):
    hour_raw, minute_raw = value.strip().split(":", 1)
    return int(hour_raw) + int(minute_raw) / 60


def _is_sunday(date_iso):
    if not date_iso:
        return False
    try:
        return date.fromisoformat(str(date_iso)).weekday() == 6
    except ValueError:
        return False


def _rolling_average(values, window):
    averages = []
    for index in range(len(values)):
        chunk = values[index : index + window]
        averages.append(sum(chunk) / len(chunk))
    return averages


def _rolling_30_minute_average(values, dt):
    """Legacy API name; returns fixed meter-block averages expanded per sample.

    The real two-component tariff PMax is measured over clock-aligned,
    non-overlapping 30-minute integration intervals. Repeating each completed
    block average across its native samples preserves the existing plotting/data
    shape while changing the billing semantics to the meter's fixed blocks.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return []
    block_steps = demand_window_steps(dt)
    block_averages = fixed_demand_block_averages(values, dt)
    expanded = np.repeat(np.asarray(block_averages, dtype=np.float64), block_steps)
    if len(expanded) != len(values):
        raise ValueError("Grid day must contain complete 30-minute meter intervals")
    return expanded.tolist()


def _demand_windows(steps, dt):
    """Fixed, non-overlapping 30-minute LP demand constraints."""
    return fixed_demand_windows(steps, dt)


def _demand_window(start, steps, dt):
    """Return the fixed meter block containing ``start`` (legacy helper)."""
    block_steps = demand_window_steps(dt)
    block_start = (int(start) // block_steps) * block_steps
    if block_start + block_steps > int(steps):
        return []
    weight = 1.0 / block_steps
    return [(step, weight) for step in range(block_start, block_start + block_steps)]


def _time_labels(step_count, dt):
    labels = []
    for step in range(step_count):
        total_minutes = round(step * dt * 60)
        hour = total_minutes // 60
        minute = total_minutes % 60
        labels.append(f"{hour:02d}:{minute:02d}")
    return labels


def _month_peaks(days, dt):
    if not days:
        return {}

    last_day_index = days[-1]["day_index"]
    peaks = {}
    for day in days:
        month_start = _month_start_day(day["day_index"])
        month_end = min(month_start + 29, last_day_index)
        best = peaks.setdefault(
            month_start,
            {
                "value_kW": 0.0,
                "day_index": None,
                "step": None,
                "time": "00:00",
                "month_start_day_index": month_start,
                "month_end_day_index": month_end,
            },
        )
        best["month_end_day_index"] = month_end
        for step, value in enumerate(day["rolling_grid"]):
            if value > best["value_kW"]:
                best.update(
                    {
                        "value_kW": value,
                        "day_index": day["day_index"],
                        "step": step,
                        "time": _step_to_time(step, dt),
                    }
                )
    return peaks


def _month_start_day(day_index):
    """Compatibility wrapper around the shared Dispatch Viewer 30-day bucket rule."""
    return dispatch_month_start_day(day_index)


def _step_to_time(step, dt):
    total_minutes = round(step * dt * 60)
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _max_step_count(days):
    return max((len(day["grid"]) for day in days), default=0)


def _rounded_series(values):
    return [round(value, 2) for value in values]


def _to_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
