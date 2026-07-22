import csv
from pathlib import Path


DATA_PATH = Path(__file__).with_name("offline_data_Youngone.csv")


def build_benchmark(parameters):
    dt = _to_float(parameters.get("dt"), 0.25)
    rows = _load_rows(DATA_PATH)
    days = _group_days(rows, dt)
    total_load_kWh = sum(day["load_kWh"] for day in days)
    total_pv_kWh = sum(day["pv_kWh"] for day in days)
    total_grid_kWh = sum(day["grid_kWh"] for day in days)
    total_surplus_kWh = sum(day["surplus_kWh"] for day in days)
    month_peaks = _month_peaks(days, dt)
    for day in days:
        day["month_peak"] = month_peaks.get(_month_start_day(day["day_index"]))

    monthly_peak = max(
        month_peaks.values(),
        key=lambda item: item["value_kW"],
        default={"value_kW": 0.0, "day_index": None, "step": None, "time": "00:00", "month_start_day_index": None, "month_end_day_index": None},
    )
    peak_grid_kW = monthly_peak["value_kW"]
    energy_cost_vnd = sum(_day_energy_cost(day, parameters, dt) for day in days)
    demand_charge_vnd = _demand_charge(parameters, peak_grid_kW)

    return {
        "dt": dt,
        "time_labels": _time_labels(_max_step_count(days), dt),
        "days": days,
        "summary": {
            "day_count": len(days),
            "month_start_day_index": monthly_peak["month_start_day_index"],
            "month_end_day_index": monthly_peak["month_end_day_index"],
            "total_load_kWh": round(total_load_kWh, 2),
            "total_pv_kWh": round(total_pv_kWh, 2),
            "total_grid_kWh": round(total_grid_kWh, 2),
            "total_surplus_kWh": round(total_surplus_kWh, 2),
            "peak_grid_kW": round(peak_grid_kW, 2),
            "peak_day_index": monthly_peak["day_index"],
            "peak_step": monthly_peak["step"],
            "peak_time": monthly_peak["time"],
            "energy_cost_vnd": round(energy_cost_vnd),
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
            }
            for row in reader
        ]


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
        rolling_grid = _rolling_average(grid, 2)

        days.append(
            {
                "day_index": day_index,
                "day_type": source_day["day_type"],
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
    is_sunday = str(day["day_type"]).lower() == "sunday"

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


def _rolling_average(values, window):
    averages = []
    for index in range(len(values)):
        chunk = values[index : index + window]
        averages.append(sum(chunk) / len(chunk))
    return averages


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
    return ((day_index - 1) // 30) * 30 + 1


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
