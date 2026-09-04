from functools import lru_cache
from pathlib import Path

from bess.core.settings import PPO_SOC_DEADLINE_ENABLED, PPO_SOC_DEADLINE_HOUR
from bess.evaluation.benchmark import (
    _annotate_day_billing,
    _day_energy_cost,
    _demand_charge,
    _demand_windows,
    _detect_dt_from_rows,
    _group_days,
    _load_rows,
    _month_peaks,
    _month_start_day,
    _prices_for_day,
    _rolling_30_minute_average,
    _rounded_series,
    _to_float,
    selected_data_path,
)


def build_oracle_lp(parameters):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import lil_matrix
    except ImportError:
        return {
            "available": False,
            "status": "SciPy is not installed in this Python environment.",
            "days": [],
            "summary": _empty_summary(),
        }

    csv_path = selected_data_path(parameters)
    csv_stat = csv_path.stat()
    dt, base_days = _prepared_oracle_input(
        str(csv_path.resolve()),
        csv_stat.st_size,
        csv_stat.st_mtime_ns,
    )
    capacity = max(0.0, _to_float(parameters.get("battery_capacity_kWh"), 0.0))
    power_limit = max(0.0, _to_float(parameters.get("battery_power_limit_kW"), 0.0))
    charge_efficiency = _clamp(_to_float(parameters.get("charge_efficiency"), 1.0), 0.001, 1.0)
    discharge_efficiency = _clamp(_to_float(parameters.get("discharge_efficiency"), 1.0), 0.001, 1.0)
    minimum_soc = _clamp(_to_float(parameters.get("minimum_soc"), 0.0), 0.0, 1.0)
    maximum_soc = _clamp(_to_float(parameters.get("maximum_soc"), 1.0), minimum_soc, 1.0)

    if not base_days:
        return {"available": True, "status": "No CSV rows found.", "days": [], "summary": _empty_summary()}

    if capacity <= 0.0 or power_limit <= 0.0:
        return _no_battery_result(base_days, parameters, dt)

    month_results = []
    month_starts = sorted({_month_start_day(day["day_index"]) for day in base_days})
    for month_start in month_starts:
        month_days = [day for day in base_days if _month_start_day(day["day_index"]) == month_start]
        month_results.extend(
            _solve_month(
                linprog,
                lil_matrix,
                month_days,
                parameters,
                dt,
                capacity,
                power_limit,
                charge_efficiency,
                discharge_efficiency,
                minimum_soc,
                maximum_soc,
            )
        )

    summary = _build_summary(base_days, month_results, parameters, dt)
    _attach_month_peaks(month_results, parameters, dt)
    return {
        "available": True,
        "status": "Oracle LP solved." if summary["solved_day_count"] else "Oracle LP could not solve any month.",
        "days": month_results,
        "summary": summary,
    }


@lru_cache(maxsize=8)
def _prepared_oracle_input(path_text, file_size, file_mtime_ns):
    """Parse and group an unchanged CSV once for all battery candidates."""
    del file_size, file_mtime_ns
    rows = _load_rows(Path(path_text))
    dt = _detect_dt_from_rows(rows)
    days = _group_days(rows, dt)
    _refresh_rolling_peaks(days, dt)
    return dt, days


def _solve_month(
    linprog,
    lil_matrix,
    days,
    parameters,
    dt,
    capacity,
    power_limit,
    charge_efficiency,
    discharge_efficiency,
    minimum_soc,
    maximum_soc,
):
    load = _flatten(days, "load")
    pv = _flatten(days, "pv")
    # PV only reduces factory demand. The Oracle cannot treat excess PV as a
    # separate battery-charging source.
    effective_load = [max(0.0, load_kw - pv_kw) for load_kw, pv_kw in zip(load, pv)]
    # TODO(PV-SURPLUS): model curtailment or export explicitly if the site later
    # needs behavior for PV production above factory load.
    prices = [price for day in days for price in _prices_for_day(day, parameters, dt)]
    steps = len(effective_load)
    if steps == 0:
        return []

    idx = _Indexes(steps)
    variable_count = idx.peak + 1
    objective = [0.0] * variable_count
    wear_cost = _to_float(parameters.get("battery_wear_cost"), 0.0)
    demand_rate = _to_float(parameters.get("billing_peak_penalty"), 0.0) if parameters.get("billing_mode") == "2tc" else 0.0

    for step, price in enumerate(prices):
        objective[idx.grid_charge(step)] = price * dt + wear_cost * dt
        objective[idx.discharge(step)] = -price * dt + wear_cost * dt
    objective[idx.peak] = demand_rate

    bounds = _variable_bounds(steps, power_limit, effective_load, minimum_soc, maximum_soc)
    a_eq, b_eq = _build_equalities(
        lil_matrix,
        steps,
        variable_count,
        idx,
        effective_load,
        dt,
        capacity,
        charge_efficiency,
        discharge_efficiency,
        minimum_soc,
        soc_deadline_steps=_soc_deadline_steps(days, dt) if PPO_SOC_DEADLINE_ENABLED else (),
        soc_deadline_target=maximum_soc,
    )
    a_ub, b_ub = _build_inequalities(
        lil_matrix,
        steps,
        variable_count,
        idx,
        dt,
    )

    result = linprog(
        objective,
        A_ub=a_ub.tocsr(),
        b_ub=b_ub,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return [_failed_day(day, result.message) for day in days]

    return _slice_days(days, result.x, idx, parameters, dt)


def _variable_bounds(steps, power_limit, effective_load, minimum_soc, maximum_soc):
    bounds = []
    for step in range(steps):
        bounds.append((0.0, min(power_limit, effective_load[step])))
    for _ in range(steps):
        bounds.append((0.0, power_limit))
    for _ in range(steps):
        bounds.append((0.0, None))
    for _ in range(steps + 1):
        bounds.append((minimum_soc, maximum_soc))
    bounds.append((0.0, None))
    return bounds


def _build_equalities(
    lil_matrix,
    steps,
    variable_count,
    idx,
    effective_load,
    dt,
    capacity,
    charge_efficiency,
    discharge_efficiency,
    initial_soc,
    *,
    soc_deadline_steps=(),
    soc_deadline_target=None,
):
    deadline_steps = tuple(int(step) for step in soc_deadline_steps)
    if len(set(deadline_steps)) != len(deadline_steps):
        raise ValueError("Oracle SOC deadline steps must be unique")
    if any(step <= 0 or step > steps for step in deadline_steps):
        raise ValueError("Oracle SOC deadline step must be inside the episode")
    if deadline_steps and soc_deadline_target is None:
        raise ValueError("Oracle SOC deadline target is required")
    a_eq = lil_matrix((1 + steps * 2 + len(deadline_steps), variable_count))
    b_eq = [0.0] * (1 + steps * 2 + len(deadline_steps))

    row = 0
    a_eq[row, idx.soc(0)] = 1.0
    b_eq[row] = initial_soc
    row += 1

    charge_soc_gain = charge_efficiency * dt / capacity
    discharge_soc_loss = dt / (discharge_efficiency * capacity)
    for step in range(steps):
        a_eq[row, idx.grid_import(step)] = 1.0
        a_eq[row, idx.grid_charge(step)] = -1.0
        a_eq[row, idx.discharge(step)] = 1.0
        b_eq[row] = effective_load[step]
        row += 1

        a_eq[row, idx.soc(step + 1)] = 1.0
        a_eq[row, idx.soc(step)] = -1.0
        a_eq[row, idx.grid_charge(step)] = -charge_soc_gain
        a_eq[row, idx.discharge(step)] = discharge_soc_loss
        row += 1

    for deadline_step in deadline_steps:
        a_eq[row, idx.soc(deadline_step)] = 1.0
        b_eq[row] = float(soc_deadline_target)
        row += 1

    return a_eq, b_eq


def _soc_deadline_steps(days, dt):
    """Return each day's global SOC index at the configured 06:00 deadline."""
    exact_step = float(PPO_SOC_DEADLINE_HOUR) / float(dt)
    deadline_step = round(exact_step)
    if abs(exact_step - deadline_step) > 1e-9:
        raise ValueError("SOC deadline hour must align with Oracle data resolution")
    offsets = []
    cursor = 0
    for day in days:
        day_steps = len(day["load"])
        if deadline_step <= 0 or deadline_step >= day_steps:
            raise ValueError("SOC deadline must fall inside every Oracle day")
        offsets.append(cursor + deadline_step)
        cursor += day_steps
    # TODO(IQ-67): Oracle and runtime must keep identical start-of-06:00 SOC semantics.
    return tuple(offsets)


def _build_inequalities(
    lil_matrix,
    steps,
    variable_count,
    idx,
    dt,
):
    demand_windows = _demand_windows(steps, dt)
    a_ub = lil_matrix((len(demand_windows), variable_count))
    b_ub = [0.0] * len(demand_windows)

    row = 0
    for window in demand_windows:
        for window_step, weight in window:
            a_ub[row, idx.grid_import(window_step)] = weight
        a_ub[row, idx.peak] = -1.0
        row += 1

    return a_ub, b_ub


def _slice_days(days, solution, idx, parameters, dt):
    output = []
    offset = 0
    for day in days:
        count = len(day["grid"])
        span = range(offset, offset + count)
        discharge = [solution[idx.discharge(step)] for step in span]
        grid_charge = [solution[idx.grid_charge(step)] for step in span]
        grid_import = [solution[idx.grid_import(step)] for step in span]
        soc = [solution[idx.soc(step)] for step in range(offset, offset + count + 1)]
        rolling_grid = _rolling_30_minute_average(grid_import, dt)
        before_cost = _day_energy_cost(day, parameters, dt)
        after_cost = sum(power * price * dt for power, price in zip(grid_import, _prices_for_day(day, parameters, dt)))
        wear_cost = _to_float(parameters.get("battery_wear_cost"), 0.0) * dt * sum(
            d + gc for d, gc in zip(discharge, grid_charge)
        )

        output.append(
            {
                "day_index": day["day_index"],
                "date_iso": day.get("date_iso"),
                "solved": True,
                "status": "optimal",
                "grid": [float(value) for value in grid_import],
                "rolling_grid": _rounded_series(rolling_grid),
                "discharge": _rounded_series(discharge),
                "grid_charge": _rounded_series(grid_charge),
                "soc": [round(value * 100, 1) for value in soc[:-1]],
                "final_soc": round(soc[-1] * 100, 1),
                "grid_kWh": round(sum(grid_import) * dt, 2),
                "charged_kWh": round(sum(grid_charge) * dt, 2),
                "discharged_kWh": round(sum(discharge) * dt, 2),
                "peak_grid_kW": round(max(rolling_grid, default=0.0), 2),
                "energy_cost_vnd": round(after_cost),
                "wear_cost_vnd": round(wear_cost),
                "day_saving_vnd": round(before_cost - after_cost - wear_cost),
            }
        )
        offset += count
    return output


def _build_summary(base_days, oracle_days, parameters, dt):
    solved_days = [day for day in oracle_days if day.get("solved")]
    before_energy = sum(_day_energy_cost(day, parameters, dt) for day in base_days)
    after_energy = sum(day.get("energy_cost_vnd", 0.0) for day in solved_days)
    wear_cost = sum(day.get("wear_cost_vnd", 0.0) for day in solved_days)
    before_month_peaks = _month_peaks(base_days, dt)
    after_month_peaks = _month_peaks(solved_days, dt)
    before_peak = max((peak["value_kW"] for peak in before_month_peaks.values()), default=0.0)
    after_peak = max((peak["value_kW"] for peak in after_month_peaks.values()), default=0.0)
    before_demand = sum(_demand_charge(parameters, peak["value_kW"]) for peak in before_month_peaks.values())
    after_demand = sum(_demand_charge(parameters, peak["value_kW"]) for peak in after_month_peaks.values())
    oracle_saving = (before_energy + before_demand) - (after_energy + after_demand + wear_cost)
    seer_factor = _clamp(_to_float(parameters.get("billing_real_saving_factor"), 1.0), 0.0, 1.0)
    total_bill = after_energy + after_demand + wear_cost
    month_count = len({_month_start_day(day["day_index"]) for day in base_days})
    oracle_annual_saving = _annualized_monthly_saving(base_days, solved_days, parameters, dt)
    seer_annual_saving = max(0.0, oracle_annual_saving) * seer_factor
    sizing_economics = _sizing_economics(
        parameters,
        oracle_annual_saving,
        seer_annual_saving,
        after_peak,
    )

    return {
        "solved_day_count": len(solved_days),
        "total_grid_kWh": round(sum(day.get("grid_kWh", 0.0) for day in solved_days), 2),
        "total_discharged_kWh": round(sum(day.get("discharged_kWh", 0.0) for day in solved_days), 2),
        "peak_grid_kW": round(after_peak, 2),
        "peak_reduction_kW": round(max(0.0, before_peak - after_peak), 2),
        "energy_cost_vnd": round(after_energy),
        "demand_charge_vnd": round(after_demand),
        "wear_cost_vnd": round(wear_cost),
        "total_bill_vnd": round(total_bill),
        "oracle_saving_vnd": round(oracle_saving),
        "seer_saving_vnd": round(max(0.0, oracle_saving) * seer_factor),
        "oracle_annual_saving_vnd": round(oracle_annual_saving),
        "seer_annual_saving_vnd": round(seer_annual_saving),
        "seer_factor": seer_factor,
        "month_count": month_count,
        "monthly_billing": [],
        "sizing_economics": sizing_economics,
    }


def _sizing_economics(parameters, oracle_annual_saving, seer_annual_saving, oracle_peak):
    capacity = _to_float(parameters.get("battery_capacity_kWh"), 0.0)
    power = _to_float(parameters.get("battery_power_limit_kW"), 0.0)
    battery_cost = (
        capacity * _to_float(parameters.get("billing_battery_per_kWh"), 0.0)
        + power * _to_float(parameters.get("billing_battery_per_kW"), 0.0)
    )
    yearly_maintenance = battery_cost * _to_float(parameters.get("billing_yearly_maintain_percentage"), 0.0)
    annual_net_cashflow = seer_annual_saving - yearly_maintenance
    discount_rate = _to_float(parameters.get("billing_discount_rate"), 0.0)
    project_years = max(0, round(_to_float(parameters.get("billing_years"), 0.0)))
    discounted_cashflow = 0.0
    for year in range(1, project_years + 1):
        discounted_cashflow += annual_net_cashflow / ((1 + discount_rate) ** year)
    npv = -battery_cost + discounted_cashflow
    return {
        "battery_capacity_kWh": round(capacity, 2),
        "battery_power_limit_kW": round(power, 2),
        "oracle_annual_saving_vnd": round(oracle_annual_saving),
        "annual_saving_vnd": round(seer_annual_saving),
        "annual_saving_million_vnd": seer_annual_saving / 1_000_000,
        "annual_maintenance_vnd": round(yearly_maintenance),
        "annual_net_cashflow_vnd": round(annual_net_cashflow),
        "npv_vnd": round(npv),
        "npv_billion_vnd": npv / 1_000_000_000,
        "payback_years": battery_cost / annual_net_cashflow if annual_net_cashflow > 0 else None,
        "recommended_contract_max_kW": round(oracle_peak * 1.05, 2),
        "oracle_peak_kW": round(oracle_peak, 2),
        "pareto_status": "...",
    }


def _annualized_monthly_saving(base_days, oracle_days, parameters, dt):
    oracle_by_day = {day["day_index"]: day for day in oracle_days if day.get("solved")}
    month_starts = sorted({_month_start_day(day["day_index"]) for day in base_days})
    monthly_savings = []
    for month_start in month_starts:
        month_base = [day for day in base_days if _month_start_day(day["day_index"]) == month_start]
        month_oracle = [
            oracle_by_day[day["day_index"]]
            for day in month_base
            if day["day_index"] in oracle_by_day
        ]
        if not month_base or not month_oracle:
            continue
        base_energy = sum(_day_energy_cost(day, parameters, dt) for day in month_base)
        oracle_energy = sum(day.get("energy_cost_vnd", 0.0) for day in month_oracle)
        oracle_wear = sum(day.get("wear_cost_vnd", 0.0) for day in month_oracle)
        n_days = max(1, len(month_base))
        energy_saving = (base_energy - oracle_energy - oracle_wear) * (30.0 / n_days)
        base_peak = max((day.get("peak_grid_kW", 0.0) for day in month_base), default=0.0)
        oracle_peak = max((day.get("peak_grid_kW", 0.0) for day in month_oracle), default=0.0)
        demand_saving = _demand_charge(parameters, base_peak) - _demand_charge(parameters, oracle_peak)
        monthly_savings.append(energy_saving + demand_saving)
    if not monthly_savings:
        return 0.0
    return (sum(monthly_savings) / len(monthly_savings)) * 12.0


def _attach_month_peaks(days, parameters, dt):
    month_peaks = _month_peaks(days, dt)
    for day in days:
        day["month_peak"] = month_peaks.get(_month_start_day(day["day_index"]))
    _annotate_day_billing(days, parameters, dt)


def _no_battery_result(days, parameters, dt):
    oracle_days = []
    for day in days:
        count = len(day["grid"])
        zeros = [0.0] * count
        rolling_grid = _rolling_30_minute_average(day["grid"], dt)
        oracle_days.append(
            {
                "day_index": day["day_index"],
                "date_iso": day.get("date_iso"),
                "solved": True,
                "status": "battery disabled",
                "grid": day["grid"],
                "rolling_grid": _rounded_series(rolling_grid),
                "discharge": zeros,
                "grid_charge": zeros,
                "soc": zeros,
                "final_soc": 0.0,
                "grid_kWh": day["grid_kWh"],
                "charged_kWh": 0.0,
                "discharged_kWh": 0.0,
                "peak_grid_kW": round(max(rolling_grid, default=0.0), 2),
                "energy_cost_vnd": round(_day_energy_cost(day, parameters, dt)),
                "wear_cost_vnd": 0,
                "day_saving_vnd": 0,
            }
        )
    _attach_month_peaks(oracle_days, parameters, dt)
    return {
        "available": True,
        "status": "Battery capacity or power is zero, so Oracle mirrors the benchmark.",
        "days": oracle_days,
        "summary": _build_summary(days, oracle_days, parameters, dt),
    }


def _failed_day(day, message):
    count = len(day["grid"])
    return {
        "day_index": day["day_index"],
        "solved": False,
        "status": message,
        "grid": day["grid"],
        "rolling_grid": day["rolling_grid"],
        "discharge": [0.0] * count,
        "grid_charge": [0.0] * count,
        "soc": [0.0] * count,
        "final_soc": 0.0,
        "grid_kWh": day["grid_kWh"],
        "charged_kWh": 0.0,
        "discharged_kWh": 0.0,
        "peak_grid_kW": day["peak_grid_kW"],
        "energy_cost_vnd": 0,
        "wear_cost_vnd": 0,
        "day_saving_vnd": 0,
    }


def _refresh_rolling_peaks(days, dt):
    for day in days:
        rolling_grid = _rolling_30_minute_average(day["grid"], dt)
        day["rolling_grid"] = _rounded_series(rolling_grid)
        day["peak_grid_kW"] = round(max(rolling_grid, default=0.0), 2)


def _flatten(days, key):
    values = []
    for day in days:
        values.extend(day[key])
    return values


def _empty_summary():
    return {
        "solved_day_count": 0,
        "total_grid_kWh": 0,
        "total_discharged_kWh": 0,
        "peak_grid_kW": 0,
        "peak_reduction_kW": 0,
        "energy_cost_vnd": 0,
        "demand_charge_vnd": 0,
        "wear_cost_vnd": 0,
        "total_bill_vnd": 0,
        "oracle_saving_vnd": 0,
        "seer_saving_vnd": 0,
        "oracle_annual_saving_vnd": 0,
        "seer_annual_saving_vnd": 0,
        "seer_factor": 0,
        "month_count": 0,
        "monthly_billing": [],
        "sizing_economics": {
            "battery_capacity_kWh": 0,
            "battery_power_limit_kW": 0,
            "oracle_annual_saving_vnd": 0,
            "annual_saving_vnd": 0,
            "annual_saving_million_vnd": 0,
            "annual_maintenance_vnd": 0,
            "annual_net_cashflow_vnd": 0,
            "npv_vnd": 0,
            "npv_billion_vnd": 0,
            "payback_years": None,
            "recommended_contract_max_kW": 0,
            "oracle_peak_kW": 0,
            "pareto_status": "...",
        },
    }


def _clamp(value, lower, upper):
    return min(max(value, lower), upper)


class _Indexes:
    def __init__(self, steps):
        self.steps = steps
        self.discharge_start = 0
        self.grid_charge_start = steps
        self.grid_import_start = steps * 2
        self.soc_start = steps * 3
        self.peak = self.soc_start + steps + 1

    def discharge(self, step):
        return self.discharge_start + step

    def grid_charge(self, step):
        return self.grid_charge_start + step

    def grid_import(self, step):
        return self.grid_import_start + step

    def soc(self, step):
        return self.soc_start + step
