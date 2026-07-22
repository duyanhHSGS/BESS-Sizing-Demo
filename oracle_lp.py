from benchmark import (
    DATA_PATH,
    _day_energy_cost,
    _demand_charge,
    _group_days,
    _load_rows,
    _month_peaks,
    _month_start_day,
    _prices_for_day,
    _rounded_series,
    _to_float,
)


DEMAND_WINDOW_HOURS = 0.5
FLOAT_EPSILON = 1e-9


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

    dt = max(0.000001, _to_float(parameters.get("dt"), 0.25))
    capacity = max(0.0, _to_float(parameters.get("battery_capacity_kWh"), 0.0))
    power_limit = max(0.0, _to_float(parameters.get("battery_power_limit_kW"), 0.0))
    charge_efficiency = _clamp(_to_float(parameters.get("charge_efficiency"), 1.0), 0.001, 1.0)
    discharge_efficiency = _clamp(_to_float(parameters.get("discharge_efficiency"), 1.0), 0.001, 1.0)
    minimum_soc = _clamp(_to_float(parameters.get("minimum_soc"), 0.0), 0.0, 1.0)
    maximum_soc = _clamp(_to_float(parameters.get("maximum_soc"), 1.0), minimum_soc, 1.0)
    required_final_soc = _clamp(_to_float(parameters.get("required_final_soc"), minimum_soc), minimum_soc, maximum_soc)

    base_days = _group_days(_load_rows(DATA_PATH), dt)
    if not base_days:
        return {"available": True, "status": "No CSV rows found.", "days": [], "summary": _empty_summary()}

    _refresh_rolling_peaks(base_days, dt)

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
                required_final_soc,
            )
        )

    summary = _build_summary(base_days, month_results, parameters, dt)
    _attach_month_peaks(month_results, dt)
    return {
        "available": True,
        "status": "Oracle LP solved." if summary["solved_day_count"] else "Oracle LP could not solve any month.",
        "days": month_results,
        "summary": summary,
    }


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
    required_final_soc,
):
    load = _flatten(days, "load")
    pv = _flatten(days, "pv")
    effective_load = [max(0.0, load_kw - pv_kw) for load_kw, pv_kw in zip(load, pv)]
    solar_surplus = [max(0.0, pv_kw - load_kw) for load_kw, pv_kw in zip(load, pv)]
    prices = [price for day in days for price in _prices_for_day(day, parameters, dt)]
    steps = len(effective_load)
    if steps == 0:
        return []

    idx = _Indexes(steps)
    variable_count = idx.peak + 1
    objective = [0.0] * variable_count
    wear_cost = _to_float(parameters.get("battery_wear_cost"), 0.0)
    demand_rate = _to_float(parameters.get("billing_peak_penalty"), 0.0) if parameters.get("billing_mode") == "2tc" else 0.0

    # math.txt objective:
    #   sum_t EnergyPrice(t) * dt * (GridCharge(t) - BatteryDischarge(t))
    # + sum_t BatteryWearCost * dt * (BatteryDischarge(t) + GridCharge(t) + SolarCharge(t))
    # + DemandChargeRate * PeakGrid.
    # EffectiveLoad(t) is omitted from the LP objective because it is a constant
    # no-battery cost; adding price * dt * EffectiveLoad(t) would change the bill
    # total, but not the optimizer's chosen dispatch.
    for step, price in enumerate(prices):
        objective[idx.grid_charge(step)] = price * dt + wear_cost * dt
        objective[idx.discharge(step)] = -price * dt + wear_cost * dt
        objective[idx.solar_charge(step)] = wear_cost * dt
    objective[idx.peak] = demand_rate

    bounds = _variable_bounds(steps, power_limit, solar_surplus, minimum_soc, maximum_soc)
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
        required_final_soc,
    )
    a_ub, b_ub = _build_inequalities(
        lil_matrix,
        steps,
        variable_count,
        idx,
        power_limit,
        dt,
        required_final_soc,
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


def _variable_bounds(steps, power_limit, solar_surplus, minimum_soc, maximum_soc):
    bounds = []
    for _ in range(steps):
        bounds.append((0.0, power_limit))
    for _ in range(steps):
        bounds.append((0.0, power_limit))
    for step in range(steps):
        bounds.append((0.0, min(power_limit, solar_surplus[step])))
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
    required_final_soc,
):
    a_eq = lil_matrix((1 + steps * 2, variable_count))
    b_eq = [0.0] * (1 + steps * 2)

    row = 0
    # The formula only states BatterySOC(end) >= RequiredFinalSOC. The LP also
    # fixes the starting SOC to RequiredFinalSOC so the oracle cannot create free
    # energy by choosing an arbitrary full battery at the beginning of the month.
    a_eq[row, idx.soc(0)] = 1.0
    b_eq[row] = required_final_soc
    row += 1

    charge_soc_gain = charge_efficiency * dt / capacity
    discharge_soc_loss = dt / (discharge_efficiency * capacity)
    for step in range(steps):
        # math.txt power balance:
        #   GridImport(t) = EffectiveLoad(t) + GridCharge(t) - BatteryDischarge(t)
        # is written for linprog as:
        #   GridImport(t) - GridCharge(t) + BatteryDischarge(t) = EffectiveLoad(t).
        a_eq[row, idx.grid_import(step)] = 1.0
        a_eq[row, idx.grid_charge(step)] = -1.0
        a_eq[row, idx.discharge(step)] = 1.0
        b_eq[row] = effective_load[step]
        row += 1

        # math.txt SOC update:
        #   SOC(t+1) = SOC(t)
        #            + ChargeEfficiency * dt / BatteryCapacity * (GridCharge(t) + SolarCharge(t))
        #            - dt / (DischargeEfficiency * BatteryCapacity) * BatteryDischarge(t).
        # Rearranged for linprog equality rows:
        #   SOC(t+1) - SOC(t) - charge_gain*GridCharge(t)
        #   - charge_gain*SolarCharge(t) + discharge_loss*BatteryDischarge(t) = 0.
        a_eq[row, idx.soc(step + 1)] = 1.0
        a_eq[row, idx.soc(step)] = -1.0
        a_eq[row, idx.grid_charge(step)] = -charge_soc_gain
        a_eq[row, idx.solar_charge(step)] = -charge_soc_gain
        a_eq[row, idx.discharge(step)] = discharge_soc_loss
        row += 1

    return a_eq, b_eq


def _build_inequalities(
    lil_matrix,
    steps,
    variable_count,
    idx,
    power_limit,
    dt,
    required_final_soc,
):
    demand_windows = _demand_windows(steps, dt)
    a_ub = lil_matrix((steps + len(demand_windows) + 1, variable_count))
    b_ub = [0.0] * (steps + len(demand_windows) + 1)

    row = 0
    for step in range(steps):
        # BatteryPowerLimit is the total charge-side limit. math2.txt makes this
        # explicit as GridCharge(t) + SolarCharge(t) <= MaxChargePower.
        a_ub[row, idx.grid_charge(step)] = 1.0
        a_ub[row, idx.solar_charge(step)] = 1.0
        b_ub[row] = power_limit
        row += 1

    for window in demand_windows:
        # math.txt's 15-minute shortcut is:
        #   0.5 * (GridImport(t) + GridImport(t+1)) <= PeakGrid.
        # This code uses CUSTOM dt: _demand_window builds normalized time weights
        # over DEMAND_WINDOW_HOURS, so dt=0.25 becomes [0.5, 0.5], dt=0.5 becomes
        # [1.0], and non-divisible dt values get partial-step weights.
        for window_step, weight in window:
            a_ub[row, idx.grid_import(window_step)] = weight
        a_ub[row, idx.peak] = -1.0
        row += 1

    # math.txt final SOC floor:
    #   BatterySOC(end) >= RequiredFinalSOC
    # is written for linprog as:
    #   -BatterySOC(end) <= -RequiredFinalSOC.
    a_ub[row, idx.soc(steps)] = -1.0
    b_ub[row] = -required_final_soc
    return a_ub, b_ub


def _slice_days(days, solution, idx, parameters, dt):
    output = []
    offset = 0
    for day in days:
        count = len(day["grid"])
        span = range(offset, offset + count)
        discharge = [solution[idx.discharge(step)] for step in span]
        grid_charge = [solution[idx.grid_charge(step)] for step in span]
        solar_charge = [solution[idx.solar_charge(step)] for step in span]
        grid_import = [solution[idx.grid_import(step)] for step in span]
        soc = [solution[idx.soc(step)] for step in range(offset, offset + count + 1)]
        rolling_grid = _rolling_30_minute_average(grid_import, dt)
        before_cost = _day_energy_cost(day, parameters, dt)
        after_cost = sum(power * price * dt for power, price in zip(grid_import, _prices_for_day(day, parameters, dt)))
        wear_cost = _to_float(parameters.get("battery_wear_cost"), 0.0) * dt * sum(
            d + gc + sc for d, gc, sc in zip(discharge, grid_charge, solar_charge)
        )

        output.append(
            {
                "day_index": day["day_index"],
                "solved": True,
                "status": "optimal",
                "grid": _rounded_series(grid_import),
                "rolling_grid": _rounded_series(rolling_grid),
                "discharge": _rounded_series(discharge),
                "grid_charge": _rounded_series(grid_charge),
                "solar_charge": _rounded_series(solar_charge),
                "soc": [round(value * 100, 1) for value in soc[:-1]],
                "final_soc": round(soc[-1] * 100, 1),
                "grid_kWh": round(sum(grid_import) * dt, 2),
                "charged_kWh": round((sum(grid_charge) + sum(solar_charge)) * dt, 2),
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
    before_peak = max((day["peak_grid_kW"] for day in base_days), default=0.0)
    after_peak = max((day.get("peak_grid_kW", 0.0) for day in solved_days), default=0.0)
    before_demand = _demand_charge(parameters, before_peak)
    after_demand = _demand_charge(parameters, after_peak)
    oracle_saving = (before_energy + before_demand) - (after_energy + after_demand + wear_cost)
    seer_factor = _clamp(_to_float(parameters.get("billing_real_saving_factor"), 1.0), 0.0, 1.0)

    return {
        "solved_day_count": len(solved_days),
        "total_grid_kWh": round(sum(day.get("grid_kWh", 0.0) for day in solved_days), 2),
        "total_discharged_kWh": round(sum(day.get("discharged_kWh", 0.0) for day in solved_days), 2),
        "peak_grid_kW": round(after_peak, 2),
        "peak_reduction_kW": round(max(0.0, before_peak - after_peak), 2),
        "energy_cost_vnd": round(after_energy),
        "demand_charge_vnd": round(after_demand),
        "wear_cost_vnd": round(wear_cost),
        "total_bill_vnd": round(after_energy + after_demand + wear_cost),
        "oracle_saving_vnd": round(oracle_saving),
        "seer_saving_vnd": round(max(0.0, oracle_saving) * seer_factor),
        "seer_factor": seer_factor,
        "sizing_economics": _build_sizing_economics(
            parameters,
            len(base_days),
            max(0.0, oracle_saving) * seer_factor,
            after_peak,
        ),
    }


def _build_sizing_economics(parameters, day_count, scenario_saving_vnd, oracle_peak_kW):
    capacity = max(0.0, _to_float(parameters.get("battery_capacity_kWh"), 0.0))
    power_limit = max(0.0, _to_float(parameters.get("battery_power_limit_kW"), 0.0))
    battery_cost = (
        capacity * _to_float(parameters.get("billing_battery_per_kWh"), 0.0)
        + power_limit * _to_float(parameters.get("billing_battery_per_kW"), 0.0)
    )
    annualization_factor = 365.0 / day_count if day_count > 0 else 0.0
    annual_saving = scenario_saving_vnd * annualization_factor
    annual_maintenance = battery_cost * max(0.0, _to_float(parameters.get("billing_yearly_maintain_percentage"), 0.0))
    annual_net_cashflow = annual_saving - annual_maintenance
    project_years = max(0, int(round(_to_float(parameters.get("billing_years"), 0.0))))
    discount_rate = max(0.0, _to_float(parameters.get("billing_discount_rate"), 0.0))
    discounted_cashflow = 0.0
    for year in range(1, project_years + 1):
        discounted_cashflow += annual_net_cashflow / ((1.0 + discount_rate) ** year)

    payback_years = None
    if annual_net_cashflow > FLOAT_EPSILON:
        payback_years = battery_cost / annual_net_cashflow

    return {
        "battery_capacity_kWh": round(capacity, 2),
        "battery_power_limit_kW": round(power_limit, 2),
        "annual_saving_vnd": round(annual_saving),
        "annual_saving_million_vnd": round(annual_saving / 1_000_000.0, 2),
        "annual_maintenance_vnd": round(annual_maintenance),
        "annual_net_cashflow_vnd": round(annual_net_cashflow),
        "npv_vnd": round(-battery_cost + discounted_cashflow),
        "npv_billion_vnd": round((-battery_cost + discounted_cashflow) / 1_000_000_000.0, 3),
        "payback_years": None if payback_years is None else round(payback_years, 2),
        "oracle_peak_kW": round(oracle_peak_kW, 2),
        "recommended_contract_max_kW": round(oracle_peak_kW * 1.05, 2),
        "pareto_status": "Yes (single case)",
    }


def _attach_month_peaks(days, dt):
    month_peaks = _month_peaks(days, dt)
    for day in days:
        day["month_peak"] = month_peaks.get(_month_start_day(day["day_index"]))


def _no_battery_result(days, parameters, dt):
    oracle_days = []
    for day in days:
        count = len(day["grid"])
        zeros = [0.0] * count
        rolling_grid = _rolling_30_minute_average(day["grid"], dt)
        oracle_days.append(
            {
                "day_index": day["day_index"],
                "solved": True,
                "status": "battery disabled",
                "grid": day["grid"],
                "rolling_grid": _rounded_series(rolling_grid),
                "discharge": zeros,
                "grid_charge": zeros,
                "solar_charge": zeros,
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
    _attach_month_peaks(oracle_days, dt)
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
        "solar_charge": [0.0] * count,
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


def _demand_windows(steps, dt):
    return [_demand_window(step, steps, dt) for step in range(steps)]


def _demand_window(start, steps, dt):
    available_hours = max(0.0, (steps - start) * dt)
    window_hours = min(DEMAND_WINDOW_HOURS, available_hours)
    if window_hours <= 0.0:
        return [(start, 1.0)]

    window = []
    remaining_hours = window_hours
    step = start
    while step < steps and remaining_hours > FLOAT_EPSILON:
        covered_hours = min(dt, remaining_hours)
        window.append((step, covered_hours / window_hours))
        remaining_hours -= covered_hours
        step += 1
    return window


def _flatten(days, key):
    values = []
    for day in days:
        values.extend(day[key])
    return values


def _rolling_30_minute_average(values, dt):
    averages = []
    for window in _demand_windows(len(values), dt):
        averages.append(sum(values[step] * weight for step, weight in window))
    return averages


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
        "seer_factor": 0,
        "sizing_economics": {
            "battery_capacity_kWh": 0,
            "battery_power_limit_kW": 0,
            "annual_saving_vnd": 0,
            "annual_saving_million_vnd": 0,
            "annual_maintenance_vnd": 0,
            "annual_net_cashflow_vnd": 0,
            "npv_vnd": 0,
            "npv_billion_vnd": 0,
            "payback_years": None,
            "oracle_peak_kW": 0,
            "recommended_contract_max_kW": 0,
            "pareto_status": "No oracle result",
        },
    }


def _clamp(value, lower, upper):
    return min(max(value, lower), upper)


class _Indexes:
    def __init__(self, steps):
        self.steps = steps
        self.discharge_start = 0
        self.grid_charge_start = steps
        self.solar_charge_start = steps * 2
        self.grid_import_start = steps * 3
        self.soc_start = steps * 4
        self.peak = self.soc_start + steps + 1

    def discharge(self, step):
        return self.discharge_start + step

    def grid_charge(self, step):
        return self.grid_charge_start + step

    def solar_charge(self, step):
        return self.solar_charge_start + step

    def grid_import(self, step):
        return self.grid_import_start + step

    def soc(self, step):
        return self.soc_start + step
