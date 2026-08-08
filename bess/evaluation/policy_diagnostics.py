"""Tariff-aware charging diagnostics for measured policy rollouts."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from bess.core.common import is_sunday, make_bess_config, score_month, score_operating_month, tariff_vector_day
from bess.core.scenario_gen import DayData, MonthData


def _tariff_class(cfg, day, step: int) -> str:
    if step in cfg.OFF:
        return "cheap"
    if step in cfg.W1 or step in cfg.W2:
        from bess.core.common import TOU_RULES

        if not (TOU_RULES.get("sunday_no_peak") and is_sunday(day)):
            return "expensive"
    return "normal"


def _avoidable_normal_charge_kwh(
    grid_kw: np.ndarray,
    total_charge_kw: np.ndarray,
    grid_charge_kw: np.ndarray,
    soc_after: np.ndarray,
    classes: list[str],
    cfg,
    initial_running_peak_kw: float,
) -> float:
    count = len(grid_kw)
    if count == 0 or not np.any(grid_charge_kw > 1e-9):
        return 0.0
    # Variables are normal-charge removal r_t, cheap addition a_t, and the
    # cumulative counterfactual power-energy balance c_t = sum(a-r).
    bounds = []
    for index in range(count):
        upper = grid_charge_kw[index] if classes[index] == "normal" else 0.0
        bounds.append((0.0, max(0.0, float(upper))))
    for index in range(count):
        upper = cfg.P_rated_nominal - total_charge_kw[index] if classes[index] == "cheap" else 0.0
        bounds.append((0.0, max(0.0, float(upper))))

    samples_per_block = int(round(0.5 / cfg.dt))
    full_blocks = count // samples_per_block
    soc_coefficient = cfg.eta_ch * cfg.dt / cfg.E_cap
    for index in range(count):
        lower = min(0.0, float((cfg.SOC_min - soc_after[index]) / soc_coefficient))
        bounds.append((0.0, 0.0) if index == count - 1 else (lower, 0.0))

    # O(N) recurrence constraints replace O(N^2) copied cumulative rows.
    equality = lil_matrix((count, count * 3), dtype=np.float64)
    for index in range(count):
        equality[index, index] = 1.0
        equality[index, count + index] = -1.0
        equality[index, 2 * count + index] = 1.0
        if index:
            equality[index, 2 * count + index - 1] = -1.0

    peak_rows = lil_matrix((full_blocks, count * 3), dtype=np.float64)
    peak_limits = np.zeros(full_blocks, dtype=np.float64)
    running_peak = float(initial_running_peak_kw)
    block_row = 0
    for start in range(0, count, samples_per_block):
        end = min(count, start + samples_per_block)
        if end - start != samples_per_block:
            continue
        actual_sum = float(np.sum(grid_kw[start:end]))
        headroom_sum = max(0.0, running_peak * samples_per_block - actual_sum)
        peak_rows[block_row, count + start:count + end] = 1.0
        peak_limits[block_row] = headroom_sum
        block_row += 1
        running_peak = max(running_peak, actual_sum / samples_per_block)

    objective = np.concatenate((-np.ones(count), np.zeros(count * 2)))
    result = linprog(
        objective,
        A_ub=peak_rows.tocsr(),
        b_ub=peak_limits,
        A_eq=equality.tocsr(),
        b_eq=np.zeros(count),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return 0.0
    return max(0.0, float(np.sum(result.x[:count]) * cfg.dt))


def monthly_policy_diagnostics(
    month,
    rollout: dict,
    cfg,
    *,
    oracle_days: list[dict] | None = None,
    initial_running_peak_kw: float = 0.0,
) -> list[dict]:
    """Return one reproducible diagnostic record per calendar month."""
    oracle_by_index = {int(day["day_index"]): day for day in (oracle_days or [])}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, day in enumerate(month.days):
        grouped[str(day.date_iso)[:7]].append(index)

    output = []
    for month_key, indexes in sorted(grouped.items()):
        source_days = [month.days[index] for index in indexes]
        grids = [np.asarray(rollout["p_grid_days"][index], dtype=np.float64) for index in indexes]
        powers = [np.asarray(rollout["p_bess_days"][index], dtype=np.float64) for index in indexes]
        socs = [np.asarray(rollout["soc_days"][index], dtype=np.float64) for index in indexes]
        classes_by_day = [
            [_tariff_class(cfg, day, step) for step in range(len(day.load))]
            for day in source_days
        ]
        grid_charge_days = []
        pv_charge_days = []
        total_charge_days = []
        for day, power in zip(source_days, powers):
            total_charge = np.maximum(-power, 0.0)
            pv_surplus = np.maximum(np.asarray(day.pv) - np.asarray(day.load), 0.0)
            pv_charge = np.minimum(total_charge, pv_surplus)
            total_charge_days.append(total_charge)
            pv_charge_days.append(pv_charge)
            grid_charge_days.append(np.maximum(total_charge - pv_charge, 0.0))

        energy_by_class = {kind: 0.0 for kind in ("cheap", "normal", "expensive")}
        cheap_hours = 0.0
        unused_peak_safe_cheap_capacity = 0.0
        running_peak = float(initial_running_peak_kw)
        block_steps = int(round(0.5 / cfg.dt))
        for grid, total_charge, grid_charge, classes in zip(
            grids, total_charge_days, grid_charge_days, classes_by_day
        ):
            for kind in energy_by_class:
                mask = np.asarray([value == kind for value in classes])
                energy_by_class[kind] += float(np.sum(grid_charge[mask]) * cfg.dt)
            cheap_hours += sum(value == "cheap" for value in classes) * cfg.dt
            for start in range(0, len(grid), block_steps):
                end = min(len(grid), start + block_steps)
                if end - start != block_steps:
                    continue
                actual_sum = float(np.sum(grid[start:end]))
                headroom_sum = max(0.0, running_peak * block_steps - actual_sum)
                cheap_residual_sum = sum(
                    max(0.0, cfg.P_rated_nominal - total_charge[step])
                    for step in range(start, end)
                    if classes[step] == "cheap"
                )
                unused_peak_safe_cheap_capacity += min(headroom_sum, cheap_residual_sum) * cfg.dt
                running_peak = max(running_peak, actual_sum / block_steps)

        flat_grid = np.concatenate(grids)
        flat_charge = np.concatenate(total_charge_days)
        flat_grid_charge = np.concatenate(grid_charge_days)
        flat_soc_after = np.concatenate([soc[1:] for soc in socs])
        flat_classes = [value for day_classes in classes_by_day for value in day_classes]
        avoidable = _avoidable_normal_charge_kwh(
            flat_grid, flat_charge, flat_grid_charge, flat_soc_after,
            flat_classes, cfg, initial_running_peak_kw,
        )
        policy_score = score_operating_month(grids, powers, cfg, days=source_days)
        no_bess_grids = [np.maximum(np.asarray(day.load) - np.asarray(day.pv), 0.0) for day in source_days]
        no_bess_score = score_month(no_bess_grids, cfg, days=source_days)
        matching_oracle = [oracle_by_index.get(int(day.day_index)) for day in source_days]
        matching_oracle = [day for day in matching_oracle if day is not None]
        oracle_peak = None
        oracle_bill = None
        if len(matching_oracle) == len(source_days):
            oracle_grids = [day["grid"] for day in matching_oracle]
            oracle_utility = score_month(oracle_grids, cfg, days=source_days)
            oracle_bill = oracle_utility["total_cost_vnd"] + sum(
                float(day.get("wear_cost_vnd", 0.0)) for day in matching_oracle
            )
            oracle_peak = oracle_utility["pmax_month_kw"]

        six_index = int(round(6.0 / cfg.dt))
        soc_at_midnight = [float(soc[0] * 100.0) for soc in socs]
        soc_at_six = [float(soc[min(six_index, len(soc) - 1)] * 100.0) for soc in socs]
        output.append({
            "month": month_key,
            "cheap_grid_charge_kwh": round(energy_by_class["cheap"], 3),
            "normal_grid_charge_kwh": round(energy_by_class["normal"], 3),
            "expensive_grid_charge_kwh": round(energy_by_class["expensive"], 3),
            "pv_charge_kwh": round(sum(float(np.sum(values) * cfg.dt) for values in pv_charge_days), 3),
            "average_cheap_grid_charge_kw": round(energy_by_class["cheap"] / max(cheap_hours, 1e-12), 3),
            "unused_peak_safe_cheap_capacity_kwh": round(unused_peak_safe_cheap_capacity, 3),
            "avoidable_normal_charge_kwh": round(avoidable, 3),
            "soc_0000_pct_mean": round(float(np.mean(soc_at_midnight)), 3),
            "soc_0600_pct_mean": round(float(np.mean(soc_at_six)), 3),
            "policy_pmax_kw": round(policy_score["pmax_month_kw"], 3),
            "no_bess_pmax_kw": round(no_bess_score["pmax_month_kw"], 3),
            "oracle_pmax_kw": None if oracle_peak is None else round(float(oracle_peak), 3),
            "policy_total_operating_vnd": round(policy_score["total_operating_cost_vnd"]),
            "oracle_total_operating_vnd": None if oracle_bill is None else round(float(oracle_bill)),
            "policy_oracle_gap_vnd": None if oracle_bill is None else round(policy_score["total_operating_cost_vnd"] - oracle_bill),
        })
    return output


def run_cheap_window_acceptance(agent, cfg, *, reference_power_kw: float) -> dict:
    """Run the fixed two-day 250 kW-headroom versus 162 kW-needed PPO lab."""
    from bess.core.bess_env import BESSEnv

    lab_cfg = make_bess_config(cfg, 1250.0, 450.0, cfg.P_target_user)
    lab_cfg.eta_ch = 0.90
    lab_cfg.SOC_min = 0.20
    lab_cfg.SOC_max = 0.90
    lab_cfg.set_dt(0.25)
    steps = 96
    use_forecast = (getattr(agent, "meta", {}) or {}).get("obs_variant") == "fc"
    days = []
    for day_index, date_iso in enumerate(("2026-01-05", "2026-01-06")):
        day = DayData(
            load=np.full(steps, 100.0),
            pv=np.zeros(steps),
            day_type="working",
            weather="synthetic",
            day_index=day_index,
            date_iso=date_iso,
        )
        if use_forecast:
            day.forecast = np.full((steps, 4), 0.1, dtype=np.float32)
        days.append(day)
    month = MonthData(days=days, source="cheap_window_acceptance")
    meta = getattr(agent, "meta", {}) or {}
    env = BESSEnv(
        lab_cfg,
        reference_power_kw=reference_power_kw,
        initial_running_peak_kw=350.0,
        discount_factor=float(meta.get("gamma", 0.995)),
        control_interval_minutes=float(meta.get("control_dt_minutes", 15.0)),
        forecast_enabled=use_forecast,
    )
    observation = env.reset(month, initial_state_of_charge=0.20)
    # Stop exactly at 06:00 on day two. Later normal/expensive behavior must
    # not contaminate this targeted 22:30 -> cheap-window acceptance case.
    while env.current_day_index == 0 or env.current_timestep_index < 24:
        before_lab_control = env.current_day_index == 0 and env.current_timestep_index < 90
        action = 0.0 if before_lab_control else agent.predict_action(observation)
        observation, _, _, _ = env.step(action)

    grid = np.concatenate((env.grid_import_history[0][90:96], env.grid_import_history[1][:24]))
    power = np.concatenate((env.battery_power_history[0][90:96], env.battery_power_history[1][:24]))
    soc_after = np.concatenate((env.state_of_charge_history[0][91:97], env.state_of_charge_history[1][1:25]))
    classes = [
        *[_tariff_class(lab_cfg, days[0], step) for step in range(90, 96)],
        *[_tariff_class(lab_cfg, days[1], step) for step in range(24)],
    ]
    total_charge = np.maximum(-power, 0.0)
    grid_charge = total_charge.copy()
    avoidable = _avoidable_normal_charge_kwh(
        grid, total_charge, grid_charge, soc_after, classes, lab_cfg, 350.0
    )
    energy_by_class = {
        kind: float(np.sum(grid_charge[np.asarray([value == kind for value in classes])]) * lab_cfg.dt)
        for kind in ("cheap", "normal", "expensive")
    }
    block_steps = int(round(0.5 / lab_cfg.dt))
    running_peak = 350.0
    unused_peak_safe = 0.0
    for start in range(0, len(grid), block_steps):
        end = start + block_steps
        block_sum = float(np.sum(grid[start:end]))
        headroom_sum = max(0.0, running_peak * block_steps - block_sum)
        cheap_residual = sum(
            max(0.0, lab_cfg.P_rated_nominal - total_charge[step])
            for step in range(start, end) if classes[step] == "cheap"
        )
        unused_peak_safe += min(headroom_sum, cheap_residual) * lab_cfg.dt
        running_peak = max(running_peak, block_sum / block_steps)
    throughput = float(np.sum(np.abs(power)) * lab_cfg.dt)
    tariff = np.concatenate((
        tariff_vector_day(lab_cfg, days[0])[90:96],
        tariff_vector_day(lab_cfg, days[1])[:24],
    ))
    operating_bill = float(np.sum(grid * tariff) * lab_cfg.dt) + throughput * lab_cfg.battery_wear_cost_vnd_per_kwh
    diagnostics = {
        "window": "2026-01-05 22:30 -> 2026-01-06 06:00",
        "cheap_grid_charge_kwh": round(energy_by_class["cheap"], 3),
        "normal_grid_charge_kwh": round(energy_by_class["normal"], 3),
        "expensive_grid_charge_kwh": round(energy_by_class["expensive"], 3),
        "pv_charge_kwh": 0.0,
        "average_cheap_grid_charge_kw": round(energy_by_class["cheap"] / 6.0, 3),
        "unused_peak_safe_cheap_capacity_kwh": round(unused_peak_safe, 3),
        "avoidable_normal_charge_kwh": round(avoidable, 3),
        "soc_2230_pct": round(float(env.state_of_charge_history[0][90] * 100.0), 3),
        "soc_0000_pct": round(float(env.state_of_charge_history[1][0] * 100.0), 3),
        "soc_0600_pct": round(float(env.state_of_charge_history[1][24] * 100.0), 3),
        "policy_pmax_kw": round(running_peak, 3),
        "no_bess_pmax_kw": 350.0,
        "policy_total_operating_vnd": round(operating_bill),
    }
    required_cheap_energy_kwh = 1250.0 * (0.90 - 0.20) / 0.90
    required_average_charge_kw = required_cheap_energy_kwh / 6.0
    threshold_kwh = 0.05 * required_cheap_energy_kwh
    return {
        "battery_kwh": 1250.0,
        "charger_kw": 450.0,
        "night_load_kw": 100.0,
        "established_peak_kw": 350.0,
        "peak_safe_charge_kw": 250.0,
        "required_cheap_energy_kwh": required_cheap_energy_kwh,
        "required_average_charge_kw": required_average_charge_kw,
        "avoidable_normal_charge_limit_kwh": threshold_kwh,
        "passed": diagnostics["avoidable_normal_charge_kwh"] <= threshold_kwh,
        "diagnostics": diagnostics,
    }
