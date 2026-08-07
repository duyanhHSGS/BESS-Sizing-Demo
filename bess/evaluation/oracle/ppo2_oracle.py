"""Senior-reference month LP and scorer used only by PPO2.

This module ports the reference project's fixed-30-minute-block economics so the
PPO2 teacher (Oracle), reward, validation scorer, and test scorer all solve the
same problem. It is intentionally 15-minute-only because the senior reference
contract is 96 slots/day.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from bess.core.common import tariff_vector_day
from bess.core.scenario_gen import MonthData

PPO2_STEPS_PER_DAY = 96
PPO2_DT_HOURS = 0.25
PPO2_DEMAND_BLOCK_SLOTS = 2
ORACLE_COST_RTOL = 1e-6
ORACLE_SIMULTANEITY_TOL_KW = 1e-6


def _require_reference_dt(cfg) -> None:
    if abs(float(cfg.dt) - PPO2_DT_HOURS) > 1e-12:
        raise ValueError("PPO2 senior-reference mode requires exactly 15-minute data (dt=0.25 h)")


def fixed_pmax_day(p_grid_day: np.ndarray) -> float:
    grid = np.maximum(0.0, np.asarray(p_grid_day, dtype=np.float64))
    if len(grid) != PPO2_STEPS_PER_DAY:
        raise ValueError("PPO2 senior-reference days must contain exactly 96 samples")
    return float(grid.reshape(-1, PPO2_DEMAND_BLOCK_SLOTS).mean(axis=1).max(initial=0.0))


def score_month(
    p_grid_days: list[np.ndarray],
    cfg,
    *,
    days: list | None = None,
    p_bess_days: list[np.ndarray] | None = None,
    soc_days: list[np.ndarray] | None = None,
    degradation_cost_per_kwh_discharged: float,
) -> dict:
    """Exact senior objective: energy + fixed-block demand + wear + terminal SOC."""
    _require_reference_dt(cfg)
    energy = 0.0
    pmax = 0.0
    for index, grid_values in enumerate(p_grid_days):
        grid = np.maximum(0.0, np.asarray(grid_values, dtype=np.float64))
        if len(grid) != PPO2_STEPS_PER_DAY:
            raise ValueError("PPO2 senior-reference days must contain exactly 96 samples")
        day = days[index] if days is not None and index < len(days) else None
        tariff = tariff_vector_day(cfg, day) if day is not None else np.asarray(
            [cfg.price_off if t in cfg.OFF else cfg.price_peak if t in cfg.W1 or t in cfg.W2 else cfg.price_mid
             for t in range(PPO2_STEPS_PER_DAY)],
            dtype=np.float64,
        )
        energy += float(np.sum(grid * tariff) * PPO2_DT_HOURS)
        pmax = max(pmax, fixed_pmax_day(grid))

    demand = pmax * cfg.T_cap
    throughput = 0.0
    discharged = 0.0
    if p_bess_days is not None:
        for power_values in p_bess_days:
            power = np.asarray(power_values, dtype=np.float64)
            throughput += float(np.sum(np.abs(power)) * PPO2_DT_HOURS)
            discharged += float(np.sum(np.maximum(0.0, power)) * PPO2_DT_HOURS)
    degradation = discharged * float(degradation_cost_per_kwh_discharged)

    terminal = 0.0
    terminal_soc = None
    if soc_days:
        start_soc = float(np.asarray(soc_days[0], dtype=np.float64)[0])
        terminal_soc = float(np.asarray(soc_days[-1], dtype=np.float64)[-1])
        start_energy = start_soc * cfg.E_cap
        end_energy = terminal_soc * cfg.E_cap
        terminal = cfg.price_off * (
            max(0.0, start_energy - end_energy) / cfg.eta_ch
            - cfg.eta_dis * max(0.0, end_energy - start_energy)
        )

    electricity_bill = energy + demand
    return {
        "energy_cost_vnd": energy,
        "demand_cost_vnd": demand,
        "electricity_bill_vnd": electricity_bill,
        "degradation_cost_vnd": degradation,
        "terminal_settlement_vnd": terminal,
        "total_cost_vnd": electricity_bill + degradation + terminal,
        "throughput_kwh": throughput,
        "discharged_kwh": discharged,
        "equivalent_full_cycles": throughput / max(
            2.0 * cfg.E_cap * (cfg.SOC_max - cfg.SOC_min), 1e-9
        ),
        "terminal_soc_fraction": terminal_soc,
        "pmax_month_kw": pmax,
    }


def run_no_bess(month: MonthData, cfg) -> dict:
    _require_reference_dt(cfg)
    soc_start = min(cfg.SOC_max, cfg.SOC_min + cfg.SOC_safety)
    grids: list[np.ndarray] = []
    socs: list[np.ndarray] = []
    pbs: list[np.ndarray] = []
    for day in month.days:
        if len(day.load) != PPO2_STEPS_PER_DAY or len(day.pv) != PPO2_STEPS_PER_DAY:
            raise ValueError("PPO2 senior-reference days must contain exactly 96 samples")
        grids.append(np.maximum(0.0, day.load - day.pv))
        socs.append(np.full(PPO2_STEPS_PER_DAY + 1, soc_start, dtype=np.float64))
        pbs.append(np.zeros(PPO2_STEPS_PER_DAY, dtype=np.float64))
    return {"p_grid_days": grids, "soc_days": socs, "p_bess_days": pbs}


def solve_month_lp(
    days,
    cfg,
    *,
    soc_init: float,
    degradation_cost_per_kwh_discharged: float,
) -> dict:
    """Port of senior lp_core.solve_month_lp with this repo's config names."""
    _require_reference_dt(cfg)
    n_days = len(days)
    if n_days == 0:
        raise ValueError("month has no days")
    for day in days:
        if len(day.load) != PPO2_STEPS_PER_DAY or len(day.pv) != PPO2_STEPS_PER_DAY:
            raise ValueError("PPO2 senior-reference days must contain exactly 96 samples")

    steps = PPO2_STEPS_PER_DAY * n_days
    block = PPO2_DEMAND_BLOCK_SLOTS
    n_blocks = steps // block
    energy_cap = cfg.E_cap
    rated = cfg.P_rated_nominal
    eta_c, eta_d = cfg.eta_ch, cfg.eta_dis
    soc_min, soc_max = cfg.SOC_min, cfg.SOC_max
    if not soc_min <= soc_init <= soc_max:
        raise ValueError(f"soc_init {soc_init} outside [{soc_min}, {soc_max}]")

    eff_load = np.concatenate([
        np.maximum(0.0, np.asarray(day.load) - np.asarray(day.pv)) for day in days
    ])
    pv_surplus = np.concatenate([
        np.maximum(0.0, np.asarray(day.pv) - np.asarray(day.load)) for day in days
    ])
    tariff = np.concatenate([tariff_vector_day(cfg, day) for day in days]).astype(np.float64)

    ID, ICG, ICP, ISOC = 0, steps, 2 * steps, 3 * steps
    IPK, IU, IV = 4 * steps, 4 * steps + 1, 4 * steps + 2
    n = 4 * steps + 3

    def soc_idx(k: int) -> int:
        return ISOC + (k - 1)

    c = np.zeros(n)
    c[ICG:ICG + steps] = tariff * PPO2_DT_HOURS
    c[ID:ID + steps] = -tariff * PPO2_DT_HOURS
    c[ID:ID + steps] += float(degradation_cost_per_kwh_discharged) * PPO2_DT_HOURS
    c[IPK] = cfg.T_cap
    c[IU] = cfg.price_off / eta_c
    c[IV] = -cfg.price_off * eta_d
    objective_constant = float(np.sum(tariff * eff_load) * PPO2_DT_HOURS)

    charge_gain = eta_c * PPO2_DT_HOURS / energy_cap
    discharge_drain = PPO2_DT_HOURS / (eta_d * energy_cap)
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_data: list[float] = []
    b_eq = np.zeros(steps + 1)
    for t in range(steps):
        eq_rows += [t, t, t]
        eq_cols += [soc_idx(t + 1), ICG + t, ICP + t]
        eq_data += [1.0, -charge_gain, -charge_gain]
        eq_rows.append(t)
        eq_cols.append(ID + t)
        eq_data.append(discharge_drain)
        if t >= 1:
            eq_rows.append(t)
            eq_cols.append(soc_idx(t))
            eq_data.append(-1.0)
        else:
            b_eq[t] = soc_init
    terminal_row = steps
    eq_rows += [terminal_row, terminal_row, terminal_row]
    eq_cols += [soc_idx(steps), IU, IV]
    eq_data += [-energy_cap, -1.0, 1.0]
    b_eq[terminal_row] = -soc_init * energy_cap
    a_eq = coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(steps + 1, n)).tocsr()

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []
    b_ub = np.empty(2 * steps + n_blocks)
    row = 0
    for t in range(steps):
        ub_rows += [row, row]
        ub_cols += [ID + t, ICG + t]
        ub_data += [1.0, -1.0]
        b_ub[row] = eff_load[t]
        row += 1
    weight = 1.0 / block
    for block_index in range(n_blocks):
        start = block_index * block
        for t in range(start, start + block):
            ub_rows += [row, row]
            ub_cols += [ICG + t, ID + t]
            ub_data += [weight, -weight]
        ub_rows.append(row)
        ub_cols.append(IPK)
        ub_data.append(-1.0)
        b_ub[row] = -weight * float(np.sum(eff_load[start:start + block]))
        row += 1
    for t in range(steps):
        ub_rows += [row, row]
        ub_cols += [ICG + t, ICP + t]
        ub_data += [1.0, 1.0]
        b_ub[row] = rated
        row += 1
    a_ub = coo_matrix((ub_data, (ub_rows, ub_cols)), shape=(row, n)).tocsr()

    bounds = [(0.0, rated)] * steps
    bounds += [(0.0, rated)] * steps
    bounds += [(0.0, min(rated, float(s))) for s in pv_surplus]
    bounds += [(soc_min, soc_max)] * steps
    bounds += [(0.0, None), (0.0, None), (0.0, None)]

    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"PPO2 reference month LP did not solve: {result.message}")

    x = result.x
    discharge = x[ID:ID + steps]
    grid_charge = x[ICG:ICG + steps]
    pv_charge = x[ICP:ICP + steps]
    soc = np.concatenate(([soc_init], x[ISOC:ISOC + steps]))
    p_bess = discharge - (grid_charge + pv_charge)
    p_grid = eff_load + grid_charge - discharge
    p_grid = np.where(p_grid > -1e-6, np.maximum(0.0, p_grid), p_grid)
    return {
        "status": "OK",
        "message": str(result.message),
        "objective_vnd": float(result.fun) + objective_constant,
        "ppk_kw": float(x[IPK]),
        "d": discharge,
        "cg": grid_charge,
        "cp": pv_charge,
        "soc": soc,
        "p_bess": p_bess,
        "p_grid": p_grid,
    }


def _split_by_day(flat: np.ndarray) -> list[np.ndarray]:
    return [
        flat[index:index + PPO2_STEPS_PER_DAY]
        for index in range(0, len(flat), PPO2_STEPS_PER_DAY)
    ]


def run_oracle(month: MonthData, cfg, *, degradation_cost_per_kwh_discharged: float) -> dict:
    soc_init = min(cfg.SOC_max, cfg.SOC_min + cfg.SOC_safety)
    solution = solve_month_lp(
        month.days,
        cfg,
        soc_init=soc_init,
        degradation_cost_per_kwh_discharged=degradation_cost_per_kwh_discharged,
    )
    grids = _split_by_day(np.maximum(0.0, solution["p_grid"]))
    pbs = _split_by_day(solution["p_bess"])
    soc = solution["soc"]
    socs = [
        soc[index * PPO2_STEPS_PER_DAY:(index + 1) * PPO2_STEPS_PER_DAY + 1]
        for index in range(len(month.days))
    ]
    simultaneous_kw = float(np.max(np.minimum(solution["d"], solution["cg"] + solution["cp"])))
    scored = score_month(
        grids,
        cfg,
        days=month.days,
        p_bess_days=pbs,
        soc_days=socs,
        degradation_cost_per_kwh_discharged=degradation_cost_per_kwh_discharged,
    )
    objective = solution["objective_vnd"]
    total = scored["total_cost_vnd"]
    if abs(objective - total) > ORACLE_COST_RTOL * max(abs(total), 1.0):
        raise RuntimeError(f"PPO2 LP objective {objective:.4f} != scored bill {total:.4f}")
    if cfg.T_cap > 0.0:
        ppk = solution["ppk_kw"]
        pmax = scored["pmax_month_kw"]
        if abs(ppk - pmax) > ORACLE_COST_RTOL * max(abs(pmax), 1.0):
            raise RuntimeError(f"PPO2 LP Ppk {ppk:.4f} != scored PMax {pmax:.4f}")
    if simultaneous_kw > ORACLE_SIMULTANEITY_TOL_KW:
        raise RuntimeError(
            f"PPO2 LP charges and discharges simultaneously by {simultaneous_kw:.6f} kW"
        )
    if any(np.any(np.asarray(grid) < -1e-6) for grid in grids):
        raise RuntimeError("PPO2 LP violates the senior zero-export constraint")
    if any(
        np.any(np.asarray(soc_day) < cfg.SOC_min - 1e-4)
        or np.any(np.asarray(soc_day) > cfg.SOC_max + 1e-4)
        for soc_day in socs
    ):
        raise RuntimeError("PPO2 LP violates the senior SOC bounds")
    return {
        "p_grid_days": grids,
        "soc_days": socs,
        "p_bess_days": pbs,
        "solver_status": solution["status"],
        "solver_message": solution["message"],
        "lp_objective_vnd": solution["objective_vnd"],
        "ppk_lp_kw": solution["ppk_kw"] if cfg.T_cap > 0.0 else None,
        "terminal_soc": float(soc[-1]),
        "max_simultaneous_kw": simultaneous_kw,
        "valid_for_benchmark": True,
    }
