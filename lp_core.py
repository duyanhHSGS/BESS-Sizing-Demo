"""lp_core.py  shared 96-slot daily LP for BESS+PV dispatch.

Used by BOTH LP-PF (perfect-foresight upper bound) and Option 2's
day-ahead planner. The objective and constraints are written to match
EXACTLY the cost model that ``sadrbc.compute_kpi`` scores, so any plan
produced here can be fairly compared against SADRBC v13 on the same
30-day harness:

  * energy cost   =  tariff[t]max(0,grid[t])t
  * capacity cost = (T_cap / days_in_month)  PMax,
                    PMax = max 30-min ROLLING-AVERAGE of grid import
  * zero export   = grid[t]  0 t (TT05/2025, ENABLE_EXPORT=False)

Effective-load convention (WHY): PV first offsets load on site; only the
net demand reaches the meter. Excess PV (pv>load) is free energy that may
only charge the battery or be curtailed  never exported.

  eff_load[t]    = max(0, load[t] - pv[t])      # net demand after PV
  pv_surplus[t]  = max(0, pv[t] - load[t])      # free PV for charging

Decision variables (per 96-slot day):
  d[t]    0  discharge to load                       (kW)
  cg[t]   0  charge from grid (costs energy)         (kW)
  cp[t]   0  charge from free PV surplus (cost-free)  (kW),  pv_surplus[t]
  soc[k]      state of charge fraction, k = 1..96
  Ppk     0  billed peak (30-min rolling avg of grid) (kW)

  grid[t] = eff_load[t] + cg[t] - d[t]
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

# Demand charge is billed monthly but the v13 harness amortises it per day
# (capacity_cost = PMax_today  T_cap / days_in_month). We mirror that here
# so the LP optimises the SAME objective the harness scores.


def build_eff_load(P_load, P_pv):
    """Return (eff_load, pv_surplus) arrays of length 96."""
    n_steps = min(len(P_load), len(P_pv))
    eff_load = [max(0.0, P_load[t] - P_pv[t]) for t in range(n_steps)]
    pv_surplus = [max(0.0, P_pv[t] - P_load[t]) for t in range(n_steps)]
    return eff_load, pv_surplus


def solve_day_lp(eff_load, pv_surplus, cfg, demand_rate_per_kW,
                 soc_init, soc_end_min, soc_floor_frac=0.0,
                 degradation_vnd_per_kwh=0.0, peak_floor=None):
    """Solve the daily dispatch LP.

    Parameters
    ----------
    cfg : SADRBCConfig          (E_cap, P_rated_nominal, eta_ch/dis, SOC, dt, tariff windows)
    demand_rate_per_kW : float  capacity rate charged on the peak term.
                                per-day mode: T_cap / days_in_month on Ppk;
                                monthly mode: T_cap on the EXCESS over peak_floor.
    soc_init : float            SOC fraction at start of day (slot 0)
    soc_end_min : float         minimum SOC fraction required at slot 96 (cycle fairness)
    soc_floor_frac : float      reserve floor as fraction of usable band
                                (0 = LP-PF loosest bound; 0.10 = Option 2 reserve)
    degradation_vnd_per_kwh : float  optional throughput penalty (battery health)
    peak_floor : float or None  if set, only the peak ABOVE this floor is billed
                                (Option 2's `P_peak >= month_peak_so_far` design 
                                under real monthly-max EVN billing, today's demand
                                charge is the MARGINAL increment over the month's
                                running peak, so days that don't set a new monthly
                                high should not be shaved at the expense of energy).

    Returns
    -------
    dict with status, and (if feasible) per-slot d, cg, cp, soc(97), Ppk,
    p_bess(96, +dis/-chg harness sign), p_grid(96).
    """
    from benchmark import _demand_windows
    from sadrbc import tariff_for_step

    steps = min(len(eff_load), len(pv_surplus))
    eff_load = list(eff_load[:steps])
    pv_surplus = list(pv_surplus[:steps])
    E = cfg.E_cap
    dt = cfg.dt
    P_rated = cfg.P_rated_nominal
    eta_c = cfg.eta_ch
    eta_d = cfg.eta_dis
    soc_max = cfg.SOC_max
    soc_min = cfg.SOC_min
    soc_lo = soc_min + soc_floor_frac * (soc_max - soc_min)

    tariff = [tariff_for_step(t, cfg) for t in range(steps)]

    # Variable layout
    ID, ICG, ICP, ISOC, IPK = 0, steps, steps * 2, steps * 3, steps * 4
    use_excess = peak_floor is not None
    IPKX = IPK + 1                    # Ppk_excess (monthly mode only)
    n = IPKX + 1 if use_excess else IPK + 1

    def soc_idx(k):  # soc[k], k = 1..96
        return ISOC + (k - 1)

    # ---- Objective (minimise) ----
    c = np.zeros(n)
    for t in range(steps):
        c[ICG + t] += tariff[t] * dt          # buying energy to charge
        c[ID + t] += -tariff[t] * dt          # discharging displaces a grid buy
        # cp (free PV) has zero energy cost; degradation applies to throughput
        c[ID + t] += degradation_vnd_per_kwh * dt
        c[ICG + t] += degradation_vnd_per_kwh * dt
        c[ICP + t] += degradation_vnd_per_kwh * dt
    if use_excess:
        c[IPKX] = demand_rate_per_kW          # bill only the marginal monthly increment
        c[IPK] = 1e-3                          # tiny pin so Ppk tracks the true peak
    else:
        c[IPK] = demand_rate_per_kW
    # NOTE: the constant  tariffeff_loaddt (baseline net-load energy) is
    # dropped from the objective and re-added when reporting cost.

    # ---- Equality: SOC dynamics, t = 0..95 ----
    A_eq = np.zeros((steps, n))
    b_eq = np.zeros(steps)
    a = eta_c * dt / E          # charge gain per kW
    b = dt / (eta_d * E)        # discharge drain per kW
    for t in range(steps):
        A_eq[t, soc_idx(t + 1)] = 1.0
        if t >= 1:
            A_eq[t, soc_idx(t)] = -1.0
        A_eq[t, ICG + t] = -a
        A_eq[t, ICP + t] = -a
        A_eq[t, ID + t] = +b
        b_eq[t] = soc_init if t == 0 else 0.0

    # ---- Inequalities A_ub x  b_ub ----
    rows, bvec = [], []
    # (a) no export: d[t] - cg[t]  eff_load[t]
    for t in range(steps):
        r = np.zeros(n)
        r[ID + t] = 1.0
        r[ICG + t] = -1.0
        rows.append(r)
        bvec.append(eff_load[t])
    # (b) 30-min rolling-avg peak: 0.5(g[t]+g[t+1])  Ppk
    for window in _demand_windows(steps, dt):
        r = np.zeros(n)
        eff_weighted = 0.0
        for k, weight in window:
            r[ICG + k] += weight
            r[ID + k] += -weight
            eff_weighted += weight * eff_load[k]
        r[IPK] = -1.0
        rows.append(r)
        bvec.append(-eff_weighted)
    # (c) charge power: cg[t] + cp[t]  P_rated
    for t in range(steps):
        r = np.zeros(n)
        r[ICG + t] = 1.0
        r[ICP + t] = 1.0
        rows.append(r)
        bvec.append(P_rated)
    # (d) monthly mode: Ppk_excess  Ppk  peak_floor    Ppk  Ppk_excess  floor
    if use_excess:
        r = np.zeros(n)
        r[IPK] = 1.0
        r[IPKX] = -1.0
        rows.append(r)
        bvec.append(peak_floor)
    A_ub = np.array(rows)
    b_ub = np.array(bvec)

    # ---- Bounds ----
    bounds = [(None, None)] * n
    for t in range(steps):
        bounds[ID + t] = (0.0, P_rated)
        bounds[ICG + t] = (0.0, P_rated)
        bounds[ICP + t] = (0.0, min(P_rated, pv_surplus[t]))
    for k in range(1, steps + 1):
        lo = soc_lo
        if k == steps:
            lo = max(soc_lo, soc_end_min)
        bounds[soc_idx(k)] = (lo, soc_max)
    bounds[IPK] = (0.0, None)
    if use_excess:
        bounds[IPKX] = (0.0, None)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return {"status": "INFEAS", "message": res.message}

    x = res.x
    d = [float(x[ID + t]) for t in range(steps)]
    cg = [float(x[ICG + t]) for t in range(steps)]
    cp = [float(x[ICP + t]) for t in range(steps)]
    soc = [soc_init] + [float(x[soc_idx(k)]) for k in range(1, steps + 1)]
    Ppk = float(x[IPK])

    p_bess = [d[t] - (cg[t] + cp[t]) for t in range(steps)]      # +dis / -chg
    p_grid = [eff_load[t] + cg[t] - d[t] for t in range(steps)]  # >= 0
    # Clamp tiny numerical negatives from the solver.
    p_grid = [max(0.0, g) if g > -1e-6 else g for g in p_grid]

    return {
        "status": "OK", "d": d, "cg": cg, "cp": cp,
        "soc": soc, "Ppk": Ppk, "p_bess": p_bess, "p_grid": p_grid,
    }
