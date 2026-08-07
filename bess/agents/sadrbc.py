"""SADRBC v13 - P0 fixes vs v12.

* C1 Intra-day replan loop (closed-loop with forecast updates).
* C2 INFEASIBLE days now compute the actual baseline cost (BESS idle).
* C3 PMax_running_month auto-resets on calendar-month boundary.
* C4 Forward dispatch charges across the FULL OFF window (00-04 + 22-24).
* H5 Forecast input validation. H6 SOC_SAFETY_BUFFER replaces hard-coded 0.05.
"""
from __future__ import annotations
from datetime import datetime
import math

# =====================================================================
# Default configuration: derived from the one canonical settings source.
# =====================================================================
from bess.core.settings import SYSTEM_CONFIG
from bess.core.timebase import build_tariff_windows, demand_window_steps


def _system_default_config():
    battery = SYSTEM_CONFIG["BESS"]
    tariff = SYSTEM_CONFIG["Tariff"]
    time_config = SYSTEM_CONFIG["Time"]
    operation = SYSTEM_CONFIG["Operation"]
    safety = SYSTEM_CONFIG["Safety"]
    dt_hours = float(time_config["dt_hours"])
    return {
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
        "P_target_user_kW": operation["P_target_user_kW"],
        "ENABLE_EXPORT": tariff["ENABLE_EXPORT"],
        "FIT_PRICE": tariff["FIT_PRICE_VND_per_kWh"],
        "ENABLE_EVENING_CHARGE": False,
        "V_NOMINAL": safety["V_NOMINAL"],
        "V_BLACKOUT_TH": safety["V_BLACKOUT_TH"],
        "T_DERATE": list(safety["T_DERATE"]),
        "peak_windows": tariff["peak_windows"],
        "off_windows": tariff["off_windows"],
    }


DEFAULT_CONFIG = _system_default_config()


def _resolve_config(cfg=None):
    """Merge caller config over canonical defaults, computing soc_eod if None."""
    out = dict(DEFAULT_CONFIG)
    if cfg:
        out.update(cfg)
    if out.get("soc_eod") is None:
        out["soc_eod"] = out["soc_min"] + out["soc_safety_buffer"]
    return out


class SADRBCConfig:
    """Lightweight holder that exposes resolved config as attributes."""

    def __init__(self, config=None):
        cfg = _resolve_config(config)
        # BESS
        self.E_cap   = float(cfg['E_cap_kWh'])
        self.P_rated_nominal = float(cfg['P_rated_kW'])
        self.eta_ch  = float(cfg['eta_ch'])
        self.eta_dis = float(cfg['eta_dis'])
        self.eta_RT  = self.eta_ch * self.eta_dis
        # SOC (user-configurable)
        self.SOC_min     = float(cfg['soc_min'])
        self.SOC_max     = float(cfg['soc_max'])      # alias of legacy SOC_chg
        self.SOC_chg     = self.SOC_max               # backward-compatible alias
        self.SOC_safety  = float(cfg['soc_safety_buffer'])
        self.SOC_eod     = float(cfg['soc_eod'])
        self.SOC_min_emergency = float(cfg['soc_min_emergency'])
        self.SOC_SAFETY_BUFFER = self.SOC_safety
        # Clock windows are the source of truth. Step indices are derived by set_dt().
        self.peak_windows = str(cfg.get('peak_windows', ''))
        self.off_windows = str(cfg.get('off_windows', ''))
        self.set_dt(float(cfg['dt_hours']))
        # Tariff
        self.price_peak = float(cfg['price_peak'])
        self.price_mid  = float(cfg['price_mid'])
        self.price_off  = float(cfg['price_off'])
        self.T_cap      = float(cfg['T_cap'])
        # Operation
        self.ENABLE_EVENING_CHARGE = bool(cfg.get('ENABLE_EVENING_CHARGE', False))
        self.P_target_user = float(cfg['P_target_user_kW'])
        self.FIT_PRICE     = float(cfg['FIT_PRICE'])
        self.ENABLE_EXPORT = bool(cfg['ENABLE_EXPORT'])
        # Safety
        self.V_NOMINAL     = float(cfg['V_NOMINAL'])
        self.V_BLACKOUT_TH = float(cfg['V_BLACKOUT_TH'])
        self.T_DERATE      = list(cfg['T_DERATE'])
        # Derived
        self.spread_peak_off = self.price_peak - self.price_off / self.eta_RT
        self.ARB_ENABLED     = self.spread_peak_off > 0

    def set_dt(self, dt_hours: float) -> None:
        """Set native timestep and rebuild all tariff step indices from clock time."""
        self.dt = float(dt_hours)
        windows = build_tariff_windows(self.peak_windows, self.off_windows, self.dt)
        self.W1 = windows["W1"]
        self.W2 = windows["W2"]
        self.INTER = windows["INTER"]
        self.OFF = windows["OFF"]
        self.W1_START = windows["W1_START"]
        self.W2_START = windows["W2_START"]
        self.OFF_PEAK_END_STEP = windows["OFF_PEAK_END_STEP"]


# One private fallback config for helper functions that allow cfg=None.
_DEFAULTS = SADRBCConfig()


# =====================================================================
# Billing-aligned 30-minute rolling PMax
# =====================================================================
def compute_PMax_30min_rolling(p_grid, dt_h=None):
    if dt_h is None:
        dt_h = _DEFAULTS.dt
    win = demand_window_steps(dt_h)
    if len(p_grid) < win:
        return max(p_grid) if p_grid else 0.0
    rolling_max = 0.0
    for i in range(0, len(p_grid) - win + 1):
        avg = sum(p_grid[i:i + win]) / win
        if avg > rolling_max:
            rolling_max = avg
    return rolling_max


def thermal_derate(T_battery, P_nominal=None, T_DERATE_table=None):
    if P_nominal is None:
        P_nominal = _DEFAULTS.P_rated_nominal
    if T_DERATE_table is None:
        T_DERATE_table = _DEFAULTS.T_DERATE
    for T_pt, factor in T_DERATE_table:
        if T_battery <= T_pt:
            return factor * P_nominal
    return 0.0


def detect_blackout(V_grid, V_NOMINAL_=None, V_BLACKOUT_TH_=None):
    if V_NOMINAL_ is None:
        V_NOMINAL_ = _DEFAULTS.V_NOMINAL
    if V_BLACKOUT_TH_ is None:
        V_BLACKOUT_TH_ = _DEFAULTS.V_BLACKOUT_TH
    return V_grid < V_BLACKOUT_TH_ * V_NOMINAL_


def check_rest_mode(cycle_3day_history):
    return sum(cycle_3day_history) > 3.0


def _days_in_month_for(year, month):
    """Return number of days in (year, month)."""
    import calendar
    return calendar.monthrange(year, month)[1]


def replan_trigger(minutes_since_plan, fc_err_pv, fc_err_load, soc_drift):
    if abs(fc_err_pv) > 0.5 or soc_drift > 0.10:
        return 'TIER3_URGENT'
    if abs(fc_err_pv) > 0.3 or abs(fc_err_load) > 0.25:
        return 'TIER2'
    if minutes_since_plan >= 15:
        return 'TIER1'
    return None


def _validate_forecast(P_load, P_pv, name='forecast'):
    """Validate same-length, finite, non-negative forecast arrays."""
    if len(P_load) == 0:
        raise ValueError(f'{name}: P_load is empty')
    if len(P_load) != len(P_pv):
        raise ValueError(f'{name}: P_load len={len(P_load)} but P_pv len={len(P_pv)}')
    for arr, lbl in ((P_load, 'P_load'), (P_pv, 'P_pv')):
        for i, v in enumerate(arr):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                raise ValueError(f'{name}: {lbl}[{i}] is NaN/None')
            if v < 0:
                raise ValueError(f'{name}: {lbl}[{i}]={v} (negative)')


# =====================================================================
# Backward pass and Pool 2 accounting
# =====================================================================
def backward_pass(P_load, P_pv, P_target, cfg=None,
                  t_start=0, SOC_eod_target=None):
    """Build the reserve floor from t_start through the dynamic day horizon."""
    cfg = cfg or _DEFAULTS
    n_steps = len(P_load)
    SOC_floor = [0.0] * (n_steps + 1)
    if SOC_eod_target is None:
        SOC_eod_target = cfg.SOC_min + cfg.SOC_safety
    SOC_floor[n_steps] = SOC_eod_target
    for t in range(n_steps - 1, t_start - 1, -1):
        p_shave = max(0, P_load[t] - P_pv[t] - P_target)
        if p_shave > 0:
            dSOC = -p_shave * cfg.dt / (cfg.E_cap * cfg.eta_dis)
        elif P_pv[t] > P_load[t] and t < cfg.W2_START:
            pv_du = min(cfg.P_rated_nominal, P_pv[t] - P_load[t])
            dSOC = +pv_du * cfg.dt * cfg.eta_ch / cfg.E_cap
        else:
            dSOC = 0.0
        SOC_floor[t] = max(cfg.SOC_min + cfg.SOC_safety,
                           SOC_floor[t + 1] - dSOC)
    return SOC_floor


def compute_pool2_extended(P_load, P_pv, P_rated_eff, cfg=None):
    """Pool 2 includes all PV surplus from start of day up to W2."""
    cfg = cfg or _DEFAULTS
    E_pool2 = 0.0
    for t in range(0, min(cfg.W2_START, len(P_load))):
        pv_du = max(0, P_pv[t] - P_load[t])
        E_pool2 += min(P_rated_eff, pv_du) * cfg.dt * cfg.eta_ch
    return E_pool2


def compute_E_shave_total(P_load, P_pv, P_target, cfg=None):
    cfg = cfg or _DEFAULTS
    return sum(max(0, P_load[t] - P_pv[t] - P_target) * cfg.dt for t in range(len(P_load)))


def compute_E_shave_window(P_load, P_pv, P_target, window, cfg=None):
    cfg = cfg or _DEFAULTS
    return sum(max(0, P_load[t] - P_pv[t] - P_target) * cfg.dt for t in window if t < len(P_load))


def compute_E_dis_max_window(P_rated_eff, window, cfg=None, n_steps=None):
    cfg = cfg or _DEFAULTS
    valid_steps = [t for t in window if n_steps is None or t < n_steps]
    return P_rated_eff * cfg.eta_dis * cfg.dt * len(valid_steps)


def classify_case(E_pool2_delivered, E_dis_max_17_22, E_shave_17_22):
    if E_pool2_delivered > E_dis_max_17_22:
        return 'A'
    elif E_pool2_delivered > E_shave_17_22:
        return 'B'
    else:
        return 'C'


def compute_SOC_target_4am(SOC_floor_off_peak_end, W1_arb_planned, cfg=None):
    cfg = cfg or _DEFAULTS
    delta_SOC_arb = W1_arb_planned / (cfg.E_cap * cfg.eta_dis)
    SOC_target = SOC_floor_off_peak_end + delta_SOC_arb + cfg.SOC_SAFETY_BUFFER
    return min(cfg.SOC_max, SOC_target)


def plan_W1_arb(case, E_pool2_delivered, E_dis_max_17_22, E_dis_max_W1, cfg=None):
    cfg = cfg or _DEFAULTS
    if case == 'A' and cfg.ARB_ENABLED:
        surplus_after_W2 = max(0.0, E_pool2_delivered - E_dis_max_17_22)
        return min(E_dis_max_W1, surplus_after_W2)
    return 0.0


# =====================================================================
# W2 off-peak arbitrage planner
# =====================================================================
def compute_W2_arb_offpeak(E_dis_max_W2, E_pool2_to_W2, E_shave_W2,
                           BESS_capacity_room):
    """
    Plan extra W2 peak discharge sourced from off-peak charge when BESS
    still has free capacity AFTER Pool 2 allocation. Spread > 0 means
    arbitrage is positive-NPV by default.
    """
    W2_remaining_capacity = E_dis_max_W2 - E_pool2_to_W2 - E_shave_W2
    if W2_remaining_capacity > 0 and BESS_capacity_room > 0:
        return min(W2_remaining_capacity, BESS_capacity_room)
    return 0.0


def determine_off_peak_charge_target(SOC_floor, day_type, day_type_next,
                                     W1_arb_planned=0.0,
                                     W2_arb_offpeak=0.0, cfg=None):
    """
    SOC target at the end of the off-peak charge window. The target is
    lifted by both W1 arbitrage and any planned W2 off-peak arbitrage
    so the morning charge holds enough energy for the afternoon peak.
    """
    cfg = cfg or _DEFAULTS
    is_off_today = day_type in ('weekend', 'holiday')
    is_off_tmrw  = day_type_next in ('weekend', 'holiday')
    off_end = min(cfg.OFF_PEAK_END_STEP, len(SOC_floor) - 1)
    if is_off_today and is_off_tmrw:
        return cfg.SOC_eod
    if is_off_today and not is_off_tmrw:
        return min(cfg.SOC_max, SOC_floor[off_end] + cfg.SOC_SAFETY_BUFFER)
    delta_SOC_arb = (W1_arb_planned + W2_arb_offpeak) / (cfg.E_cap * cfg.eta_dis)
    SOC_target = SOC_floor[off_end] + delta_SOC_arb + cfg.SOC_SAFETY_BUFFER
    return min(cfg.SOC_max, SOC_target)


def early_infeasible(P_load, P_pv, P_target, P_rated_eff, cfg=None):
    cfg = cfg or _DEFAULTS
    E_shave   = compute_E_shave_total(P_load, P_pv, P_target, cfg)
    E_cap_use = cfg.E_cap * (cfg.SOC_max - cfg.SOC_min) * cfg.eta_dis
    E_pool2   = compute_pool2_extended(P_load, P_pv, P_rated_eff, cfg)
    E_total   = E_cap_use + E_pool2
    return E_shave > E_total * 1.02


def forward_dispatch(P_load, P_pv, SOC_floor, P_target,
                     P_rated_eff, SOC_target_4am, blackout_steps=None,
                     rest_mode=False, cfg=None,
                     t_start=0, SOC_start=None,
                     SOC_history=None, p_bess_history=None,
                     p_grid_history=None):
    """v13 C1: t_start/SOC_start enable intra-day replan."""
    cfg = cfg or _DEFAULTS
    n_steps = len(P_load)
    SOC = list(SOC_history) if SOC_history else [0.0] * (n_steps + 1)
    p_bess = list(p_bess_history) if p_bess_history else [0.0] * n_steps
    p_grid = list(p_grid_history) if p_grid_history else [0.0] * n_steps
    if SOC_start is None:
        SOC_start = cfg.SOC_eod
    SOC[t_start] = SOC_start
    blackout_steps = blackout_steps or set()

    for t in range(t_start, n_steps):
        if t in blackout_steps:
            P_net = max(0, P_load[t] - P_pv[t])
            min_SOC = cfg.SOC_min_emergency
            p_dis_cap = max(0, (SOC[t] - min_SOC) * cfg.E_cap * cfg.eta_dis / cfg.dt)
            p_dis = min(P_rated_eff * cfg.eta_dis, P_net, p_dis_cap)
            p_bess[t] = +p_dis
            p_grid[t] = 0.0
        elif (((t < cfg.OFF_PEAK_END_STEP)
                or (cfg.ENABLE_EVENING_CHARGE and t in cfg.OFF))
              and SOC[t] < SOC_target_4am and not rest_mode):
            # v13 C4: morning OFF (00-04) always charges. Evening OFF
            # (22-24) charges only if ENABLE_EVENING_CHARGE flag is set
            # (default off  current sizing makes morning sufficient).
            p_ch = min(P_rated_eff,
                       (SOC_target_4am - SOC[t]) * cfg.E_cap / (cfg.eta_ch * cfg.dt))
            headroom = P_target - P_load[t]
            p_ch = max(0, min(p_ch, headroom))
            p_bess[t] = -p_ch
            p_grid[t] = P_load[t] + p_ch
        elif P_pv[t] > P_load[t]:
            pv_du = P_pv[t] - P_load[t]
            room = max(0, (cfg.SOC_max - SOC[t]) * cfg.E_cap)
            p_ch = min(P_rated_eff, pv_du, room / (cfg.eta_ch * cfg.dt))
            p_bess[t] = -p_ch
            p_grid[t] = max(0, P_load[t] - P_pv[t])
        elif P_load[t] - P_pv[t] > P_target:
            p_shave = P_load[t] - P_pv[t] - P_target
            if rest_mode and p_shave < 0.05 * P_target:
                p_shave = 0.0
            p_dis = min(P_rated_eff * cfg.eta_dis, p_shave)
            new_SOC = SOC[t] - p_dis * cfg.dt / (cfg.E_cap * cfg.eta_dis)
            if new_SOC < SOC_floor[t + 1]:
                p_dis = max(0, (SOC[t] - SOC_floor[t + 1]) * cfg.E_cap * cfg.eta_dis / cfg.dt)
            p_bess[t] = +p_dis
            p_grid[t] = P_load[t] - P_pv[t] - p_dis
        elif (t in cfg.W1 or t in cfg.W2) and not rest_mode and cfg.ARB_ENABLED:
            # Arbitrage discharge in W1 / W2 - draws on the extra
            # off-peak charge accumulated overnight.
            P_net_t = max(0, P_load[t] - P_pv[t])
            p_dis = min(P_rated_eff * cfg.eta_dis, P_net_t)
            new_SOC = SOC[t] - p_dis * cfg.dt / (cfg.E_cap * cfg.eta_dis)
            min_SOC_required = SOC_floor[t + 1] if t + 1 < n_steps else cfg.SOC_eod
            if new_SOC < min_SOC_required:
                p_dis = max(0, (SOC[t] - min_SOC_required) * cfg.E_cap * cfg.eta_dis / cfg.dt)
            p_bess[t] = +p_dis
            p_grid[t] = P_load[t] - P_pv[t] - p_dis
        else:
            # Hold SOC during off-peak overnight (no discharge).
            p_bess[t] = 0.0
            p_grid[t] = max(0, P_load[t] - P_pv[t])

        if p_bess[t] < 0:
            SOC[t + 1] = SOC[t] + (-p_bess[t]) * cfg.dt * cfg.eta_ch / cfg.E_cap
        else:
            SOC[t + 1] = SOC[t] - p_bess[t] * cfg.dt / (cfg.E_cap * cfg.eta_dis)
        SOC[t + 1] = max(cfg.SOC_min_emergency, min(cfg.SOC_max, SOC[t + 1]))

    return SOC, p_bess, p_grid


def tariff_for_step(t, cfg=None):
    cfg = cfg or _DEFAULTS
    if t in cfg.OFF:           return cfg.price_off
    if t in cfg.W1 or t in cfg.W2: return cfg.price_peak
    return cfg.price_mid


def compute_kpi(P_load, P_pv, SOC, p_bess, p_grid, day_type, case=None,
                cfg=None, days_in_month=None):
    """v13 H3+H4: throughput-based cycle counter + days_in_month prorate."""
    cfg = cfg or _DEFAULTS
    n_steps = min(len(P_load), len(P_pv), len(p_bess), len(p_grid))
    if days_in_month is None:
        days_in_month = 30      # backward-compat default
    PMax_today    = compute_PMax_30min_rolling(p_grid, cfg.dt)
    energy_cost   = sum(max(0, p_grid[t]) * cfg.dt * tariff_for_step(t, cfg) for t in range(n_steps))
    capacity_cost = PMax_today * cfg.T_cap / float(days_in_month)
    # H3 FIX: cycle = discharged_kWh / E_cap (industry-standard EFC)
    discharged_kWh = sum(max(0, p_bess[t]) * cfg.dt for t in range(n_steps))
    cycle_today   = discharged_kWh / cfg.E_cap if cfg.E_cap > 0 else 0.0
    pv_total      = sum(P_pv[t] * cfg.dt for t in range(n_steps))
    pv_used       = sum(min(P_load[t], P_pv[t]) * cfg.dt for t in range(n_steps))
    pv_to_bess    = sum(-p_bess[t] * cfg.dt for t in range(n_steps)
                        if p_bess[t] < 0 and P_pv[t] > P_load[t])
    pv_curtailed  = max(0.0, pv_total - pv_used - pv_to_bess)
    fit_revenue = pv_curtailed * cfg.FIT_PRICE if cfg.ENABLE_EXPORT else 0.0
    total_cost  = energy_cost + capacity_cost - fit_revenue
    return {
        'PMax':         round(PMax_today, 2),
        'energy_cost':  round(energy_cost, 0),
        'capacity_cost':round(capacity_cost, 0),
        'fit_revenue':  round(fit_revenue, 0),
        'total_cost':   round(total_cost, 0),
        'cycle':        round(cycle_today, 3),
        'pv_curtailed': round(pv_curtailed, 2),
        'SOC_eod':      round(SOC[n_steps], 4),
        'case':         case if case else '-',
    }


def baseline_kpi(P_load, P_pv, cfg=None, days_in_month=None):
    """v13 H4: days_in_month prorate."""
    cfg = cfg or _DEFAULTS
    n_steps = min(len(P_load), len(P_pv))
    if days_in_month is None:
        days_in_month = 30
    p_grid = [max(0, P_load[t] - P_pv[t]) for t in range(n_steps)]
    PMax = compute_PMax_30min_rolling(p_grid, cfg.dt)
    energy_cost = sum(p_grid[t] * cfg.dt * tariff_for_step(t, cfg) for t in range(n_steps))
    capacity_cost = PMax * cfg.T_cap / float(days_in_month)
    return {
        'PMax_base':          round(PMax, 2),
        'energy_cost_base':   round(energy_cost, 0),
        'capacity_cost_base': round(capacity_cost, 0),
        'total_cost_base':    round(energy_cost + capacity_cost, 0),
    }


def _build_p_plan(p_bess):
    return [round(float(v), 3) for v in p_bess]


def phase3_strategic_plan(P_load, P_pv, day_type, day_type_next=None,
                          T_battery=30.0, V_grid=1.0, blackout_steps=None,
                          PMax_running_month=None, cycle_3day=None,
                          plan_timestamp=None, cfg=None,
                          t_start=0, SOC_start=None,
                          SOC_history=None, p_bess_history=None,
                          p_grid_history=None, validate=True):
    """v13: t_start/SOC_start/history enable intra-day replan (C1)."""
    cfg = cfg or _DEFAULTS
    if day_type_next is None:
        # v13 H1: warn  caller should pass day_type_next explicitly so
        # weekendworking transition is handled correctly. Falling back
        # to 'working' may inflate SOC_target_4am unnecessarily.
        day_type_next = 'working'
        if not getattr(cfg, '_h1_warned', False):
            import warnings
            warnings.warn('phase3_strategic_plan: day_type_next not '
                          'provided; defaulting to "working". Pass it '
                          'from the calendar for accurate planning.',
                          stacklevel=2)
            cfg._h1_warned = True
    if validate:
        _validate_forecast(P_load, P_pv, name='phase3')
    n_steps = len(P_load)
    if PMax_running_month is None:
        PMax_running_month = cfg.P_target_user
    cycle_3day = cycle_3day if cycle_3day is not None else []
    plan_timestamp = plan_timestamp or datetime.now().isoformat(timespec='seconds')

    P_rated_eff = thermal_derate(T_battery, cfg.P_rated_nominal, cfg.T_DERATE)
    derate_active = P_rated_eff < cfg.P_rated_nominal
    rest_mode = (check_rest_mode(cycle_3day) if len(cycle_3day) >= 3 else False)
    if blackout_steps is None:
        blackout_steps = set()
    if detect_blackout(V_grid, cfg.V_NOMINAL, cfg.V_BLACKOUT_TH):
        blackout_steps = set(range(n_steps))
    UPS_mode = bool(blackout_steps)

    P_target_eff = max(cfg.P_target_user, PMax_running_month)
    SOC_floor = backward_pass(P_load, P_pv, P_target_eff, cfg, t_start=t_start)

    if (P_rated_eff <= 0
            or early_infeasible(P_load, P_pv, P_target_eff, P_rated_eff, cfg)
            or SOC_floor[min(cfg.OFF_PEAK_END_STEP, n_steps)] > cfg.SOC_max):
        case_id = 'INFEAS'
        soc_trace = [cfg.SOC_eod] * (n_steps + 1)
        p_bess_trace = [0.0] * n_steps
        return {
            'execution': {
                'p_plan': _build_p_plan(p_bess_trace),
                'SOC_plan': [round(float(v), 4) for v in soc_trace],
                'SOC_floor': [round(float(v), 4) for v in SOC_floor],
                'P_target_eff': float(P_target_eff),
                'dt': cfg.dt,
                'ARB_ENABLED': cfg.ARB_ENABLED,
            },
            'replan_ref': {
                'forecast_PV': [float(v) for v in P_pv],
                'forecast_load': [float(v) for v in P_load],
                'case_id': case_id,
                'expected_KPIs': {
                    'cycle_predicted': 0.0,
                    'savings_predicted': 0.0,
                    'PMax_predicted': float('nan'),
                },
                'plan_timestamp': plan_timestamp,
            },
            'audit_log': {
                'W1_arb': 0.0,
                'W2_arb': 0.0,
                'W2_arb_pool2': 0.0,
                'W2_arb_offpeak': 0.0,
                'SOC_target_4am': float(cfg.SOC_eod),
                'cycle_predicted': 0.0,
                'E_pv_curtailed_predicted': 0.0,
                'case_id': case_id,
                'mode_flags': {
                    'rest_mode': rest_mode,
                    'UPS_mode': UPS_mode,
                    'derate_active': derate_active,
                    'offpeak_arb_active': False,
                },
            },
        }

    E_pool2     = compute_pool2_extended(P_load, P_pv, P_rated_eff, cfg)
    E_shave_W2  = compute_E_shave_window(P_load, P_pv, P_target_eff, cfg.W2, cfg)
    E_shave_total = compute_E_shave_total(P_load, P_pv, P_target_eff, cfg)
    E_dis_W2    = compute_E_dis_max_window(P_rated_eff, cfg.W2, cfg, n_steps)
    E_dis_W1    = compute_E_dis_max_window(P_rated_eff, cfg.W1, cfg, n_steps)
    case_id     = classify_case(E_pool2, E_dis_W2, E_shave_W2)
    BESS_useful = (cfg.SOC_max - cfg.SOC_min) * cfg.E_cap * cfg.eta_dis  # max usable from full charge

    # Plan W1 / W2 arbitrage by case.
    if case_id == 'A':
        W1_arb         = plan_W1_arb(case_id, E_pool2, E_dis_W2, E_dis_W1, cfg)
        W2_arb_pool2   = min(E_pool2, E_dis_W2)
        W2_arb_offpeak = 0.0
        deficit_offpeak = 0.0
    elif case_id == 'B':
        W1_arb       = 0.0
        W2_arb_pool2 = max(0.0, E_pool2 - E_shave_W2)
        E_pool2_charged = E_pool2
        BESS_room = max(0.0, BESS_useful - E_pool2_charged)
        W2_arb_offpeak = compute_W2_arb_offpeak(
            E_dis_max_W2=E_dis_W2,
            E_pool2_to_W2=W2_arb_pool2 + E_shave_W2,
            E_shave_W2=E_shave_W2,
            BESS_capacity_room=BESS_room,
        )
        deficit_offpeak = 0.0
    else:  # case 'C'
        W1_arb       = 0.0
        W2_arb_pool2 = max(0.0, E_pool2 - E_shave_W2)
        deficit_offpeak = max(0.0, E_shave_W2 - E_pool2)
        BESS_room_after_shave = max(0.0, BESS_useful - E_shave_total)
        W2_arb_offpeak = compute_W2_arb_offpeak(
            E_dis_max_W2=E_dis_W2,
            E_pool2_to_W2=E_pool2,
            E_shave_W2=E_shave_W2,
            BESS_capacity_room=BESS_room_after_shave,
        )

    W2_arb_total = W2_arb_pool2 + W2_arb_offpeak

    SOC_target_4am = determine_off_peak_charge_target(
        SOC_floor, day_type, day_type_next,
        W1_arb_planned=W1_arb,
        W2_arb_offpeak=W2_arb_offpeak + (deficit_offpeak if case_id == 'C' else 0.0),
        cfg=cfg)

    soc_trace, p_bess_trace, p_grid_trace = forward_dispatch(
        P_load, P_pv, SOC_floor, P_target_eff,
        P_rated_eff, SOC_target_4am, blackout_steps, rest_mode, cfg,
        t_start=t_start, SOC_start=SOC_start,
        SOC_history=SOC_history,
        p_bess_history=p_bess_history,
        p_grid_history=p_grid_history)

    audit_case_id = case_id
    offpeak_arb_active = W2_arb_offpeak > 1e-3
    if offpeak_arb_active:
        audit_case_id = f'{case_id}+arb'

    kpi = compute_kpi(P_load, P_pv, soc_trace, p_bess_trace,
                      p_grid_trace, day_type, case=audit_case_id, cfg=cfg)
    base = baseline_kpi(P_load, P_pv, cfg)
    savings_predicted = base['total_cost_base'] - kpi['total_cost']
    PMax_predicted = (float(kpi['PMax']) if isinstance(kpi['PMax'], (int, float))
                      else float('nan'))

    return {
        'execution': {
            'p_plan': _build_p_plan(p_bess_trace),
            'SOC_plan': [round(float(v), 4) for v in soc_trace],
            'SOC_floor': [round(float(v), 4) for v in SOC_floor],
            'P_target_eff': float(P_target_eff),
            'dt': cfg.dt,
            'ARB_ENABLED': cfg.ARB_ENABLED,
        },
        'replan_ref': {
            'forecast_PV': [float(v) for v in P_pv],
            'forecast_load': [float(v) for v in P_load],
            'case_id': audit_case_id,
            'expected_KPIs': {
                'cycle_predicted': float(kpi['cycle']),
                'savings_predicted': float(savings_predicted),
                'PMax_predicted': PMax_predicted,
            },
            'plan_timestamp': plan_timestamp,
        },
        'audit_log': {
            'W1_arb': float(W1_arb),
            'W2_arb': float(W2_arb_total),
            'W2_arb_pool2': float(W2_arb_pool2),
            'W2_arb_offpeak': float(W2_arb_offpeak),
            'SOC_target_4am': float(SOC_target_4am),
            'cycle_predicted': float(kpi['cycle']),
            'E_pv_curtailed_predicted': float(kpi['pv_curtailed']),
            'case_id': audit_case_id,
            'mode_flags': {
                'rest_mode': rest_mode,
                'UPS_mode': UPS_mode,
                'derate_active': derate_active,
                'offpeak_arb_active': offpeak_arb_active,
            },
        },
        '_internal': {
            'kpi': kpi,
            'soc_trace': soc_trace,
            'p_bess_trace': p_bess_trace,
            'p_grid_trace': p_grid_trace,
        },
    }


class SADRBCRunner:
    """
    Day-by-day controller wrapper. Pass an optional `config` dict to
    override SOC bounds, BESS sizing, tariff prices, etc.
    """

    def __init__(self, config=None):
        self.cfg = SADRBCConfig(config)
        self.PMax_running_month = self.cfg.P_target_user
        self.cycle_3day = []
        # v13 C3 FIX: track calendar month for auto-reset.
        self._current_month = None
        self._monthly_log = []

    def reset_monthly(self, reason='month_boundary'):
        """v13 C3: reset PMax_running_month at month boundary."""
        if self._current_month is not None:
            self._monthly_log.append({
                'month': self._current_month,
                'peak_committed': float(self.PMax_running_month),
                'reason': reason,
            })
        self.PMax_running_month = self.cfg.P_target_user

    # Convenience properties exposing the user-configurable SOC bounds.
    @property
    def SOC_min(self):     return self.cfg.SOC_min
    @property
    def SOC_max(self):     return self.cfg.SOC_max
    @property
    def SOC_safety(self):  return self.cfg.SOC_safety
    @property
    def SOC_eod(self):     return self.cfg.SOC_eod

    def step_day(self, P_load, P_pv, day_type, day_type_next=None,
                 T_battery=30.0, V_grid=1.0, blackout_steps=None,
                 plan_timestamp=None, days_in_month=None):
        # v13 H4: derive days_in_month from plan_timestamp if not given.
        # v13 C3 FIX: detect calendar-month rollover and reset PMax.
        if plan_timestamp:
            try:
                ts = (datetime.fromisoformat(plan_timestamp)
                      if isinstance(plan_timestamp, str) else plan_timestamp)
                this_month = (ts.year, ts.month)
                if (self._current_month is not None
                        and this_month != self._current_month):
                    self.reset_monthly(reason='month_boundary')
                self._current_month = this_month
                if days_in_month is None:
                    days_in_month = _days_in_month_for(ts.year, ts.month)
            except (ValueError, AttributeError):
                pass
        if days_in_month is None:
            days_in_month = 30
        phase3_out = phase3_strategic_plan(
            P_load, P_pv, day_type, day_type_next,
            T_battery=T_battery, V_grid=V_grid, blackout_steps=blackout_steps,
            PMax_running_month=self.PMax_running_month,
            cycle_3day=list(self.cycle_3day),
            plan_timestamp=plan_timestamp,
            cfg=self.cfg,
        )
        case_id = phase3_out['replan_ref']['case_id']
        if case_id == 'INFEAS':
            # v13 C2 FIX: BESS idle on infeas day still incurs full cost.
            n_steps = min(len(P_load), len(P_pv))
            soc_trace    = [self.cfg.SOC_eod] * (n_steps + 1)
            p_bess_trace = [0.0] * n_steps
            p_grid_trace = [max(0, P_load[t] - P_pv[t]) for t in range(n_steps)]
            kpi_actual = compute_kpi(P_load, P_pv, soc_trace,
                                     p_bess_trace, p_grid_trace,
                                     day_type, case='INFEAS', cfg=self.cfg,
                                     days_in_month=days_in_month)
            kpi = {
                'INFEASIBLE': True,
                'PMax': kpi_actual['PMax'],
                'energy_cost': kpi_actual['energy_cost'],
                'capacity_cost': kpi_actual['capacity_cost'],
                'fit_revenue': kpi_actual['fit_revenue'],
                'total_cost': kpi_actual['total_cost'],
                'cycle': 0,
                'pv_curtailed': kpi_actual['pv_curtailed'],
                'SOC_eod': self.cfg.SOC_eod,
                'case': 'INFEAS',
            }
            if isinstance(kpi['PMax'], (int, float)):
                self.PMax_running_month = max(self.PMax_running_month,
                                              kpi['PMax'])
            self._update_cycle_history(0.0)
            return kpi, soc_trace, p_bess_trace, p_grid_trace, phase3_out

        internal     = phase3_out['_internal']
        kpi          = internal['kpi']
        soc_trace    = internal['soc_trace']
        p_bess_trace = internal['p_bess_trace']
        p_grid_trace = internal['p_grid_trace']

        if isinstance(kpi['PMax'], (int, float)):
            self.PMax_running_month = max(self.PMax_running_month, kpi['PMax'])
        self._update_cycle_history(kpi['cycle'])
        return kpi, soc_trace, p_bess_trace, p_grid_trace, phase3_out

    def _update_cycle_history(self, c):
        self.cycle_3day.append(c)
        if len(self.cycle_3day) > 3:
            self.cycle_3day = self.cycle_3day[-3:]

    # ============================================================
    # v13 C1: intra-day replan + closed-loop simulator
    # ============================================================
    def step_intraday(self, t_now, SOC_now,
                      P_load_actual, P_pv_actual,
                      P_load_fc, P_pv_fc,
                      day_type, day_type_next='working',
                      SOC_history=None, p_bess_history=None,
                      p_grid_history=None,
                      T_battery=30.0, V_grid=1.0, blackout_steps=None,
                      plan_timestamp=None):
        """Recompute the plan from t_now to the dynamic end-of-day step."""
        n_steps = min(len(P_load_actual), len(P_pv_actual), len(P_load_fc), len(P_pv_fc))
        if not (1 <= t_now < n_steps):
            raise ValueError(f'step_intraday: t_now={t_now} (1..{n_steps - 1})')
        P_load = list(P_load_actual[:t_now]) + list(P_load_fc[t_now:])
        P_pv   = list(P_pv_actual[:t_now])  + list(P_pv_fc[t_now:])
        _validate_forecast(P_load, P_pv, name='replan')
        return phase3_strategic_plan(
            P_load, P_pv, day_type, day_type_next,
            T_battery=T_battery, V_grid=V_grid, blackout_steps=blackout_steps,
            PMax_running_month=self.PMax_running_month,
            cycle_3day=list(self.cycle_3day),
            plan_timestamp=plan_timestamp,
            cfg=self.cfg,
            t_start=t_now, SOC_start=SOC_now,
            SOC_history=SOC_history,
            p_bess_history=p_bess_history,
            p_grid_history=p_grid_history,
            validate=False,
        )

    def simulate_closed_loop(self, P_load_actual, P_pv_actual,
                             P_load_forecast, P_pv_forecast,
                             day_type, day_type_next=None,
                             plan_timestamp=None,
                             replan_cadence_steps=4,
                             T_battery=30.0, V_grid=1.0,
                             blackout_steps=None):
        """Closed-loop daily simulation with replan every ~1h.

        Initial plan uses forecast. At every replan_cadence_steps (and
        immediately on TIER2/TIER3 trigger), the plan is rebuilt with
        the actuals-so-far + forecast-for-rest. Final KPI uses actuals.
        """
        # v13 C3: month boundary check.
        if plan_timestamp:
            try:
                ts = (datetime.fromisoformat(plan_timestamp)
                      if isinstance(plan_timestamp, str) else plan_timestamp)
                this_month = (ts.year, ts.month)
                if (self._current_month is not None
                        and this_month != self._current_month):
                    self.reset_monthly(reason='month_boundary')
                self._current_month = this_month
            except (ValueError, AttributeError):
                pass

        phase3_0 = phase3_strategic_plan(
            P_load_forecast, P_pv_forecast, day_type, day_type_next,
            T_battery=T_battery, V_grid=V_grid, blackout_steps=blackout_steps,
            PMax_running_month=self.PMax_running_month,
            cycle_3day=list(self.cycle_3day),
            plan_timestamp=plan_timestamp, cfg=self.cfg)
        case_id = phase3_0['replan_ref']['case_id']
        replan_log = []
        n_steps = min(len(P_load_actual), len(P_pv_actual), len(P_load_forecast), len(P_pv_forecast))
        one_hour_steps = max(1, int(round(1.0 / self.cfg.dt)))

        if case_id == 'INFEAS':
            soc_trace    = [self.cfg.SOC_eod] * (n_steps + 1)
            p_bess_trace = [0.0] * n_steps
            p_grid_trace = [max(0, P_load_actual[t] - P_pv_actual[t])
                            for t in range(n_steps)]
            kpi = compute_kpi(P_load_actual, P_pv_actual, soc_trace,
                              p_bess_trace, p_grid_trace,
                              day_type, case='INFEAS', cfg=self.cfg)
            kpi['INFEASIBLE'] = True
            if isinstance(kpi['PMax'], (int, float)):
                self.PMax_running_month = max(self.PMax_running_month,
                                              kpi['PMax'])
            self._update_cycle_history(0.0)
            return kpi, soc_trace, p_bess_trace, p_grid_trace, replan_log

        soc_trace    = list(phase3_0['execution']['SOC_plan'])
        p_bess_trace = list(phase3_0['execution']['p_plan'])
        p_grid_trace = list(phase3_0['_internal']['p_grid_trace'])
        soc_trace[0] = self.cfg.SOC_eod
        plan_soc_initial = list(phase3_0['execution']['SOC_plan'])

        steps_since_replan = 0
        for t in range(n_steps):
            p_bess_t = p_bess_trace[t]
            # Recompute SOC update using planned p_bess (exec follows plan).
            if p_bess_t < 0:
                soc_trace[t + 1] = (soc_trace[t]
                                    + (-p_bess_t) * self.cfg.dt
                                    * self.cfg.eta_ch / self.cfg.E_cap)
            else:
                soc_trace[t + 1] = (soc_trace[t]
                                    - p_bess_t * self.cfg.dt
                                    / (self.cfg.E_cap * self.cfg.eta_dis))
            soc_trace[t + 1] = max(self.cfg.SOC_min_emergency,
                                   min(self.cfg.SOC_max, soc_trace[t + 1]))
            # Real grid using actuals.
            net_load = max(0, P_load_actual[t] - P_pv_actual[t])
            if p_bess_t < 0:
                if P_pv_actual[t] > P_load_actual[t]:
                    p_grid_trace[t] = max(0, net_load)
                else:
                    p_grid_trace[t] = P_load_actual[t] + (-p_bess_t)
            else:
                p_grid_trace[t] = max(0, net_load - p_bess_t)
            steps_since_replan += 1
            if t + 1 >= n_steps:
                break
            t_next = t + 1
            window_end = min(t_next + one_hour_steps, n_steps)
            sum_load_fc = sum(P_load_forecast[t_next:window_end]) or 1.0
            sum_pv_fc   = sum(P_pv_forecast[t_next:window_end])   or 1.0
            sum_load_ac = sum(P_load_actual[t_next:window_end])
            sum_pv_ac   = sum(P_pv_actual[t_next:window_end])
            fc_err_load = (sum_load_ac - sum_load_fc) / sum_load_fc
            fc_err_pv   = (sum_pv_ac   - sum_pv_fc)   / sum_pv_fc
            soc_drift = abs(soc_trace[t_next] - plan_soc_initial[t_next])
            tier = replan_trigger(
                minutes_since_plan=steps_since_replan * self.cfg.dt * 60,
                fc_err_pv=fc_err_pv, fc_err_load=fc_err_load,
                soc_drift=soc_drift)
            need_replan = (tier == 'TIER3_URGENT' or tier == 'TIER2'
                           or (tier == 'TIER1'
                               and steps_since_replan >= replan_cadence_steps))
            if need_replan:
                replan_log.append({
                    't': t_next, 'tier': tier,
                    'fc_err_load': round(fc_err_load, 3),
                    'fc_err_pv':   round(fc_err_pv, 3),
                    'soc_drift':   round(soc_drift, 4),
                })
                phase3_new = self.step_intraday(
                    t_now=t_next, SOC_now=soc_trace[t_next],
                    P_load_actual=P_load_actual, P_pv_actual=P_pv_actual,
                    P_load_fc=P_load_forecast, P_pv_fc=P_pv_forecast,
                    day_type=day_type, day_type_next=day_type_next,
                    SOC_history=soc_trace,
                    p_bess_history=p_bess_trace,
                    p_grid_history=p_grid_trace,
                    T_battery=T_battery, V_grid=V_grid,
                    blackout_steps=blackout_steps,
                    plan_timestamp=plan_timestamp)
                if phase3_new['replan_ref']['case_id'] != 'INFEAS':
                    new_pbess = phase3_new['execution']['p_plan']
                    for k in range(t_next, n_steps):
                        p_bess_trace[k] = new_pbess[k]
                steps_since_replan = 0

        kpi = compute_kpi(P_load_actual, P_pv_actual, soc_trace,
                          p_bess_trace, p_grid_trace, day_type,
                          case=phase3_0['replan_ref']['case_id'],
                          cfg=self.cfg)
        if isinstance(kpi['PMax'], (int, float)):
            self.PMax_running_month = max(self.PMax_running_month,
                                          kpi['PMax'])
        self._update_cycle_history(kpi['cycle'])
        return kpi, soc_trace, p_bess_trace, p_grid_trace, replan_log
