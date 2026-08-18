"""Exact senior-reference environment used only by PPO2.

This is a direct logic port of the archived senior causal BESS environment in
git-plz-ignore/senior-ppo2-reference/. The only
changes are attribute/import adapters for this repository's shared BESSConfig and
DayData/MonthData types. The reference contract is intentionally 15-minute-only:
96 slots/day, two slots per fixed 30-minute demand block, and a 17D observation.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date as date_cls

import numpy as np

from bess.core.common import TOU_RULES, is_sunday, tariff_vector_day
from bess.core.scenario_gen import MonthData

PPO2_STEPS_PER_DAY = 96
PPO2_DT_HOURS = 0.25
PPO2_DEMAND_BLOCK_SLOTS = 2
PPO2_OBS_DIM = 17
PPO2_HISTORY_WINDOW_SLOTS = 5
PPO2_EWMA_ALPHA = 0.4


@dataclass(frozen=True)
class PPO2NetLoadHistory:
    ewma_kw: float = 0.0
    window: tuple[float, ...] = ()

    @property
    def is_ready(self) -> bool:
        return len(self.window) >= PPO2_HISTORY_WINDOW_SLOTS

    @property
    def trend_kw_per_hour(self) -> float:
        if not self.is_ready:
            return 0.0
        return self.window[-1] - self.window[-PPO2_HISTORY_WINDOW_SLOTS]

    def update(self, effective_load_kw: float) -> PPO2NetLoadHistory:
        value = float(effective_load_kw)
        ewma = (
            value
            if not self.window
            else PPO2_EWMA_ALPHA * value + (1.0 - PPO2_EWMA_ALPHA) * self.ewma_kw
        )
        window = (*self.window, value)[-PPO2_HISTORY_WINDOW_SLOTS:]
        return PPO2NetLoadHistory(ewma_kw=ewma, window=window)


@dataclass(frozen=True)
class PPO2FeasibleAction:
    mapped_power_kw: float
    discharge_kw: float
    charge_grid_kw: float
    charge_pv_kw: float
    clip_reason: str | None


def _clip_reason(requested_kw: float, rated_kw: float, *limits: float) -> str | None:
    binding = min(limits)
    if requested_kw <= binding + 1e-9 or binding >= rated_kw - 1e-9:
        return None
    return "soc" if binding == limits[0] else "export"


def _map_feasible_action(
    *,
    action_raw: float,
    soc_fraction: float,
    load_kw: float,
    pv_kw: float,
    p_rated_kw: float,
    e_cap_kwh: float,
    soc_min: float,
    soc_max: float,
    eta_charge: float,
    eta_discharge: float,
    dt_hours: float,
    allow_export: bool,
) -> PPO2FeasibleAction:
    action = max(-1.0, min(1.0, float(action_raw)))
    rated = max(0.0, float(p_rated_kw))
    if dt_hours <= 0.0:
        raise ValueError("dt_hours must be > 0")
    requested = action * rated

    effective_load = max(0.0, float(load_kw) - float(pv_kw))
    pv_surplus = max(0.0, float(pv_kw) - float(load_kw))
    discharge_soc_cap = max(
        0.0,
        (float(soc_fraction) - float(soc_min))
        * float(e_cap_kwh)
        * float(eta_discharge)
        / float(dt_hours),
    )
    export_cap = rated if allow_export else effective_load
    max_discharge = min(rated, export_cap, discharge_soc_cap)
    charge_soc_cap = max(
        0.0,
        (float(soc_max) - float(soc_fraction))
        * float(e_cap_kwh)
        / (float(eta_charge) * float(dt_hours)),
    )
    max_charge = min(rated, charge_soc_cap)

    if requested >= 0.0:
        discharge = min(requested, max_discharge)
        return PPO2FeasibleAction(
            mapped_power_kw=discharge,
            discharge_kw=discharge,
            charge_grid_kw=0.0,
            charge_pv_kw=0.0,
            clip_reason=_clip_reason(requested, rated, discharge_soc_cap, export_cap),
        )

    total_charge = min(-requested, max_charge)
    charge_pv = min(total_charge, pv_surplus)
    charge_grid = total_charge - charge_pv
    return PPO2FeasibleAction(
        mapped_power_kw=-total_charge,
        discharge_kw=0.0,
        charge_grid_kw=charge_grid,
        charge_pv_kw=charge_pv,
        clip_reason=_clip_reason(-requested, rated, charge_soc_cap),
    )


class PPO2Env:
    """Senior's causal 15-minute PPO environment, adapted to shared BESSConfig names."""

    def __init__(
        self,
        cfg,
        p_ref_kw: float,
        degradation_cost_per_kwh_discharged: float,
        clip_penalty_per_kwh: float = 0.0,
    ):
        if abs(float(cfg.dt) - PPO2_DT_HOURS) > 1e-12:
            raise ValueError("PPO2 senior-reference mode requires exactly 15-minute data (dt=0.25 h)")
        if cfg.P_rated_nominal <= 0.0 or cfg.E_cap <= 0.0:
            raise ValueError("p_rated_kw and e_cap_kwh must be > 0")
        if p_ref_kw <= 0.0:
            raise ValueError("p_ref_kw must be > 0")
        if clip_penalty_per_kwh < 0.0:
            raise ValueError("clip_penalty_per_kwh must be >= 0")
        self.cfg = cfg
        self.n_steps = PPO2_STEPS_PER_DAY
        self.dt = PPO2_DT_HOURS
        self.block_slots = PPO2_DEMAND_BLOCK_SLOTS
        self.p_ref = float(p_ref_kw)
        self.deg = float(degradation_cost_per_kwh_discharged)
        self.clip_penalty_per_kwh = float(clip_penalty_per_kwh)
        self.reward_scale_vnd = max(
            1.0,
            self.p_ref * cfg.T_cap,
            self.p_ref * cfg.price_peak * self.dt,
        )
        self.obs_dim = PPO2_OBS_DIM
        self.month: MonthData | None = None
        self.log_grid: list[np.ndarray] = []
        self.log_soc: list[np.ndarray] = []
        self.log_pbess: list[np.ndarray] = []

    @property
    def history_ready(self) -> bool:
        return self.net_load.is_ready

    def reset(
        self,
        month: MonthData,
        soc_init: float | None = None,
        d_run_shaping_init_kw: float = 0.0,
    ) -> np.ndarray:
        if d_run_shaping_init_kw < 0.0:
            raise ValueError("d_run_shaping_init_kw must be >= 0")
        if not month.days:
            raise ValueError("month has no days")
        for day in month.days:
            if len(day.load) != self.n_steps or len(day.pv) != self.n_steps:
                raise ValueError("PPO2 senior-reference days must contain exactly 96 samples")
            if not day.date_iso:
                raise ValueError("PPO2 senior-reference mode requires calendar date_iso on every day")

        self.month = month
        self.day = 0
        self.t = 0
        self.soc = float(soc_init if soc_init is not None else self.cfg.SOC_min)
        self.soc_start = self.soc
        self.d_run = 0.0
        self.d_run_nb = 0.0
        self.d_run_shaping = float(d_run_shaping_init_kw)
        self.d_run_shaping_nb = float(d_run_shaping_init_kw)
        self.log_grid = [np.zeros(self.n_steps) for _ in month.days]
        self.log_soc = [np.zeros(self.n_steps + 1) for _ in month.days]
        self.log_pbess = [np.zeros(self.n_steps) for _ in month.days]
        self.total_degradation_cost = 0.0
        self.total_terminal_cost = 0.0
        self.peak_actual_grid_charge_kwh = 0.0
        self.prev_load = 0.0
        self.prev_pv = 0.0
        self.prev_p_bess = 0.0
        self.prev_demand = 0.0
        self.net_load = PPO2NetLoadHistory()
        self.steps_in_month = 0
        self._block_grid_sum = 0.0
        self._block_grid_sum_nb = 0.0
        self.log_soc[0][0] = self.soc
        self._set_day_tariff()
        return self._obs()

    def _set_day_tariff(self) -> None:
        day = self.month.days[self.day]
        self.tariff = tariff_vector_day(self.cfg, day)
        if len(self.tariff) != self.n_steps:
            raise ValueError("PPO2 senior-reference tariff must contain exactly 96 slots")
        peak_steps = set(self.cfg.W1) | set(self.cfg.W2)
        if TOU_RULES.get("sunday_no_peak") and is_sunday(day):
            peak_steps = set()
        self._peak_mask = np.asarray(
            [slot in peak_steps for slot in range(self.n_steps)], dtype=bool
        )

    def _time_to_tariff_transition(self, slot: int) -> float:
        current = self.tariff[slot]
        for offset in range(1, self.n_steps + 1):
            if self.tariff[(slot + offset) % self.n_steps] != current:
                return offset / self.n_steps
        return 1.0

    def _obs(self) -> np.ndarray:
        day = self.month.days[self.day]
        date = date_cls.fromisoformat(str(day.date_iso))
        slot = self.t
        p_ref = self.p_ref
        angle = 2.0 * np.pi * slot / self.n_steps
        effective_load = max(0.0, self.prev_load - self.prev_pv)
        pv_surplus = max(0.0, self.prev_pv - self.prev_load)
        block_phase = float(slot % 2)
        return np.array(
            [
                np.sin(angle),
                np.cos(angle),
                effective_load / p_ref,
                pv_surplus / p_ref,
                self.prev_p_bess / self.cfg.P_rated_nominal,
                self.prev_demand / p_ref,
                self.soc,
                self.tariff[slot] / self.cfg.price_peak,
                self._time_to_tariff_transition(slot),
                self.d_run / p_ref,
                self.d_run_nb / p_ref,
                1.0 if day.day_type == "working" else 0.0,
                (date.day - 1) / calendar.monthrange(date.year, date.month)[1],
                block_phase,
                self._block_grid_sum / p_ref,
                self.net_load.ewma_kw / p_ref,
                self.net_load.trend_kw_per_hour / p_ref,
            ],
            dtype=np.float32,
        )

    def project_action(self, action: float, load: float, pv: float) -> PPO2FeasibleAction:
        cfg = self.cfg
        return _map_feasible_action(
            action_raw=action,
            soc_fraction=self.soc,
            load_kw=load,
            pv_kw=pv,
            p_rated_kw=cfg.P_rated_nominal,
            e_cap_kwh=cfg.E_cap,
            soc_min=cfg.SOC_min,
            soc_max=cfg.SOC_max,
            eta_charge=cfg.eta_ch,
            eta_discharge=cfg.eta_dis,
            dt_hours=self.dt,
            allow_export=bool(cfg.ENABLE_EXPORT),
        )

    def step(self, action: float):
        cfg = self.cfg
        day = self.month.days[self.day]
        t = self.t
        load, pv = float(day.load[t]), float(day.pv[t])
        effective_load = max(0.0, load - pv)
        is_peak_t = bool(self._peak_mask[t])

        held = not self.history_ready
        action_effective = 0.0 if held else max(-1.0, min(1.0, float(action)))
        p_requested = action_effective * cfg.P_rated_nominal
        mapped = self.project_action(action_effective, load, pv)
        discharge = mapped.discharge_kw
        charge_grid = mapped.charge_grid_kw
        charge_pv = mapped.charge_pv_kw
        p_executed = mapped.mapped_power_kw

        grid = effective_load + charge_grid - discharge
        self.soc = (
            self.soc
            + (charge_grid + charge_pv) * self.dt * cfg.eta_ch / cfg.E_cap
            - discharge * self.dt / (cfg.eta_dis * cfg.E_cap)
        )
        self.soc = min(cfg.SOC_max, max(cfg.SOC_min, self.soc))

        energy_cost = self.tariff[t] * grid * self.dt
        self._block_grid_sum += max(0.0, grid)
        self._block_grid_sum_nb += max(0.0, effective_load)
        block_closed = (t + 1) % self.block_slots == 0
        peak_pen = 0.0
        peak_pen_nb = 0.0
        demand_kw: float | None = None
        demand_nb_kw: float | None = None
        if block_closed:
            demand_kw = self._block_grid_sum / self.block_slots
            demand_nb_kw = self._block_grid_sum_nb / self.block_slots
            peak_pen = cfg.T_cap * max(0.0, demand_kw - self.d_run_shaping)
            self.d_run_shaping = max(self.d_run_shaping, demand_kw)
            self.d_run = max(self.d_run, demand_kw)
            peak_pen_nb = cfg.T_cap * max(0.0, demand_nb_kw - self.d_run_shaping_nb)
            self.d_run_shaping_nb = max(self.d_run_shaping_nb, demand_nb_kw)
            self.d_run_nb = max(self.d_run_nb, demand_nb_kw)
            self.prev_demand = demand_kw
            self._block_grid_sum = 0.0
            self._block_grid_sum_nb = 0.0

        deg_cost = self.deg * discharge * self.dt
        self.total_degradation_cost += deg_cost
        energy_delta = self.tariff[t] * (charge_grid - discharge) * self.dt
        peak_delta = peak_pen - peak_pen_nb
        terminal_cost = 0.0
        is_terminal_step = self.day == len(self.month.days) - 1 and t == self.n_steps - 1
        if is_terminal_step:
            start_energy = self.soc_start * cfg.E_cap
            end_energy = self.soc * cfg.E_cap
            terminal_cost = cfg.price_off * (
                max(0.0, start_energy - end_energy) / cfg.eta_ch
                - cfg.eta_dis * max(0.0, end_energy - start_energy)
            )
            self.total_terminal_cost = terminal_cost
        clip_cost = self.clip_penalty_per_kwh * abs(p_requested - p_executed) * self.dt
        reward = -(
            energy_delta + peak_delta + deg_cost + terminal_cost + clip_cost
        ) / self.reward_scale_vnd

        if is_peak_t:
            self.peak_actual_grid_charge_kwh += charge_grid * self.dt

        self.log_grid[self.day][t] = grid
        self.log_pbess[self.day][t] = p_executed
        self.log_soc[self.day][t + 1] = self.soc
        self.prev_load = load
        self.prev_pv = pv
        self.prev_p_bess = p_executed
        self.net_load = self.net_load.update(effective_load)
        self.steps_in_month += 1

        self.t += 1
        done = False
        if self.t >= self.n_steps:
            self.t = 0
            self.day += 1
            self.prev_demand = 0.0
            self._block_grid_sum = 0.0
            self._block_grid_sum_nb = 0.0
            if self.day >= len(self.month.days):
                done = True
            else:
                self.log_soc[self.day][0] = self.soc
                self._set_day_tariff()

        obs = None if done else self._obs()
        info = {
            "grid_kw": grid,
            "energy_cost": energy_cost,
            "peak_pen": peak_pen,
            "peak_pen_nb": peak_pen_nb,
            "demand_kw": demand_kw,
            "demand_nb_kw": demand_nb_kw,
            "demand_block_closed": block_closed,
            "d_run": self.d_run,
            "d_run_shaping": self.d_run_shaping,
            "rew_energy_delta": energy_delta,
            "rew_peak_delta": peak_delta,
            "rew_deg_cost": deg_cost,
            "rew_terminal_cost": terminal_cost,
            "rew_clip_cost": clip_cost,
            "action_held": held,
            "p_requested_kw": p_requested,
            "p_executed_kw": p_executed,
            "clip_reason": mapped.clip_reason,
            "block_phase": t % self.block_slots,
            "block_grid_sum_kw": self._block_grid_sum,
            "charge_grid_kw": charge_grid,
            "charge_pv_kw": charge_pv,
            "is_peak": is_peak_t,
            "rew_total": reward,
        }
        return obs, reward, done, info
