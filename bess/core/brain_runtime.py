"""Caller-side runtime helpers for the canonical :mod:`bess.core.bess_env`.

The environment intentionally owns only one native timestep at a time.  This
module handles concerns that belong outside the environment: converting legacy
``MonthData`` inputs into ``BrainEpisode`` values, holding a policy decision
across an explicit control interval, and recording trajectories for scoring/UI.
It does not emulate the removed ``BESSEnv`` API.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bess.core.bess_env import (
    ACTION_MAX,
    ACTION_MIN,
    DEMAND_BLOCK_HOURS,
    OBSERVATION_DIM,
    BrainEnv,
    BrainEnvironmentStepResult,
    BrainEpisode,
    BrainObservation,
    BrainTimestepInput,
)
from bess.core.common import tariff_vector_day, validate_control_interval_minutes
from bess.core.scenario_gen import DayData, MonthData
from bess.core.timebase import (
    demand_window_steps,
    dt_from_steps_per_day,
    fixed_demand_block_averages,
    steps_per_day_from_dt,
)

REWARD_SCALE_VND = 1_000_000.0


def native_steps_per_action(
    native_timestep_hours: float,
    control_interval_minutes: float | None,
) -> int:
    """Return the exact number of native environment steps per policy action."""
    native_minutes = float(native_timestep_hours) * 60.0
    control_minutes = (
        native_minutes
        if control_interval_minutes is None
        else float(control_interval_minutes)
    )
    validated = validate_control_interval_minutes(native_minutes, control_minutes)
    return round(validated / native_minutes)


def build_brain_episode(
    month: MonthData,
    cfg,
    *,
    power_scale_kw: float,
) -> BrainEpisode:
    """Convert measured load/PV days into the new env's net-load episode input."""
    if not month.days:
        raise ValueError("BrainEnv episode requires at least one day")

    steps_per_day = len(month.days[0].load)
    if steps_per_day <= 0:
        raise ValueError("BrainEnv episode contains an empty day")

    expected_dt = dt_from_steps_per_day(steps_per_day)
    if not math.isclose(float(cfg.dt), expected_dt, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"config timestep {float(cfg.dt) * 60:g} minutes does not match "
            f"episode resolution {expected_dt * 60:g} minutes"
        )

    timesteps: list[BrainTimestepInput] = []
    for day in month.days:
        load = np.asarray(day.load, dtype=np.float64)
        pv = np.asarray(day.pv, dtype=np.float64)
        if len(load) != steps_per_day or len(pv) != steps_per_day:
            raise ValueError("all BrainEnv days must have the same number of samples")
        if not np.isfinite(load).all() or not np.isfinite(pv).all():
            raise ValueError("BrainEnv load/PV inputs must be finite")

        tariff = np.asarray(tariff_vector_day(cfg, day), dtype=np.float64)
        if len(tariff) != steps_per_day:
            raise ValueError("tariff resolution does not match BrainEnv episode data")

        net_load = np.maximum(0.0, load - pv)
        working_day = day.day_type == "working"
        timesteps.extend(
            BrainTimestepInput(
                net_load_kw=float(net_load[index]),
                tariff_vnd_per_kwh=float(tariff[index]),
                is_working_day=working_day,
            )
            for index in range(steps_per_day)
        )

    return BrainEpisode(
        timesteps=tuple(timesteps),
        steps_per_day=steps_per_day,
        power_scale_kw=float(power_scale_kw),
    )


def make_brain_env(
    month: MonthData,
    cfg,
    *,
    power_scale_kw: float,
    battery_wear_vnd_per_kwh: float = 0.0,
    initial_state_of_charge: float | None = None,
) -> BrainEnv:
    """Construct canonical ``BrainEnv``; episodes start at SOC_min by default."""
    episode = build_brain_episode(month, cfg, power_scale_kw=power_scale_kw)
    initial_soc = (
        float(cfg.SOC_min)
        if initial_state_of_charge is None
        else float(initial_state_of_charge)
    )
    return BrainEnv(
        initial_state_of_charge=initial_soc,
        minimum_state_of_charge=float(cfg.SOC_min),
        maximum_state_of_charge=float(cfg.SOC_max),
        battery_capacity_kwh=float(cfg.E_cap),
        battery_power_kw=float(cfg.P_rated_nominal),
        timestep_hours=float(cfg.dt),
        charge_efficiency=float(cfg.eta_ch),
        discharge_efficiency=float(cfg.eta_dis),
        demand_charge_vnd_per_kw=float(cfg.T_cap),
        battery_wear_vnd_per_kwh=float(battery_wear_vnd_per_kwh),
        episode=episode,
    )


@dataclass(slots=True)
class BrainTrajectoryRecorder:
    """Record native typed env results into the repo's scoring array format."""

    month: MonthData
    initial_state_of_charge: float
    grid_import_days: list[np.ndarray] = field(init=False)
    state_of_charge_days: list[np.ndarray] = field(init=False)
    battery_power_days: list[np.ndarray] = field(init=False)
    _day_index: int = field(default=0, init=False)
    _step_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.grid_import_days = [
            np.zeros(len(day.load), dtype=np.float64) for day in self.month.days
        ]
        self.battery_power_days = [
            np.zeros(len(day.load), dtype=np.float64) for day in self.month.days
        ]
        self.state_of_charge_days = [
            np.zeros(len(day.load) + 1, dtype=np.float64) for day in self.month.days
        ]
        if self.state_of_charge_days:
            self.state_of_charge_days[0][0] = float(self.initial_state_of_charge)

    def record(self, result: BrainEnvironmentStepResult) -> None:
        if self._day_index >= len(self.month.days):
            raise RuntimeError("trajectory recorder received more rows than the episode contains")
        expected = len(self.month.days[self._day_index].load)
        if self._step_index >= expected:
            raise RuntimeError("trajectory recorder day index is out of sync")

        physics = result.bess.physics
        self.grid_import_days[self._day_index][self._step_index] = physics.grid_import_kw
        self.battery_power_days[self._day_index][self._step_index] = physics.final_battery_kw
        self.state_of_charge_days[self._day_index][self._step_index + 1] = physics.next_soc

        self._step_index += 1
        if self._step_index == expected:
            ending_soc = physics.next_soc
            self._day_index += 1
            self._step_index = 0
            if self._day_index < len(self.state_of_charge_days):
                self.state_of_charge_days[self._day_index][0] = ending_soc


@dataclass(frozen=True, slots=True)
class BrainControlTransition:
    """One policy decision after holding its action over native env timesteps."""

    next_observation: BrainObservation | None
    reward_vnd: float
    reward_million_vnd: float
    done: bool
    native_results: tuple[BrainEnvironmentStepResult, ...]
    adjusted_action: bool
    tariff_blocked_charge_steps: int
    peak_guard_trigger_steps: int
    peak_guard_override_steps: int
    peak_guard_unmet_steps: int
    soc_deadline_trigger_steps: int
    soc_deadline_override_steps: int
    soc_deadline_unmet_count: int
    soc_deadline_shortfall_penalty_vnd: float
    requested_policy_action: float
    applied_native_actions: tuple[float, ...]
    peak_guard_planned_targets_kw: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class PeakGuardDecision:
    """One native-step minimum-action clamp protecting the seen demand peak."""

    action: float
    triggered: bool
    adjusted: bool
    allowed_grid_kw: float | None


def elastic_peak_target_kw(
    base_target_kw: float,
    *,
    state_of_charge: float,
    minimum_state_of_charge: float,
    maximum_state_of_charge: float,
    battery_capacity_kwh: float,
    battery_power_kw: float,
    net_load_kw: float,
    block_energy_kwh: float,
    block_elapsed_hours: float,
    native_step_in_day: int,
    steps_per_day: int,
    timestep_hours: float,
    discharge_efficiency: float,
    reserve_fraction: float,
    reserve_release_start_hour: float,
    reserve_release_end_hour: float,
    enabled: bool,
) -> float:
    """Raise an optimistic peak bid only enough to protect a causal SOC reserve.

    The base bid comes only from completed historical days. This helper never
    looks ahead. It computes the smallest current fixed-meter cap that Peak
    Police can enforce without *forcing* SOC below a time-varying reserve.
    PPO remains free to discharge more strongly on its own.
    """
    values = (
        base_target_kw,
        state_of_charge,
        minimum_state_of_charge,
        maximum_state_of_charge,
        battery_capacity_kwh,
        battery_power_kw,
        net_load_kw,
        block_energy_kwh,
        block_elapsed_hours,
        timestep_hours,
        discharge_efficiency,
        reserve_fraction,
        reserve_release_start_hour,
        reserve_release_end_hour,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Elastic Peak Bid inputs must all be finite")
    if not isinstance(native_step_in_day, int) or not isinstance(steps_per_day, int):
        raise TypeError("Elastic Peak Bid step indexes must be integers")

    base_target = float(base_target_kw)
    soc = float(state_of_charge)
    soc_min = float(minimum_state_of_charge)
    soc_max = float(maximum_state_of_charge)
    capacity = float(battery_capacity_kwh)
    rated_power = float(battery_power_kw)
    net_load = max(0.0, float(net_load_kw))
    open_energy = float(block_energy_kwh)
    open_elapsed = float(block_elapsed_hours)
    dt_hours = float(timestep_hours)
    eta_dis = float(discharge_efficiency)
    reserve = float(reserve_fraction)
    release_start = float(reserve_release_start_hour)
    release_end = float(reserve_release_end_hour)

    if base_target < 0.0:
        raise ValueError("Elastic Peak Bid base target must be non-negative")
    if capacity <= 0.0 or rated_power <= 0.0 or dt_hours <= 0.0:
        raise ValueError("Elastic Peak Bid battery and timestep values must be positive")
    if not 0.0 <= soc_min < soc_max <= 1.0:
        raise ValueError("Elastic Peak Bid SOC limits must satisfy 0 <= min < max <= 1")
    if not soc_min - 1e-9 <= soc <= soc_max + 1e-9:
        raise ValueError("Elastic Peak Bid SOC must be inside configured limits")
    if not 0.0 < eta_dis <= 1.0:
        raise ValueError("Elastic Peak Bid discharge efficiency must be inside (0, 1]")
    if not 0.0 <= reserve < 1.0:
        raise ValueError("Elastic Peak Bid reserve fraction must be inside [0, 1)")
    if not 0.0 <= release_start < release_end <= 24.0:
        raise ValueError("Elastic Peak Bid reserve release hours must satisfy 0 <= start < end <= 24")
    if steps_per_day <= 0 or not 0 <= native_step_in_day < steps_per_day:
        raise ValueError("Elastic Peak Bid native step must be inside the day")
    if open_energy < 0.0 or open_elapsed < 0.0 or open_elapsed >= DEMAND_BLOCK_HOURS:
        raise ValueError("Elastic Peak Bid meter state is invalid")
    if open_elapsed + dt_hours > DEMAND_BLOCK_HOURS + 1e-12:
        raise ValueError("Elastic Peak Bid timestep would overrun the fixed meter block")
    if not enabled:
        return base_target

    hour = native_step_in_day * dt_hours
    if hour < release_start:
        live_reserve_fraction = reserve
    elif hour >= release_end:
        live_reserve_fraction = 0.0
    else:
        live_reserve_fraction = reserve * (release_end - hour) / (release_end - release_start)

    reserve_soc = soc_min + live_reserve_fraction * (soc_max - soc_min)
    energy_above_reserve_kwh = max(0.0, soc - reserve_soc) * capacity
    battery_discharge_budget_kw = min(rated_power, energy_above_reserve_kwh / dt_hours)
    outside_discharge_budget_kw = battery_discharge_budget_kw * eta_dis

    remaining_time_hours = DEMAND_BLOCK_HOURS - open_elapsed
    minimum_grid_kw = max(0.0, net_load - outside_discharge_budget_kw)
    feasibility_floor_kw = (
        open_energy + minimum_grid_kw * remaining_time_hours
    ) / DEMAND_BLOCK_HOURS
    # TODO(IQ-75): this floor may relax only from current SOC/meter facts. Never
    # feed future current-bucket load, Oracle peak, or validation/test foresight here.
    return max(base_target, feasibility_floor_kw)


def _minimum_daily_peak_target_kw(
    meter_blocks_kw: list[float],
    *,
    maximum_grid_reduction_kw: float,
    output_energy_budget_kwh: float,
) -> float:
    """Return the lowest cap that one full battery could defend for one day."""
    if not meter_blocks_kw:
        return 0.0

    ordered = sorted((max(0.0, float(value)) for value in meter_blocks_kw), reverse=True)
    power_floor = max(0.0, ordered[0] - float(maximum_grid_reduction_kw))
    energy_budget_kw_blocks = float(output_energy_budget_kwh) / DEMAND_BLOCK_HOURS
    prefix = 0.0
    energy_floor = 0.0
    for count, value in enumerate(ordered, start=1):
        prefix += value
        next_value = ordered[count] if count < len(ordered) else 0.0
        candidate = (prefix - energy_budget_kw_blocks) / count
        if candidate >= next_value:
            energy_floor = max(0.0, candidate)
            break
    return max(power_floor, energy_floor)


def causal_peak_target_from_history(
    history_days: list[DayData] | tuple[DayData, ...],
    cfg,
    *,
    lookback_days: int,
    day_quantile: float,
    energy_reserve_fraction: float,
    guard_start_hour: float,
    fallback_kw: float | None,
) -> float:
    """Estimate a causal peak cap from completed days and battery feasibility.

    The estimate never reads the episode it will control.  Each historical day
    is reduced to the lowest fixed-meter cap defendable by battery power and by
    the reserved usable energy.  A nearest-rank day quantile then provides one
    stable month-start target; new checkpoints use the worst historical day.
    """
    if isinstance(lookback_days, bool) or int(lookback_days) != lookback_days:
        raise ValueError("causal peak target lookback must be a positive integer")
    lookback = int(lookback_days)
    if lookback <= 0:
        raise ValueError("causal peak target lookback must be a positive integer")

    quantile = float(day_quantile)
    reserve = float(energy_reserve_fraction)
    start_hour = float(guard_start_hour)
    if not math.isfinite(quantile) or not 0.0 < quantile <= 1.0:
        raise ValueError("causal peak target day quantile must be inside (0, 1]")
    if not math.isfinite(reserve) or not 0.0 <= reserve < 1.0:
        raise ValueError("causal peak target energy reserve must be inside [0, 1)")
    if not math.isfinite(start_hour) or not 0.0 <= start_hour < 24.0:
        raise ValueError("causal peak target guard start hour must be inside [0, 24)")

    fallback = None if fallback_kw is None else float(fallback_kw)
    if fallback is not None and (not math.isfinite(fallback) or fallback < 0.0):
        raise ValueError("causal peak target fallback must be finite and non-negative")

    selected_days = list(history_days)[-lookback:]
    if not selected_days:
        if fallback is None:
            raise ValueError("causal peak target requires completed history or a fallback")
        return fallback

    dt_hours = float(cfg.dt)
    steps_per_block = demand_window_steps(dt_hours)
    exact_start_step = start_hour / dt_hours
    start_step = round(exact_start_step)
    if abs(exact_start_step - start_step) > 1e-9:
        raise ValueError("causal peak target guard start hour must align with the native timestep")
    if start_step % steps_per_block:
        raise ValueError("causal peak target guard start hour must align with the fixed meter")

    maximum_grid_reduction_kw = float(cfg.P_rated_nominal) * float(cfg.eta_dis)
    output_energy_budget_kwh = (
        max(0.0, float(cfg.SOC_max) - float(cfg.SOC_min))
        * float(cfg.E_cap)
        * float(cfg.eta_dis)
        * (1.0 - reserve)
    )
    if maximum_grid_reduction_kw <= 0.0 or output_energy_budget_kwh < 0.0:
        raise ValueError("causal peak target requires positive battery power and valid energy")

    first_block = start_step // steps_per_block
    expected_steps_per_day = steps_per_day_from_dt(dt_hours)
    daily_targets: list[float] = []
    for day in selected_days:
        load = np.asarray(day.load, dtype=np.float64)
        pv = np.asarray(day.pv, dtype=np.float64)
        if load.shape != pv.shape or load.ndim != 1 or load.size == 0:
            raise ValueError("causal peak target history days require matching non-empty load/PV")
        if load.size != expected_steps_per_day:
            raise ValueError("causal peak target history resolution does not match config")
        if not np.isfinite(load).all() or not np.isfinite(pv).all():
            raise ValueError("causal peak target history must be finite")
        net_load = np.maximum(0.0, load - pv)
        meter_blocks = fixed_demand_block_averages(net_load, dt_hours)[first_block:]
        daily_targets.append(_minimum_daily_peak_target_kw(
            meter_blocks,
            maximum_grid_reduction_kw=maximum_grid_reduction_kw,
            output_energy_budget_kwh=output_energy_budget_kwh,
        ))

    # Zero-target days (for example very low-load Sundays) must not drag an
    # optimistic working-day bid toward zero. The old IQ-72 worst-day q=1.0
    # result is unchanged because removing zeros cannot change the maximum.
    positive_targets = [value for value in daily_targets if value > 1e-9]
    ordered_targets = sorted(positive_targets or daily_targets)
    rank = max(0, math.ceil(quantile * len(ordered_targets)) - 1)
    # TODO(IQ-75): keep this causal lower-quartile base bid only with the elastic
    # SOC feasibility floor; never turn it into current-bucket look-ahead.
    return float(ordered_targets[rank])


@dataclass(frozen=True, slots=True)
class SocDeadlineDecision:
    """One native-step exact charging schedule that fills SOC by a clock deadline."""

    action: float
    triggered: bool
    adjusted: bool
    required_charge_kw: float
    physically_feasible: bool


def enforce_soc_deadline_guard(
    action: float,
    *,
    state_of_charge: float,
    maximum_state_of_charge: float,
    battery_capacity_kwh: float,
    battery_power_kw: float,
    native_step_in_day: int,
    steps_per_day: int,
    timestep_hours: float,
    deadline_hour: float,
    enabled: bool,
) -> SocDeadlineDecision:
    """Spread the required charging evenly from midnight to a daily deadline.

    Positive action is discharge and negative action is charge.  Recomputing the
    required average every native step replaces both weaker and stronger policy
    requests, so the battery follows one smooth line to the target without an
    early charging spike.
    """
    values = (
        action,
        state_of_charge,
        maximum_state_of_charge,
        battery_capacity_kwh,
        battery_power_kw,
        timestep_hours,
        deadline_hour,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("SOC Deadline Guard inputs must all be finite")
    if not isinstance(native_step_in_day, int) or not isinstance(steps_per_day, int):
        raise TypeError("SOC Deadline Guard step indexes must be integers")
    if steps_per_day <= 0 or not 0 <= native_step_in_day < steps_per_day:
        raise ValueError("SOC Deadline Guard native step must be inside the day")

    action_value = float(action)
    soc = float(state_of_charge)
    soc_max = float(maximum_state_of_charge)
    capacity = float(battery_capacity_kwh)
    rated_power = float(battery_power_kw)
    dt_hours = float(timestep_hours)
    deadline = float(deadline_hour)
    if not ACTION_MIN <= action_value <= ACTION_MAX:
        raise ValueError("SOC Deadline Guard action must be inside [-1, 1]")
    if capacity <= 0.0 or rated_power <= 0.0 or dt_hours <= 0.0:
        raise ValueError("SOC Deadline Guard battery and timestep values must be positive")
    if not 0.0 < deadline < 24.0:
        raise ValueError("SOC Deadline Guard hour must be inside (0, 24)")
    if soc > soc_max + 1e-9:
        raise ValueError("SOC Deadline Guard state of charge exceeds its maximum")

    exact_deadline_step = deadline / dt_hours
    deadline_step = round(exact_deadline_step)
    if abs(exact_deadline_step - deadline_step) > 1e-9:
        raise ValueError("SOC Deadline Guard hour must align with the native timestep")
    if deadline_step <= 0 or deadline_step >= steps_per_day:
        raise ValueError("SOC Deadline Guard deadline step must be inside the day")
    if not enabled or native_step_in_day >= deadline_step:
        return SocDeadlineDecision(action_value, False, False, 0.0, True)

    missing_energy_kwh = max(0.0, soc_max - soc) * capacity
    if missing_energy_kwh <= 1e-9:
        guarded_action = 0.0
        adjusted = not math.isclose(
            guarded_action,
            action_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        return SocDeadlineDecision(
            guarded_action,
            adjusted,
            adjusted,
            0.0,
            True,
        )
    remaining_hours = (deadline_step - native_step_in_day) * dt_hours
    required_charge_kw = missing_energy_kwh / remaining_hours
    required_action = -min(rated_power, required_charge_kw) / rated_power
    guarded_action = required_action
    adjusted = not math.isclose(guarded_action, action_value, rel_tol=0.0, abs_tol=1e-12)
    # TODO(IQ-68-SMOOTH): keep this O(1) exact schedule authoritative over PPO
    # and Oracle-BC requests; both weaker and stronger requests break smoothness.
    return SocDeadlineDecision(
        guarded_action,
        True,
        adjusted,
        required_charge_kw,
        required_charge_kw <= rated_power + 1e-9,
    )


def enforce_seen_peak_guard(
    action: float,
    *,
    net_load_kw: float,
    monthly_peak_kw: float,
    block_energy_kwh: float,
    block_elapsed_hours: float,
    timestep_hours: float,
    battery_power_kw: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    enabled: bool,
    armed: bool,
    deadband_kw: float,
) -> PeakGuardDecision:
    """Raise a policy action only when projected grid would break its peak cap.

    ``monthly_peak_kw`` may be truthful Eye 6 or a larger causal planned target.
    The fixed meter has an energy budget of ``monthly_peak_kw * 0.5 h`` for
    each block.  The remaining budget is shared evenly across the remaining
    native samples.  This lets the guard reconsider the second 15-minute row
    even when PPO's original action is held for 30 minutes.

    The result is a lower bound, not a replacement policy: a stronger PPO
    discharge remains untouched, and safe below-peak discharge is not blocked.
    """
    values = (
        action,
        net_load_kw,
        monthly_peak_kw,
        block_energy_kwh,
        block_elapsed_hours,
        timestep_hours,
        battery_power_kw,
        charge_efficiency,
        discharge_efficiency,
        deadband_kw,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Peak Guard inputs must all be finite")

    action_value = float(action)
    net_load = max(0.0, float(net_load_kw))
    seen_peak = float(monthly_peak_kw)
    open_energy = float(block_energy_kwh)
    open_elapsed = float(block_elapsed_hours)
    dt_hours = float(timestep_hours)
    rated_power = float(battery_power_kw)
    charge_eta = float(charge_efficiency)
    discharge_eta = float(discharge_efficiency)
    deadband = float(deadband_kw)

    if action_value < ACTION_MIN or action_value > ACTION_MAX:
        raise ValueError("Peak Guard action must be inside [-1, 1]")
    if seen_peak < 0.0 or open_energy < 0.0:
        raise ValueError("Peak Guard meter values must not be negative")
    if dt_hours <= 0.0 or dt_hours > DEMAND_BLOCK_HOURS:
        raise ValueError("Peak Guard timestep must be inside (0, 0.5]")
    if open_elapsed < 0.0 or open_elapsed >= DEMAND_BLOCK_HOURS:
        raise ValueError("Peak Guard open-block elapsed time must be inside [0, 0.5)")
    if open_elapsed + dt_hours > DEMAND_BLOCK_HOURS + 1e-12:
        raise ValueError("Peak Guard timestep would overrun the open meter block")
    if rated_power <= 0.0:
        raise ValueError("Peak Guard battery power must be greater than 0")
    if not 0.0 < charge_eta <= 1.0 or not 0.0 < discharge_eta <= 1.0:
        raise ValueError("Peak Guard efficiencies must be inside (0, 1]")
    if deadband < 0.0:
        raise ValueError("Peak Guard deadband must not be negative")

    if not enabled or not armed or seen_peak <= 0.0:
        return PeakGuardDecision(action_value, False, False, None)

    remaining_time_hours = DEMAND_BLOCK_HOURS - open_elapsed
    remaining_energy_budget_kwh = seen_peak * DEMAND_BLOCK_HOURS - open_energy
    allowed_grid_kw = max(0.0, remaining_energy_budget_kwh / remaining_time_hours)

    requested_battery_kw = action_value * rated_power
    requested_outside_kw = (
        requested_battery_kw * discharge_eta
        if requested_battery_kw >= 0.0
        else requested_battery_kw / charge_eta
    )
    projected_grid_kw = max(0.0, net_load - requested_outside_kw)
    if projected_grid_kw <= allowed_grid_kw + deadband:
        return PeakGuardDecision(action_value, False, False, allowed_grid_kw)

    outside_power_needed_kw = net_load - allowed_grid_kw
    minimum_battery_kw = (
        outside_power_needed_kw / discharge_eta
        if outside_power_needed_kw >= 0.0
        else outside_power_needed_kw * charge_eta
    )
    minimum_action = min(ACTION_MAX, max(ACTION_MIN, minimum_battery_kw / rated_power))
    guarded_action = max(action_value, minimum_action)
    adjusted = not math.isclose(guarded_action, action_value, rel_tol=0.0, abs_tol=1e-12)
    # TODO(IQ-66): keep this hard lower-bound only if the untouched test bucket
    # improves and human review confirms it stops feasible new meter peaks.
    return PeakGuardDecision(guarded_action, True, adjusted, allowed_grid_kw)


def constrain_charge_to_cheap_window(
    action: float,
    *,
    native_step_in_day: int,
    cheap_tariff_steps: set[int] | frozenset[int],
    enabled: bool,
) -> float:
    """Return zero for charging requests outside configured off-peak clock slots."""
    action_value = float(action)
    if not math.isfinite(action_value):
        raise ValueError("charge window constraint action must be finite")
    step = int(native_step_in_day)
    if step < 0:
        raise ValueError("native_step_in_day must not be negative")
    if not enabled or action_value >= 0.0:
        return action_value
    return action_value if step in cheap_tariff_steps else 0.0


def step_brain_control(
    env: BrainEnv,
    action: float,
    *,
    native_steps: int,
    recorder: BrainTrajectoryRecorder | None = None,
    charge_only_during_cheap_tariff: bool = False,
    cheap_tariff_steps: set[int] | frozenset[int] | None = None,
    peak_guard_enabled: bool = False,
    peak_guard_min_completed_days: int = 1,
    peak_guard_first_day_arm_step: int | None = None,
    peak_guard_target_kw: float | None = None,
    peak_guard_deadband_kw: float = 1.0,
    elastic_peak_bid_enabled: bool = False,
    elastic_peak_reserve_fraction: float = 0.20,
    elastic_peak_reserve_release_start_hour: float = 17.5,
    elastic_peak_reserve_release_end_hour: float = 22.5,
    soc_deadline_enabled: bool = False,
    soc_deadline_hour: float = 6.0,
    soc_deadline_shortfall_penalty_vnd: float = 0.0,
) -> BrainControlTransition:
    """Hold one request while native guards derive each physically scheduled action."""
    if native_steps <= 0:
        raise ValueError("native_steps must be greater than 0")
    if charge_only_during_cheap_tariff and cheap_tariff_steps is None:
        raise ValueError("cheap tariff step indexes are required when cheap-only charging is enabled")
    if charge_only_during_cheap_tariff and env.episode is None:
        raise ValueError("cheap-only charging requires an episode-backed BrainEnv")
    if peak_guard_enabled and env.episode is None:
        raise ValueError("Peak Guard requires an episode-backed BrainEnv")
    if soc_deadline_enabled and env.episode is None:
        raise ValueError("SOC Deadline Guard requires an episode-backed BrainEnv")
    if isinstance(peak_guard_min_completed_days, bool) or peak_guard_min_completed_days < 0:
        raise ValueError("Peak Guard minimum completed days must be a non-negative integer")
    if int(peak_guard_min_completed_days) != peak_guard_min_completed_days:
        raise ValueError("Peak Guard minimum completed days must be a non-negative integer")
    if peak_guard_first_day_arm_step is not None:
        if isinstance(peak_guard_first_day_arm_step, bool):
            raise ValueError("Peak Guard first-day arm step must be an integer step index")
        if int(peak_guard_first_day_arm_step) != peak_guard_first_day_arm_step:
            raise ValueError("Peak Guard first-day arm step must be an integer step index")
        assert env.episode is not None
        if not 0 <= int(peak_guard_first_day_arm_step) < env.episode.steps_per_day:
            raise ValueError("Peak Guard first-day arm step must be inside the day")
    if peak_guard_target_kw is not None:
        target_kw = float(peak_guard_target_kw)
        if not math.isfinite(target_kw) or target_kw < 0.0:
            raise ValueError("Peak Guard planned target must be finite and non-negative")

    results: list[BrainEnvironmentStepResult] = []
    reward_vnd = 0.0
    next_observation: BrainObservation | None = None
    done = False
    adjusted = False
    tariff_blocked_charge_steps = 0
    peak_guard_trigger_steps = 0
    peak_guard_override_steps = 0
    peak_guard_unmet_steps = 0
    soc_deadline_trigger_steps = 0
    soc_deadline_override_steps = 0
    soc_deadline_unmet_count = 0
    soc_deadline_shortfall_penalty_vnd = float(soc_deadline_shortfall_penalty_vnd)
    if (
        not math.isfinite(soc_deadline_shortfall_penalty_vnd)
        or soc_deadline_shortfall_penalty_vnd < 0.0
    ):
        raise ValueError("SOC deadline shortfall penalty must be finite and non-negative")
    applied_shortfall_penalty_vnd = 0.0
    applied_native_actions: list[float] = []
    peak_guard_planned_targets_kw: list[float | None] = []

    for _ in range(native_steps):
        step_action = float(action)
        if charge_only_during_cheap_tariff:
            assert env.episode is not None
            timestep_index = env.bess_world.timestep_index
            step_action = constrain_charge_to_cheap_window(
                step_action,
                native_step_in_day=timestep_index % env.episode.steps_per_day,
                cheap_tariff_steps=cheap_tariff_steps,
                enabled=True,
            )
            if not math.isclose(step_action, float(action), rel_tol=0.0, abs_tol=1e-12):
                adjusted = True
                tariff_blocked_charge_steps += 1
        peak_guard = PeakGuardDecision(step_action, False, False, None)
        if peak_guard_enabled:
            assert env.episode is not None
            timestep_index = env.bess_world.timestep_index
            timestep = env.episode.timesteps[timestep_index]
            meter = env.bess_world.meter_state
            completed_days = timestep_index // env.episode.steps_per_day
            native_step_in_day = timestep_index % env.episode.steps_per_day
            armed_after_completed_days = (
                completed_days >= int(peak_guard_min_completed_days)
            )
            armed_on_first_day = (
                completed_days == 0
                and peak_guard_first_day_arm_step is not None
                and native_step_in_day >= int(peak_guard_first_day_arm_step)
            )
            active_planned_target_kw = peak_guard_target_kw
            if active_planned_target_kw is not None and elastic_peak_bid_enabled:
                active_planned_target_kw = elastic_peak_target_kw(
                    float(active_planned_target_kw),
                    state_of_charge=env.bess_world.state_of_charge,
                    minimum_state_of_charge=env.bess_world.minimum_state_of_charge,
                    maximum_state_of_charge=env.bess_world.maximum_state_of_charge,
                    battery_capacity_kwh=env.bess_world.battery_capacity_kwh,
                    battery_power_kw=env.bess_world.battery_power_kw,
                    net_load_kw=timestep.net_load_kw,
                    block_energy_kwh=meter.block_energy_kwh,
                    block_elapsed_hours=meter.block_elapsed_hours,
                    native_step_in_day=native_step_in_day,
                    steps_per_day=env.episode.steps_per_day,
                    timestep_hours=env.bess_world.timestep_hours,
                    discharge_efficiency=env.bess_world.discharge_efficiency,
                    reserve_fraction=elastic_peak_reserve_fraction,
                    reserve_release_start_hour=elastic_peak_reserve_release_start_hour,
                    reserve_release_end_hour=elastic_peak_reserve_release_end_hour,
                    enabled=True,
                )
            peak_guard_planned_targets_kw.append(
                None if active_planned_target_kw is None else float(active_planned_target_kw)
            )
            peak_guard = enforce_seen_peak_guard(
                step_action,
                net_load_kw=timestep.net_load_kw,
                monthly_peak_kw=max(
                    float(meter.monthly_peak_kw),
                    0.0 if active_planned_target_kw is None else float(active_planned_target_kw),
                ),
                block_energy_kwh=meter.block_energy_kwh,
                block_elapsed_hours=meter.block_elapsed_hours,
                timestep_hours=env.bess_world.timestep_hours,
                battery_power_kw=env.bess_world.battery_power_kw,
                charge_efficiency=env.bess_world.charge_efficiency,
                discharge_efficiency=env.bess_world.discharge_efficiency,
                enabled=True,
                armed=armed_after_completed_days or armed_on_first_day,
                deadband_kw=peak_guard_deadband_kw,
            )
            step_action = peak_guard.action
            peak_guard_trigger_steps += int(peak_guard.triggered)
            peak_guard_override_steps += int(peak_guard.adjusted)
            adjusted = adjusted or peak_guard.adjusted
            # TODO(IQ-75): keep truthful Eye 6 separate from the elastic planned
            # cap; old checkpoints without elastic metadata must remain unchanged.
        else:
            peak_guard_planned_targets_kw.append(None)
        deadline_step = None
        native_step_in_day = None
        if soc_deadline_enabled:
            assert env.episode is not None
            deadline_step = round(float(soc_deadline_hour) / env.bess_world.timestep_hours)
            native_step_in_day = env.bess_world.timestep_index % env.episode.steps_per_day
            deadline = enforce_soc_deadline_guard(
                step_action,
                state_of_charge=env.bess_world.state_of_charge,
                maximum_state_of_charge=env.bess_world.maximum_state_of_charge,
                battery_capacity_kwh=env.bess_world.battery_capacity_kwh,
                battery_power_kw=env.bess_world.battery_power_kw,
                native_step_in_day=native_step_in_day,
                steps_per_day=env.episode.steps_per_day,
                timestep_hours=env.bess_world.timestep_hours,
                deadline_hour=soc_deadline_hour,
                enabled=True,
            )
            step_action = deadline.action
            soc_deadline_trigger_steps += int(deadline.triggered)
            soc_deadline_override_steps += int(deadline.adjusted)
            adjusted = adjusted or deadline.adjusted
        applied_native_actions.append(step_action)
        result = env.step(step_action)
        if not isinstance(result, BrainEnvironmentStepResult):
            raise TypeError("owned BrainEnv episode returned a non-episode step result")
        results.append(result)
        reward_vnd += result.reward.timestep_savings_vnd
        if (
            soc_deadline_enabled
            and native_step_in_day is not None
            and deadline_step is not None
            and native_step_in_day + 1 == deadline_step
        ):
            shortfall = max(
                0.0,
                env.bess_world.maximum_state_of_charge - result.bess.physics.next_soc,
            )
            if shortfall > 1e-9:
                soc_deadline_unmet_count += 1
                penalty = soc_deadline_shortfall_penalty_vnd * shortfall * shortfall
                reward_vnd -= penalty
                applied_shortfall_penalty_vnd += penalty
        next_observation = result.next_observation
        done = result.done
        physics = result.bess.physics
        adjusted = adjusted or result.horizon_adjusted or not math.isclose(
            result.projected_battery_kw,
            physics.final_battery_kw,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        if recorder is not None:
            recorder.record(result)
        if (
            peak_guard.allowed_grid_kw is not None
            and result.bess.physics.grid_import_kw
            > peak_guard.allowed_grid_kw + float(peak_guard_deadband_kw) + 1e-9
        ):
            peak_guard_unmet_steps += 1
        if done:
            break

    return BrainControlTransition(
        next_observation=next_observation,
        reward_vnd=reward_vnd,
        reward_million_vnd=reward_vnd / REWARD_SCALE_VND,
        done=done,
        native_results=tuple(results),
        adjusted_action=adjusted,
        tariff_blocked_charge_steps=tariff_blocked_charge_steps,
        peak_guard_trigger_steps=peak_guard_trigger_steps,
        peak_guard_override_steps=peak_guard_override_steps,
        peak_guard_unmet_steps=peak_guard_unmet_steps,
        soc_deadline_trigger_steps=soc_deadline_trigger_steps,
        soc_deadline_override_steps=soc_deadline_override_steps,
        soc_deadline_unmet_count=soc_deadline_unmet_count,
        soc_deadline_shortfall_penalty_vnd=applied_shortfall_penalty_vnd,
        requested_policy_action=float(action),
        applied_native_actions=tuple(applied_native_actions),
        peak_guard_planned_targets_kw=tuple(peak_guard_planned_targets_kw),
    )


def observation_array(observation: BrainObservation) -> np.ndarray:
    """Convert the env's immutable seven-eye tuple to agent-friendly float32."""
    array = np.asarray(observation, dtype=np.float32)
    if array.shape != (OBSERVATION_DIM,):
        raise RuntimeError(f"BrainEnv returned unexpected observation shape {array.shape}")
    return array
