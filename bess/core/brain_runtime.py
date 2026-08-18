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
    OBSERVATION_DIM,
    BrainEnv,
    BrainEnvironmentStepResult,
    BrainEpisode,
    BrainObservation,
    BrainTimestepInput,
)
from bess.core.common import tariff_vector_day, validate_control_interval_minutes
from bess.core.scenario_gen import MonthData
from bess.core.timebase import dt_from_steps_per_day

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


def step_brain_control(
    env: BrainEnv,
    action: float,
    *,
    native_steps: int,
    recorder: BrainTrajectoryRecorder | None = None,
) -> BrainControlTransition:
    """Hold one requested action for ``native_steps`` canonical env transitions."""
    if native_steps <= 0:
        raise ValueError("native_steps must be greater than 0")

    results: list[BrainEnvironmentStepResult] = []
    reward_vnd = 0.0
    next_observation: BrainObservation | None = None
    done = False
    adjusted = False

    for _ in range(native_steps):
        result = env.step(float(action))
        if not isinstance(result, BrainEnvironmentStepResult):
            raise TypeError("owned BrainEnv episode returned a non-episode step result")
        results.append(result)
        reward_vnd += result.reward.timestep_savings_vnd
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
        if done:
            break

    return BrainControlTransition(
        next_observation=next_observation,
        reward_vnd=reward_vnd,
        reward_million_vnd=reward_vnd / REWARD_SCALE_VND,
        done=done,
        native_results=tuple(results),
        adjusted_action=adjusted,
    )


def observation_array(observation: BrainObservation) -> np.ndarray:
    """Convert the env's immutable seven-eye tuple to agent-friendly float32."""
    array = np.asarray(observation, dtype=np.float32)
    if array.shape != (OBSERVATION_DIM,):
        raise RuntimeError(f"BrainEnv returned unexpected observation shape {array.shape}")
    return array
