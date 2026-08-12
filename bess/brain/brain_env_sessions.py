"""Flask-facing sessions for the canonical human BrainEnv playground."""
from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from bess.brain.runtime import load_csv_days
from bess.evaluation.benchmark import detect_dt_hours, selected_data_filename, selected_data_path
from bess.core.timebase import steps_per_day_from_dt
from bess.brain.brain_env import BrainEnv, ElectricityMeterState


class BrainEnvSessionError(ValueError):
    """A user-facing playground request cannot be completed."""


class BrainEnvSessionNotFound(LookupError):
    """The requested in-memory session does not exist."""


class BrainEnvSessionComplete(RuntimeError):
    """The selected day has no remaining timesteps."""


def _finite_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BrainEnvSessionError(f"{label} must be a number.") from exc
    if not math.isfinite(number):
        raise BrainEnvSessionError(f"{label} must be finite.")
    if minimum is not None and number < minimum:
        raise BrainEnvSessionError(f"{label} must be at least {minimum:g}.")
    return number


def _clock_minutes(value: str) -> int:
    pieces = value.strip().split(":", 1)
    if len(pieces) != 2:
        raise BrainEnvSessionError(f"Invalid tariff time: {value!r}.")
    try:
        hour, minute = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise BrainEnvSessionError(f"Invalid tariff time: {value!r}.") from exc
    if hour < 0 or hour > 24 or minute < 0 or minute > 59 or (hour == 24 and minute != 0):
        raise BrainEnvSessionError(f"Invalid tariff time: {value!r}.")
    return hour * 60 + minute


def _tariff_windows(raw: Any, label: str) -> tuple[tuple[int, int], ...]:
    windows: list[tuple[int, int]] = []
    for raw_window in str(raw or "").split(","):
        candidate = raw_window.strip()
        if not candidate:
            continue
        pieces = candidate.split("-", 1)
        if len(pieces) != 2:
            raise BrainEnvSessionError(f"{label} contains an invalid window: {candidate!r}.")
        start = _clock_minutes(pieces[0])
        end = _clock_minutes(pieces[1])
        if start == end:
            raise BrainEnvSessionError(f"{label} window cannot start and end together: {candidate!r}.")
        windows.append((start, end))
    return tuple(windows)


def _inside_window(minute: float, windows: tuple[tuple[int, int], ...]) -> bool:
    for start, end in windows:
        if start < end and start <= minute < end:
            return True
        if start > end and (minute >= start or minute < end):
            return True
    return False


def _day_is_sunday(date_iso: str | None) -> bool:
    if not date_iso:
        return False
    try:
        return date.fromisoformat(str(date_iso)).weekday() == 6
    except ValueError:
        return False


def _tariffs_for_day(
    *,
    step_count: int,
    timestep_hours: float,
    date_iso: str | None,
    parameters: dict[str, Any],
) -> tuple[float, ...]:
    cheap_price = _finite_float(parameters.get("billing_cheap"), "Cheap tariff", minimum=0.0)
    normal_price = _finite_float(parameters.get("billing_normal"), "Normal tariff", minimum=0.0)
    expensive_price = _finite_float(
        parameters.get("billing_expensive"), "Expensive tariff", minimum=0.0
    )
    cheap_windows = _tariff_windows(parameters.get("billing_windows_cheap"), "Cheap tariff")
    expensive_windows = _tariff_windows(
        parameters.get("billing_windows_expensive"), "Expensive tariff"
    )
    sunday_has_no_peak = bool(parameters.get("billing_sunday")) and _day_is_sunday(date_iso)

    prices: list[float] = []
    for step_index in range(step_count):
        minute = step_index * timestep_hours * 60.0
        if _inside_window(minute, cheap_windows):
            prices.append(cheap_price)
        elif not sunday_has_no_peak and _inside_window(minute, expensive_windows):
            prices.append(expensive_price)
        else:
            prices.append(normal_price)
    return tuple(prices)


def _clock_label(step_index: int, timestep_hours: float) -> str:
    total_minutes = int(round(step_index * timestep_hours * 60.0)) % (24 * 60)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


@dataclass(slots=True)
class BrainPlaygroundSession:
    session_id: str
    dataset_name: str
    day_index: int
    day_type: str
    date_iso: str | None
    timestep_hours: float
    load_kw: tuple[float, ...]
    pv_kw: tuple[float, ...]
    net_load_kw: tuple[float, ...]
    tariffs_vnd_per_kwh: tuple[float, ...]
    starting_peak_kw: float
    settings: dict[str, Any]
    env: BrainEnv
    trace: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def step_index(self) -> int:
        return self.env.bess_world.timestep_index

    @property
    def total_steps(self) -> int:
        return len(self.net_load_kw)

    @property
    def complete(self) -> bool:
        return self.step_index >= self.total_steps

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state_of_charge": self.env.bess_world.state_of_charge,
                "bess_monthly_peak_kw": self.env.bess_world.meter_state.monthly_peak_kw,
                "raw_monthly_peak_kw": self.env.raw_world.meter_state.monthly_peak_kw,
                "bess_energy_cost_vnd": self.env.bess_world.total_electricity_energy_cost_vnd,
                "bess_demand_cost_vnd": self.env.bess_world.total_demand_cost_vnd,
                "bess_wear_cost_vnd": self.env.bess_world.total_battery_wear_cost_vnd,
                "bess_operating_cost_vnd": self.env.bess_world.total_operating_cost_vnd,
                "raw_energy_cost_vnd": self.env.raw_world.total_electricity_energy_cost_vnd,
                "raw_demand_cost_vnd": self.env.raw_world.total_demand_cost_vnd,
                "raw_operating_cost_vnd": self.env.raw_world.total_operating_cost_vnd,
                "energy_savings_vnd": self.env.electricity_energy_savings_vnd,
                "demand_savings_vnd": self.env.demand_savings_vnd,
                "battery_wear_cost_vnd": self.env.battery_wear_cost_vnd,
                "net_battery_savings_vnd": self.env.net_battery_savings_vnd,
            }

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.session_id,
                "dataset_name": self.dataset_name,
                "day_index": self.day_index,
                "day_type": self.day_type,
                "date_iso": self.date_iso,
                "timestep_hours": self.timestep_hours,
                "step_index": self.step_index,
                "total_steps": self.total_steps,
                "complete": self.complete,
                "starting_peak_kw": self.starting_peak_kw,
                "settings": dict(self.settings),
                "summary": self.summary(),
            }

    def detail(self) -> dict[str, Any]:
        with self.lock:
            return {"status": self.status(), "trace": list(self.trace)}

    def step(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        action_value = _finite_float(action, "Action")
        if action_value < -1.0 or action_value > 1.0:
            raise BrainEnvSessionError("Action must be between -1 and 1.")

        with self.lock:
            if self.complete:
                raise BrainEnvSessionComplete("The selected BrainEnv day is already complete.")

            index = self.step_index
            result = self.env.step(
                action=action_value,
                net_load_kw=self.net_load_kw[index],
                tariff_vnd_per_kwh=self.tariffs_vnd_per_kwh[index],
            )
            physics = result.bess.physics
            bess_meter = result.bess.meter
            raw_meter = result.raw.meter
            entry = {
                "step_index": index,
                "time": _clock_label(index, self.timestep_hours),
                "load_kw": self.load_kw[index],
                "pv_kw": self.pv_kw[index],
                "net_load_kw": self.net_load_kw[index],
                "tariff_vnd_per_kwh": self.tariffs_vnd_per_kwh[index],
                "action": action_value,
                "requested_battery_kw": physics.requested_battery_kw,
                "battery_after_police_kw": physics.battery_after_police_kw,
                "final_battery_kw": physics.final_battery_kw,
                "battery_to_factory_kw": physics.battery_to_factory_kw,
                "grid_to_battery_kw": physics.grid_to_battery_kw,
                "conversion_loss_kw": physics.conversion_loss_kw,
                "battery_throughput_kwh": physics.battery_throughput_kwh,
                "starting_soc": physics.starting_soc,
                "next_soc": physics.next_soc,
                "bess_grid_import_kw": physics.grid_import_kw,
                "raw_grid_import_kw": result.raw.grid_import_kw,
                "bess_block_completed": bess_meter.block_completed,
                "bess_completed_block_demand_kw": bess_meter.completed_block_demand_kw,
                "bess_monthly_peak_kw": bess_meter.monthly_peak_kw,
                "bess_new_monthly_peak": bess_meter.new_monthly_peak,
                "raw_block_completed": raw_meter.block_completed,
                "raw_completed_block_demand_kw": raw_meter.completed_block_demand_kw,
                "raw_monthly_peak_kw": raw_meter.monthly_peak_kw,
                "raw_new_monthly_peak": raw_meter.new_monthly_peak,
                "bess_energy_cost_vnd": result.bess.cost.electricity_energy_cost_vnd,
                "bess_demand_cost_vnd": result.bess.cost.demand_cost_vnd,
                "bess_wear_cost_vnd": result.bess.cost.battery_wear_cost_vnd,
                "bess_operating_cost_vnd": result.bess.cost.operating_cost_vnd,
                "raw_energy_cost_vnd": result.raw.cost.electricity_energy_cost_vnd,
                "raw_demand_cost_vnd": result.raw.cost.demand_cost_vnd,
                "raw_operating_cost_vnd": result.raw.cost.operating_cost_vnd,
                "energy_savings_vnd": result.electricity_energy_savings_vnd,
                "demand_savings_vnd": result.demand_savings_vnd,
                "battery_wear_cost_vnd": result.battery_wear_cost_vnd,
                "net_battery_savings_vnd": result.net_battery_savings_vnd,
                "cumulative_energy_savings_vnd": result.cumulative_electricity_energy_savings_vnd,
                "cumulative_demand_savings_vnd": result.cumulative_demand_savings_vnd,
                "cumulative_battery_wear_cost_vnd": result.cumulative_battery_wear_cost_vnd,
                "cumulative_net_battery_savings_vnd": result.cumulative_net_battery_savings_vnd,
            }
            self.trace.append(entry)
            return entry, self.status()


_SESSIONS: dict[str, BrainPlaygroundSession] = {}
_SESSIONS_LOCK = threading.RLock()


def context(parameters: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(parameters)
    path = selected_data_path(snapshot)
    timestep_hours = detect_dt_hours(path)
    source_days = load_csv_days(path)
    days = [
        {
            "day_index": day.day_index,
            "day_type": day.day_type,
            "date_iso": day.date_iso,
            "step_count": min(len(day.load_kw), len(day.pv_kw)),
        }
        for day in source_days
    ]
    return {
        "dataset_name": selected_data_filename(snapshot),
        "timestep_hours": timestep_hours,
        "days": days,
        "settings": _settings_snapshot(snapshot, timestep_hours),
    }


def _settings_snapshot(parameters: dict[str, Any], timestep_hours: float) -> dict[str, Any]:
    demand_charge = (
        _finite_float(parameters.get("billing_peak_penalty"), "Monthly peak fee", minimum=0.0)
        if parameters.get("billing_mode") == "2tc"
        else 0.0
    )
    return {
        "battery_capacity_kwh": _finite_float(
            parameters.get("battery_capacity_kWh"), "Battery capacity", minimum=0.0
        ),
        "battery_power_kw": _finite_float(
            parameters.get("battery_power_limit_kW"), "Battery power", minimum=0.0
        ),
        "charge_efficiency": _finite_float(parameters.get("charge_efficiency"), "Charge efficiency"),
        "discharge_efficiency": _finite_float(
            parameters.get("discharge_efficiency"), "Discharge efficiency"
        ),
        "minimum_soc": _finite_float(parameters.get("minimum_soc"), "Minimum SOC"),
        "maximum_soc": _finite_float(parameters.get("maximum_soc"), "Maximum SOC"),
        "initial_soc": _finite_float(parameters.get("minimum_soc"), "Minimum SOC"),
        "timestep_hours": timestep_hours,
        "battery_wear_vnd_per_kwh": _finite_float(
            parameters.get("battery_wear_cost"), "Battery wear cost", minimum=0.0
        ),
        "billing_mode": str(parameters.get("billing_mode") or ""),
        "demand_charge_vnd_per_kw": demand_charge,
        "cheap_tariff_vnd_per_kwh": _finite_float(
            parameters.get("billing_cheap"), "Cheap tariff", minimum=0.0
        ),
        "normal_tariff_vnd_per_kwh": _finite_float(
            parameters.get("billing_normal"), "Normal tariff", minimum=0.0
        ),
        "expensive_tariff_vnd_per_kwh": _finite_float(
            parameters.get("billing_expensive"), "Expensive tariff", minimum=0.0
        ),
        "cheap_windows": str(parameters.get("billing_windows_cheap") or ""),
        "expensive_windows": str(parameters.get("billing_windows_expensive") or ""),
        "sunday_no_peak": bool(parameters.get("billing_sunday")),
    }


def create_session(
    parameters: dict[str, Any], *, day_index: Any, starting_peak_kw: Any = 0.0
) -> BrainPlaygroundSession:
    snapshot = dict(parameters)
    try:
        requested_day = int(day_index)
    except (TypeError, ValueError) as exc:
        raise BrainEnvSessionError("Choose a valid CSV day.") from exc
    carry_in_peak = _finite_float(starting_peak_kw, "Carry-in monthly peak", minimum=0.0)

    path = selected_data_path(snapshot)
    timestep_hours = detect_dt_hours(path)
    steps_per_day = steps_per_day_from_dt(timestep_hours)
    source_days = load_csv_days(path)
    day = next((candidate for candidate in source_days if candidate.day_index == requested_day), None)
    if day is None:
        raise BrainEnvSessionError("The selected CSV day does not exist.")
    if len(day.load_kw) != len(day.pv_kw) or len(day.load_kw) != steps_per_day:
        raise BrainEnvSessionError(
            f"Day {requested_day} must contain exactly {steps_per_day} aligned load/PV samples."
        )

    settings = _settings_snapshot(snapshot, timestep_hours)
    env = BrainEnv(
        initial_state_of_charge=settings["initial_soc"],
        minimum_state_of_charge=settings["minimum_soc"],
        maximum_state_of_charge=settings["maximum_soc"],
        battery_capacity_kwh=settings["battery_capacity_kwh"],
        battery_power_kw=settings["battery_power_kw"],
        timestep_hours=timestep_hours,
        charge_efficiency=settings["charge_efficiency"],
        discharge_efficiency=settings["discharge_efficiency"],
        demand_charge_vnd_per_kw=settings["demand_charge_vnd_per_kw"],
        battery_wear_vnd_per_kwh=settings["battery_wear_vnd_per_kwh"],
    )
    starting_meter = ElectricityMeterState(monthly_peak_kw=carry_in_peak)
    env.bess_world.meter_state = starting_meter
    env.raw_world.meter_state = starting_meter

    load_kw = tuple(float(value) for value in day.load_kw)
    pv_kw = tuple(float(value) for value in day.pv_kw)
    net_load_kw = tuple(load - pv for load, pv in zip(load_kw, pv_kw))
    tariffs = _tariffs_for_day(
        step_count=steps_per_day,
        timestep_hours=timestep_hours,
        date_iso=day.date_iso,
        parameters=snapshot,
    )
    session = BrainPlaygroundSession(
        session_id=str(uuid.uuid4()),
        dataset_name=selected_data_filename(snapshot),
        day_index=requested_day,
        day_type=day.day_type,
        date_iso=day.date_iso,
        timestep_hours=timestep_hours,
        load_kw=load_kw,
        pv_kw=pv_kw,
        net_load_kw=net_load_kw,
        tariffs_vnd_per_kwh=tariffs,
        starting_peak_kw=carry_in_peak,
        settings=settings,
        env=env,
    )
    with _SESSIONS_LOCK:
        _SESSIONS[session.session_id] = session
    return session


def get_session(session_id: str) -> BrainPlaygroundSession:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(str(session_id))
    if session is None:
        raise BrainEnvSessionNotFound("BrainEnv session not found.")
    return session


def drop_session(session_id: str) -> bool:
    with _SESSIONS_LOCK:
        return _SESSIONS.pop(str(session_id), None) is not None
