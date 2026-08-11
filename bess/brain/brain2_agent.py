from __future__ import annotations

import math
from dataclasses import dataclass, field

from bess.brain.brain_env import BrainObservation


@dataclass(frozen=True, slots=True)
class Brain2Decision:
    """One explainable Brain 2 schedule decision."""

    action: float
    label: str
    reason_code: str
    reason: str
    tariff_period: str
    minute_of_day: float
    normalized_soc: float
    requested_battery_power_kw: float
    target_battery_energy_kwh: float
    remaining_cheap_steps: int
    normal_action: float
    expensive_action: float


@dataclass(frozen=True, slots=True)
class Brain2Agent:
    """Deterministic schedule brain with adaptive charging and weighted discharge.

    Brain 2 knows the battery body, native timestep, tariff prices, and tariff windows.
    It does not perform any battery/grid physics itself; it only requests an action.

    Cheap window:
      Recompute the battery-side charge rate every step so the remaining usable room is
      spread evenly over the remaining cheap steps. If physically feasible, the battery
      reaches its configured maximum SOC exactly when the cheap window ends.

    After the cheap window:
      Request one constant action in normal-price time and one larger constant action in
      expensive-price time. The two actions are proportional to their tariff prices and
      are solved once from the configured usable battery energy so an unclipped battery
      that leaves the cheap window full reaches minimum SOC exactly at midnight.
    """

    battery_capacity_kwh: float
    battery_power_kw: float
    minimum_state_of_charge: float
    maximum_state_of_charge: float
    timestep_minutes: float
    cheap_tariff_vnd_per_kwh: float
    normal_tariff_vnd_per_kwh: float
    expensive_tariff_vnd_per_kwh: float
    cheap_start_minute: float
    cheap_end_minute: float
    expensive_start_minute: float
    expensive_end_minute: float

    usable_capacity_kwh: float = field(init=False)
    normal_action: float = field(init=False)
    expensive_action: float = field(init=False)
    steps_per_day: int = field(init=False)
    cheap_steps: int = field(init=False)
    discharge_normal_steps: int = field(init=False)
    discharge_expensive_steps: int = field(init=False)

    def __post_init__(self) -> None:
        numeric_names = (
            "battery_capacity_kwh",
            "battery_power_kw",
            "minimum_state_of_charge",
            "maximum_state_of_charge",
            "timestep_minutes",
            "cheap_tariff_vnd_per_kwh",
            "normal_tariff_vnd_per_kwh",
            "expensive_tariff_vnd_per_kwh",
            "cheap_start_minute",
            "cheap_end_minute",
            "expensive_start_minute",
            "expensive_end_minute",
        )
        values: dict[str, float] = {}
        for name in numeric_names:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"Brain2 {name} must be finite")
            values[name] = value
            object.__setattr__(self, name, value)

        capacity_kwh = values["battery_capacity_kwh"]
        power_kw = values["battery_power_kw"]
        minimum_soc = values["minimum_state_of_charge"]
        maximum_soc = values["maximum_state_of_charge"]
        timestep_minutes = values["timestep_minutes"]
        cheap_tariff = values["cheap_tariff_vnd_per_kwh"]
        normal_tariff = values["normal_tariff_vnd_per_kwh"]
        expensive_tariff = values["expensive_tariff_vnd_per_kwh"]
        cheap_start = values["cheap_start_minute"]
        cheap_end = values["cheap_end_minute"]
        expensive_start = values["expensive_start_minute"]
        expensive_end = values["expensive_end_minute"]

        if capacity_kwh <= 0.0:
            raise ValueError("Brain2 battery_capacity_kwh must be greater than 0")
        if power_kw <= 0.0:
            raise ValueError("Brain2 battery_power_kw must be greater than 0")
        if minimum_soc < 0.0 or maximum_soc > 1.0 or maximum_soc <= minimum_soc:
            raise ValueError(
                "Brain2 SOC limits must satisfy 0 <= minimum_state_of_charge "
                "< maximum_state_of_charge <= 1"
            )
        if timestep_minutes <= 0.0:
            raise ValueError("Brain2 timestep_minutes must be greater than 0")
        if not (0.0 < cheap_tariff < normal_tariff < expensive_tariff):
            raise ValueError(
                "Brain2 tariffs must satisfy 0 < cheap < normal < expensive"
            )

        steps_per_day_float = 1440.0 / timestep_minutes
        steps_per_day = round(steps_per_day_float)
        if steps_per_day <= 0 or not math.isclose(
            steps_per_day_float, steps_per_day, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("Brain2 timestep_minutes must divide a 24-hour day exactly")

        for name, minute in (
            ("cheap_start_minute", cheap_start),
            ("cheap_end_minute", cheap_end),
            ("expensive_start_minute", expensive_start),
            ("expensive_end_minute", expensive_end),
        ):
            if minute < 0.0 or minute > 1440.0:
                raise ValueError(f"Brain2 {name} must be inside [0, 1440]")
            step_float = minute / timestep_minutes
            if not math.isclose(step_float, round(step_float), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"Brain2 {name} must align exactly to the native timestep")

        if not (cheap_start < cheap_end):
            raise ValueError("Brain2 cheap window must have start < end and must not wrap midnight")
        if not (expensive_start < expensive_end):
            raise ValueError(
                "Brain2 expensive window must have start < end and must not wrap midnight"
            )
        if max(cheap_start, expensive_start) < min(cheap_end, expensive_end):
            raise ValueError("Brain2 cheap and expensive tariff windows must not overlap")
        if cheap_end >= 1440.0:
            raise ValueError("Brain2 cheap window must end before midnight to allow discharge")

        cheap_start_step = round(cheap_start / timestep_minutes)
        cheap_end_step = round(cheap_end / timestep_minutes)
        expensive_start_step = round(expensive_start / timestep_minutes)
        expensive_end_step = round(expensive_end / timestep_minutes)
        cheap_steps = cheap_end_step - cheap_start_step
        if cheap_steps <= 0:
            raise ValueError("Brain2 cheap window must contain at least one timestep")

        usable_capacity_kwh = capacity_kwh * (maximum_soc - minimum_soc)
        timestep_hours = timestep_minutes / 60.0
        maximum_cheap_charge_kwh = power_kw * timestep_hours * cheap_steps
        if usable_capacity_kwh > maximum_cheap_charge_kwh + 1e-9:
            raise ValueError(
                "Brain2 cannot fill the entire usable battery inside the cheap window "
                "at the configured battery power"
            )

        discharge_normal_steps = 0
        discharge_expensive_steps = 0
        for step in range(cheap_end_step, steps_per_day):
            if expensive_start_step <= step < expensive_end_step:
                discharge_expensive_steps += 1
            else:
                discharge_normal_steps += 1

        if discharge_normal_steps + discharge_expensive_steps <= 0:
            raise ValueError("Brain2 needs at least one post-cheap timestep before midnight")

        weighted_tariff_steps = (
            discharge_normal_steps * normal_tariff
            + discharge_expensive_steps * expensive_tariff
        )
        action_scale = usable_capacity_kwh / (
            power_kw * timestep_hours * weighted_tariff_steps
        )
        normal_action = action_scale * normal_tariff
        expensive_action = action_scale * expensive_tariff

        if normal_action > 1.0 + 1e-12 or expensive_action > 1.0 + 1e-12:
            raise ValueError(
                "Brain2 weighted discharge schedule requires an action above +1; "
                "the configured battery cannot follow this tariff weighting and still "
                "reach minimum SOC at midnight"
            )

        object.__setattr__(self, "usable_capacity_kwh", usable_capacity_kwh)
        object.__setattr__(self, "normal_action", min(1.0, normal_action))
        object.__setattr__(self, "expensive_action", min(1.0, expensive_action))
        object.__setattr__(self, "steps_per_day", steps_per_day)
        object.__setattr__(self, "cheap_steps", cheap_steps)
        object.__setattr__(self, "discharge_normal_steps", discharge_normal_steps)
        object.__setattr__(self, "discharge_expensive_steps", discharge_expensive_steps)

    def act(self, observation: BrainObservation) -> float:
        """Return Brain2's single requested action in [-1, 1]."""
        return self.decide(observation).action

    def decide(self, observation: BrainObservation) -> Brain2Decision:
        """Return the action plus named arithmetic showing exactly why Brain2 chose it."""
        try:
            observation_length = len(observation)
        except TypeError as exc:
            raise TypeError("Brain2 observation must be a seven-value sequence") from exc
        if observation_length != 7:
            raise ValueError("Brain2 observation must contain exactly 7 values")

        values: list[float] = []
        for value in observation:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("Brain2 observation values must be numeric") from exc
            if not math.isfinite(numeric_value):
                raise ValueError("Brain2 observation values must all be finite")
            values.append(numeric_value)

        time_sin = values[0]
        time_cos = values[1]
        normalized_soc = values[3]
        normalized_tariff = values[4]

        if normalized_soc < 0.0 or normalized_soc > 1.0:
            raise ValueError("Brain2 SOC eye must be inside [0, 1]")
        if normalized_tariff < 0.0 or normalized_tariff > 1.0:
            raise ValueError("Brain2 tariff eye must be inside [0, 1]")

        time_radius = math.hypot(time_sin, time_cos)
        if not math.isclose(time_radius, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Brain2 time eyes must lie on the unit circle")

        angle = math.atan2(time_sin, time_cos)
        if angle < 0.0:
            angle += 2.0 * math.pi
        raw_step = angle * self.steps_per_day / (2.0 * math.pi)
        nearest_step = round(raw_step)
        if not math.isclose(raw_step, nearest_step, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Brain2 time eyes must describe an exact native timestep")
        current_step = nearest_step % self.steps_per_day
        minute_of_day = current_step * self.timestep_minutes

        cheap_start_step = round(self.cheap_start_minute / self.timestep_minutes)
        cheap_end_step = round(self.cheap_end_minute / self.timestep_minutes)
        expensive_start_step = round(self.expensive_start_minute / self.timestep_minutes)
        expensive_end_step = round(self.expensive_end_minute / self.timestep_minutes)

        if cheap_start_step <= current_step < cheap_end_step:
            tariff_period = "cheap"
            remaining_cheap_steps = cheap_end_step - current_step
            remaining_room_kwh = (1.0 - normalized_soc) * self.usable_capacity_kwh
            if remaining_room_kwh <= 1e-12:
                return Brain2Decision(
                    action=0.0,
                    label="IDLE",
                    reason_code="cheap_but_full",
                    reason="Cheap time, but the usable battery is already full.",
                    tariff_period=tariff_period,
                    minute_of_day=minute_of_day,
                    normalized_soc=normalized_soc,
                    requested_battery_power_kw=0.0,
                    target_battery_energy_kwh=0.0,
                    remaining_cheap_steps=remaining_cheap_steps,
                    normal_action=self.normal_action,
                    expensive_action=self.expensive_action,
                )

            timestep_hours = self.timestep_minutes / 60.0
            required_battery_kwh_per_step = remaining_room_kwh / remaining_cheap_steps
            required_charge_kw = required_battery_kwh_per_step / timestep_hours
            required_action = required_charge_kw / self.battery_power_kw
            if required_action > 1.0 + 1e-12:
                raise RuntimeError(
                    "Brain2 cannot reach maximum SOC by the end of the cheap window "
                    "from the current SOC without requesting action below -1"
                )
            action = -min(1.0, required_action)
            return Brain2Decision(
                action=action,
                label="CHARGE",
                reason_code="adaptive_cheap_fill",
                reason=(
                    "Spread the remaining usable battery room evenly across the remaining "
                    "cheap timesteps so the planned SOC reaches full at cheap-window end."
                ),
                tariff_period=tariff_period,
                minute_of_day=minute_of_day,
                normalized_soc=normalized_soc,
                requested_battery_power_kw=action * self.battery_power_kw,
                target_battery_energy_kwh=remaining_room_kwh,
                remaining_cheap_steps=remaining_cheap_steps,
                normal_action=self.normal_action,
                expensive_action=self.expensive_action,
            )

        remaining_cheap_steps = 0
        available_battery_kwh = normalized_soc * self.usable_capacity_kwh
        if available_battery_kwh <= 1e-12:
            if expensive_start_step <= current_step < expensive_end_step:
                tariff_period = "expensive"
            else:
                tariff_period = "normal"
            return Brain2Decision(
                action=0.0,
                label="IDLE",
                reason_code=f"{tariff_period}_but_empty",
                reason=f"{tariff_period.capitalize()} time, but no usable battery energy remains.",
                tariff_period=tariff_period,
                minute_of_day=minute_of_day,
                normalized_soc=normalized_soc,
                requested_battery_power_kw=0.0,
                target_battery_energy_kwh=available_battery_kwh,
                remaining_cheap_steps=remaining_cheap_steps,
                normal_action=self.normal_action,
                expensive_action=self.expensive_action,
            )

        if expensive_start_step <= current_step < expensive_end_step:
            tariff_period = "expensive"
            action = self.expensive_action
            reason_code = "weighted_expensive_discharge"
            reason = (
                "Expensive time gets the larger constant tariff-weighted discharge action."
            )
        else:
            tariff_period = "normal"
            action = self.normal_action
            reason_code = "weighted_normal_discharge"
            reason = "Normal time gets the smaller constant tariff-weighted discharge action."

        return Brain2Decision(
            action=action,
            label="DISCHARGE",
            reason_code=reason_code,
            reason=reason,
            tariff_period=tariff_period,
            minute_of_day=minute_of_day,
            normalized_soc=normalized_soc,
            requested_battery_power_kw=action * self.battery_power_kw,
            target_battery_energy_kwh=available_battery_kwh,
            remaining_cheap_steps=remaining_cheap_steps,
            normal_action=self.normal_action,
            expensive_action=self.expensive_action,
        )
