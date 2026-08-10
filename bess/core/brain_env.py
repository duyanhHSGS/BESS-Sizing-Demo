# ============================================================
# WHAT THIS BESS BRAIN CAN SEE — HUMAN LANGUAGE
#
# Eye 1 + Eye 2: What time of day is it?
#   - Time is represented by TWO numbers (sin + cos) so midnight wraps around smoothly.
#
# Eye 3: How much electricity does the factory need from the grid right now?
#   - This file receives NET LOAD directly. PV does not exist as a separate input here.
#   - Any PV effect must already be included in net_load_kw before it reaches this env.
#
# Eye 4: How full is the battery?
#   - 0.0 = at minimum allowed SOC, 1.0 = at maximum allowed SOC.
#
# Eye 5: How expensive is electricity right now?
#   - Current tariff divided by the biggest tariff in the tariff schedule.
#
# Eye 6: How scary is the monthly demand peak so far?
#   - The highest monthly peak seen so far, normalized by the temporary power ruler.
#
# Eye 7: Is this a working day?
#   - 1.0 = yes, 0.0 = no.
#
# TOTAL: 7 numbers go into the brain.
#
# WHAT THIS BESS BRAIN CAN DO — ONE BABY LEVER
#
# The brain outputs ONE action number from -1.0 to +1.0.
#   -1.0 = ask for maximum charging
#    0.0 = do nothing
#   +1.0 = ask for maximum discharging
#
# Requested battery power = action * battery rated power.
# Example with a 450 kW battery:
#   action -1.0 -> requested power -450 kW (charge)
#   action -0.5 -> requested power -225 kW (charge)
#   action  0.0 -> requested power    0 kW (nap)
#   action +0.5 -> requested power +225 kW (discharge)
#   action +1.0 -> requested power +450 kW (discharge)
#
# TWO SEPARATE GUARDS — DO NOT MIX THEM:
#
# BATTERY POLICE:
#   - Pure battery-side only.
#   - Knows SOC, SOC min/max, battery capacity, and timestep.
#   - Clips requested BATTERY-SIDE power so the battery cannot cross SOC min/max.
#   - Does NOT know net load, grid, factory, tariff, PV, or efficiency.
#
# GRID GUARD:
#   - Outside-world no-export rule only.
#   - Knows net load and discharge efficiency.
#   - Converts battery-side discharge into outside delivered power and clips it so
#     delivered power cannot exceed net load.
#   - Charging is untouched by the no-export guard.
#
# SOC PHYSICS:
#   - Uses the FINAL battery-side power after both guards.
#   - Battery-side power changes stored battery energy exactly; efficiency does NOT
#     change SOC in this model. Efficiency belongs to the outside conversion layer.
#
# This file is a standalone BESS brain playground and is NOT wired into the project.
# ============================================================

from __future__ import annotations

import math


OBSERVATION_DIM = 7
ACTION_DIM = 1
ACTION_MIN = -1.0
ACTION_MAX = 1.0

# TODO BIG JUICY TODO: 1000 kW is an arbitrary temporary ruler. Replace this with a normalization rule that is principled for the site/data and does not silently erase large values.
POWER_SCALE_KW = 1000.0


def action_to_requested_battery_power_kw(action: float, battery_power_kw: float) -> float:
    """Turn the brain's one action into requested BATTERY-SIDE power in kW.

    Negative = charge, zero = idle, positive = discharge.
    The returned value is only a request; Battery Police and Grid Guard may reduce it.
    """
    # TODO: check at init.
    if battery_power_kw <= 0.0:
        raise ValueError("battery_power_kw must be greater than 0")
    # TODO: should this exist when learner use tanh?
    clipped_action = min(ACTION_MAX, max(ACTION_MIN, float(action)))
    return clipped_action * float(battery_power_kw)


def police_battery_power(
    *,
    requested_battery_power_kw: float,
    state_of_charge: float,
    minimum_state_of_charge: float,
    maximum_state_of_charge: float,
    battery_capacity_kwh: float,
    timestep_hours: float,
) -> float:
    """Battery Police: enforce SOC limits using pure BATTERY-SIDE quantities only.

    Returns the SOC-safe battery-side power in kW.

    Sign convention:
      negative power = charge battery
      positive power = discharge battery

    This function deliberately knows NOTHING about net load, grid power,
    factory power, PV, tariff, or efficiency.
    """
    minimum_soc = float(minimum_state_of_charge)
    maximum_soc = float(maximum_state_of_charge)
    capacity_kwh = float(battery_capacity_kwh)
    dt_hours = float(timestep_hours)
    current_soc = float(state_of_charge)
    requested_power_kw = float(requested_battery_power_kw)

    if not all(
        math.isfinite(value)
        for value in (
            minimum_soc,
            maximum_soc,
            capacity_kwh,
            dt_hours,
            current_soc,
            requested_power_kw,
        )
    ):
        raise ValueError("Battery Police inputs must all be finite numbers")
    if maximum_soc <= minimum_soc:
        raise ValueError("maximum_state_of_charge must be greater than minimum_state_of_charge")
    if capacity_kwh <= 0.0:
        raise ValueError("battery_capacity_kwh must be greater than 0")
    if dt_hours <= 0.0:
        raise ValueError("timestep_hours must be greater than 0")
    if current_soc < minimum_soc or current_soc > maximum_soc:
        raise ValueError(
            "state_of_charge must already be inside "
            "[minimum_state_of_charge, maximum_state_of_charge]"
        )

    if requested_power_kw > 0.0:
        # Pure battery-side discharge limit: how much stored energy exists above SOC_min?
        available_battery_energy_kwh = (current_soc - minimum_soc) * capacity_kwh
        maximum_soc_safe_discharge_kw = available_battery_energy_kwh / dt_hours
        return min(requested_power_kw, maximum_soc_safe_discharge_kw)

    if requested_power_kw < 0.0:
        # Pure battery-side charge limit: how much empty battery room exists below SOC_max?
        available_battery_room_kwh = (maximum_soc - current_soc) * capacity_kwh
        maximum_soc_safe_charge_kw = available_battery_room_kwh / dt_hours
        return -min(-requested_power_kw, maximum_soc_safe_charge_kw)

    return 0.0


def grid_guard_no_export(
    *,
    battery_power_kw: float,
    net_load_kw: float,
    discharge_efficiency: float,
) -> float:
    """Grid Guard: enforce no export while keeping power expressed battery-side.

    ``battery_power_kw`` is already a BATTERY-SIDE number.
    Positive battery power is discharge. Only discharge can create export.

    Example with efficiency=0.9:
      +100 battery kW -> +90 kW delivered outside the battery.

    The guard clips battery-side discharge so outside delivered power never exceeds
    the non-negative net load. Charging passes through unchanged.
    """
    battery_side_power_kw = float(battery_power_kw)
    final_net_load_kw = float(net_load_kw)
    efficiency = float(discharge_efficiency)

    if not all(
        math.isfinite(value)
        for value in (
            battery_side_power_kw,
            final_net_load_kw,
            efficiency,
        )
    ):
        raise ValueError("Grid Guard inputs must all be finite numbers")
    if efficiency <= 0.0 or efficiency > 1.0:
        raise ValueError("discharge_efficiency must be greater than 0 and at most 1")

    # Charging cannot create export, so no-export does not modify charge power.
    if battery_side_power_kw <= 0.0:
        return battery_side_power_kw

    # net_load_kw is the already-prepared outside-world demand signal.
    # If it is negative, there is zero demand available for battery discharge.
    non_negative_net_load_kw = max(0.0, final_net_load_kw)

    # outside_delivered_kw = battery_side_discharge_kw * discharge_efficiency
    # Therefore battery_side_discharge_kw may be at most net_load / efficiency.
    maximum_no_export_battery_discharge_kw = non_negative_net_load_kw / efficiency
    return min(battery_side_power_kw, maximum_no_export_battery_discharge_kw)


def next_battery_state_of_charge(
    *,
    state_of_charge: float,
    battery_power_kw: float,
    battery_capacity_kwh: float,
    timestep_hours: float,
    minimum_state_of_charge: float,
    maximum_state_of_charge: float,
) -> float:
    """Apply FINAL battery-side power to SOC with no efficiency term.

    Positive battery power removes stored energy.
    Negative battery power adds stored energy.
    """
    current_soc = float(state_of_charge)
    battery_side_power_kw = float(battery_power_kw)
    capacity_kwh = float(battery_capacity_kwh)
    dt_hours = float(timestep_hours)
    minimum_soc = float(minimum_state_of_charge)
    maximum_soc = float(maximum_state_of_charge)

    if not all(
        math.isfinite(value)
        for value in (
            current_soc,
            battery_side_power_kw,
            capacity_kwh,
            dt_hours,
            minimum_soc,
            maximum_soc,
        )
    ):
        raise ValueError("SOC physics inputs must all be finite numbers")
    if capacity_kwh <= 0.0:
        raise ValueError("battery_capacity_kwh must be greater than 0")
    if dt_hours <= 0.0:
        raise ValueError("timestep_hours must be greater than 0")
    if maximum_soc <= minimum_soc:
        raise ValueError("maximum_state_of_charge must be greater than minimum_state_of_charge")

    next_soc = current_soc - (battery_side_power_kw * dt_hours / capacity_kwh)

    numerical_tolerance = 1e-12
    if next_soc < minimum_soc - numerical_tolerance or next_soc > maximum_soc + numerical_tolerance:
        raise RuntimeError(
            "Final battery-side power would push SOC outside the legal range; "
            "run Battery Police before SOC physics"
        )
    if next_soc < minimum_soc:
        return minimum_soc
    if next_soc > maximum_soc:
        return maximum_soc
    return next_soc


def build_observation(
    *,
    timestep_index: int,
    steps_per_day: int,
    net_load_kw: float,
    state_of_charge: float,
    minimum_state_of_charge: float,
    maximum_state_of_charge: float,
    tariff_vnd_per_kwh: float,
    maximum_tariff_vnd_per_kwh: float,
    monthly_peak_kw: float,
    is_working_day: bool,
    power_scale_kw: float = POWER_SCALE_KW,
) -> tuple[float, float, float, float, float, float, float]:
    """Build the tiny 7-eye observation the agent gets to see."""
    if steps_per_day <= 0:
        raise ValueError("steps_per_day must be greater than 0")
    if power_scale_kw <= 0.0:
        raise ValueError("power_scale_kw must be greater than 0")
    if maximum_tariff_vnd_per_kwh <= 0.0:
        raise ValueError("maximum_tariff_vnd_per_kwh must be greater than 0")
    if maximum_state_of_charge <= minimum_state_of_charge:
        raise ValueError("maximum_state_of_charge must be greater than minimum_state_of_charge")

    time_angle = 2.0 * math.pi * (timestep_index % steps_per_day) / steps_per_day
    time_sin = math.sin(time_angle)
    time_cos = math.cos(time_angle)

    # PV is already baked into this value before it enters brain_env.
    # No separate load/PV arithmetic exists in this env.
    grid_facing_net_load_kw = max(float(net_load_kw), 0.0)

    # TODO BIG JUICY TODO: net load currently uses the temporary fixed 1000 kW-style ruler above. Find a better site-aware normalization later.
    normalized_net_load = grid_facing_net_load_kw / power_scale_kw

    normalized_state_of_charge = (
        (float(state_of_charge) - float(minimum_state_of_charge))
        / (float(maximum_state_of_charge) - float(minimum_state_of_charge))
    )
    normalized_state_of_charge = min(1.0, max(0.0, normalized_state_of_charge))

    # TODO: current tariff divided by the biggest tariff in the tariff schedule. Not log; later research how good/bad log is.
    normalized_tariff = max(float(tariff_vnd_per_kwh), 0.0) / float(maximum_tariff_vnd_per_kwh)

    # TODO BIG JUICY TODO: monthly peak shares the same arbitrary temporary power ruler as net load. Replace this with something principled later.
    normalized_monthly_peak = max(float(monthly_peak_kw), 0.0) / power_scale_kw

    working_day = 1.0 if is_working_day else 0.0

    return (
        time_sin,
        time_cos,
        normalized_net_load,
        normalized_state_of_charge,
        normalized_tariff,
        normalized_monthly_peak,
        working_day,
    )
