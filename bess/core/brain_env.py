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
# PHYSICS POLICE V1:
#   LAW 1: SOC may never leave [minimum_state_of_charge, maximum_state_of_charge].
#          Requested battery power is clipped BEFORE physics so energy is not
#          magically created/deleted by clipping SOC after an illegal action.
#   LAW 2: No export. Discharge may never exceed the current non-negative net load.
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
    """Turn the brain's one action into requested battery power in kW.
    Negative = charge, zero = idle, positive = discharge.
    The returned value is only a request; Physics Police must approve it.
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
    net_load_kw: float,
    state_of_charge: float,
    minimum_state_of_charge: float,
    maximum_state_of_charge: float,
    battery_capacity_kwh: float,
    timestep_hours: float,
) -> tuple[float, float]:
    """Apply the two Physics Police laws.
    Returns ``(actual_battery_power_kw, next_state_of_charge)``.
    Sign convention:
      negative power = charge
      positive power = discharge
    V1 deliberately knows nothing about PV, tariffs, rewards, or strategy.
    ``net_load_kw`` is already the final load signal supplied to this env.

    TODO FUTURE PHYSICS:
      - Charge/discharge efficiency is NOT modeled yet; current SOC math assumes 100% efficiency.
      - PV is NOT handled here. Any PV effect must already be baked into ``net_load_kw`` upstream.
        How PV/net-load preprocessing should work is a future problem for this scratch env.
    """
    minimum_soc = float(minimum_state_of_charge)
    maximum_soc = float(maximum_state_of_charge)
    capacity_kwh = float(battery_capacity_kwh)
    dt_hours = float(timestep_hours)
    current_soc = float(state_of_charge)
    requested_power_kw = float(requested_battery_power_kw)
    final_net_load_kw = float(net_load_kw)

    if not all(
        math.isfinite(value)
        for value in (
            minimum_soc,
            maximum_soc,
            capacity_kwh,
            dt_hours,
            current_soc,
            requested_power_kw,
            final_net_load_kw,
        )
    ):
        raise ValueError("Physics Police inputs must all be finite numbers")
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
        # LAW 1 — do not discharge below minimum SOC.
        available_energy_kwh = (current_soc - minimum_soc) * capacity_kwh
        maximum_soc_safe_discharge_kw = available_energy_kwh / dt_hours

        # LAW 2 — no export. With no separate PV concept, discharge can only
        # serve the already-computed positive net load.
        maximum_no_export_discharge_kw = max(0.0, final_net_load_kw)

        actual_power_kw = min(
            requested_power_kw,
            maximum_soc_safe_discharge_kw,
            maximum_no_export_discharge_kw,
        )

    elif requested_power_kw < 0.0:
        # LAW 1 — do not charge above maximum SOC.
        available_room_kwh = (maximum_soc - current_soc) * capacity_kwh
        maximum_soc_safe_charge_kw = available_room_kwh / dt_hours
        actual_power_kw = -min(-requested_power_kw, maximum_soc_safe_charge_kw)

    else:
        actual_power_kw = 0.0

    # Positive power discharges the battery; negative power charges it.
    next_state_of_charge = current_soc - (actual_power_kw * dt_hours / capacity_kwh)

    # Do not hide physics bugs by broadly clipping SOC after the fact.
    # Only snap microscopic floating-point roundoff back onto the exact boundary.
    numerical_tolerance = 1e-12
    if next_state_of_charge < minimum_soc - numerical_tolerance or next_state_of_charge > maximum_soc + numerical_tolerance:
        raise RuntimeError("Physics Police produced an out-of-range next_state_of_charge")
    if next_state_of_charge < minimum_soc:
        next_state_of_charge = minimum_soc
    elif next_state_of_charge > maximum_soc:
        next_state_of_charge = maximum_soc

    return actual_power_kw, next_state_of_charge


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
