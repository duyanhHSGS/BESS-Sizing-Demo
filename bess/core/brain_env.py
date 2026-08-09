# ============================================================
# WHAT THIS BESS BRAIN CAN SEE — HUMAN LANGUAGE
#
# Eye 1 + Eye 2: What time of day is it?
#   - Time is represented by TWO numbers (sin + cos) so midnight wraps around smoothly.
#
# Eye 3: How much electricity does the factory need from somewhere right now?
#   - Net load = factory load minus solar, never below zero.
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
# BIG JUICY TODO — PHYSICS POLICE DOES NOT EXIST YET:
# This is only what the brain REQUESTS, not guaranteed actual battery power.
# Later, add physical limits for SOC min/max, charge/discharge efficiency,
# available energy, rated power behavior, timestep energy conversion, grid export,
# and any other real battery constraints. DO NOT silently call requested power actual power.
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
    """Turn the brain's one action into a requested battery power in kW.

    Negative = charge, zero = idle, positive = discharge.
    This function deliberately does NOT apply battery physics yet.
    """
    # TODO: check at init.
    if battery_power_kw <= 0.0:
        raise ValueError("battery_power_kw must be greater than 0")
    # TODO: should this exist when learner use tanh?
    clipped_action = min(ACTION_MAX, max(ACTION_MIN, float(action)))
    return clipped_action * float(battery_power_kw)


def build_observation(
    *,
    timestep_index: int,
    steps_per_day: int,
    load_kw: float,
    pv_kw: float,
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

    net_load_kw = max(float(load_kw) - float(pv_kw), 0.0)

    # TODO BIG JUICY TODO: net load currently uses the temporary fixed 1000 kW-style ruler above. Find a better site-aware normalization later.
    normalized_net_load = net_load_kw / power_scale_kw

    normalized_state_of_charge = (
        (float(state_of_charge) - float(minimum_state_of_charge))
        / (float(maximum_state_of_charge) - float(minimum_state_of_charge))
    )
    normalized_state_of_charge = min(1.0, max(0.0, normalized_state_of_charge))

    # TODO: ent tariff divided by the biggest tariff in the tariff schedule. not log, later research how good/bad log is.
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
