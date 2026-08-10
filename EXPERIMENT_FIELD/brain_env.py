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
# OUTSIDE ELECTRICAL PHYSICS:
#   - Converts FINAL battery-side power into the power seen by the factory/grid.
#   - Discharge: outside gets battery_power * discharge_efficiency.
#   - Charge: grid must provide battery_charge / charge_efficiency.
#   - Final grid import = non-negative net load - signed outside battery power.
#   - Grid import must never become negative; Grid Guard must run first.
#
# SOC PHYSICS:
#   - Uses the FINAL battery-side power after both guards.
#   - Battery-side power changes stored battery energy exactly; efficiency does NOT
#     change SOC in this model. Efficiency belongs only to outside electrical physics.
#
# UTILITY ELECTRICITY METER:
#   - Sees ONLY final grid_import_kw from physics.
#   - Integrates energy inside fixed, clock-aligned, non-overlapping 30-minute blocks.
#   - A monthly demand peak changes only when a complete 30-minute block finishes.
#   - Knows NOTHING about tariff, money, battery wear, reward, or PPO.
#
# MONEY ACCOUNTANT:
#   - Sees only finished accounting facts: grid energy, battery throughput, and
#     monthly-peak increase, plus their prices.
#   - Energy cost = grid energy * tariff.
#   - Wear cost = BATTERY-SIDE throughput * wear rate. No efficiency appears here.
#   - Demand cost = increase in the completed 30-minute monthly peak * demand fee.
#   - Operating cost = energy + demand + wear. No reward, PPO, or physical control.
#
# This file is a standalone BESS brain playground and is NOT wired into the project.
# ============================================================

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


OBSERVATION_DIM = 7
ACTION_DIM = 1
ACTION_MIN = -1.0
ACTION_MAX = 1.0

# TODO BIG JUICY TODO: 1000 kW is an arbitrary temporary ruler. Replace this with a normalization rule that is principled for the site/data and does not silently erase large values.
POWER_SCALE_KW = 1000.0
DEMAND_BLOCK_HOURS = 0.5


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


def battery_power_to_outside_power_kw(
    *,
    battery_power_kw: float,
    charge_efficiency: float,
    discharge_efficiency: float,
) -> float:
    """Convert FINAL battery-side power into signed outside-world power.

    Sign convention stays the same on both sides:
      positive = battery supplies power outward (discharge)
      negative = outside world must supply power inward (charge)

    Battery-side energy is exact in this model. Efficiency only changes what the
    factory/grid sees outside the battery.

    Examples with 90% efficiency:
      +100 battery kW discharge -> +90 outside kW delivered
      -100 battery kW charge    -> -111.111... outside kW demanded
    """
    battery_side_power_kw = float(battery_power_kw)
    charge_eta = float(charge_efficiency)
    discharge_eta = float(discharge_efficiency)

    if not all(
        math.isfinite(value)
        for value in (
            battery_side_power_kw,
            charge_eta,
            discharge_eta,
        )
    ):
        raise ValueError("Outside physics inputs must all be finite numbers")
    if charge_eta <= 0.0 or charge_eta > 1.0:
        raise ValueError("charge_efficiency must be greater than 0 and at most 1")
    if discharge_eta <= 0.0 or discharge_eta > 1.0:
        raise ValueError("discharge_efficiency must be greater than 0 and at most 1")

    if battery_side_power_kw > 0.0:
        return battery_side_power_kw * discharge_eta
    if battery_side_power_kw < 0.0:
        return battery_side_power_kw / charge_eta
    return 0.0


def grid_import_from_outside_power_kw(
    *,
    net_load_kw: float,
    outside_battery_power_kw: float,
) -> float:
    """Calculate final grid import from prepared net load and outside battery power.

    ``outside_battery_power_kw`` uses the same sign convention:
      positive discharge reduces grid import
      negative charge increases grid import

    No export is allowed. Grid Guard must have clipped discharge before this function.
    """
    prepared_net_load_kw = float(net_load_kw)
    outside_power_kw = float(outside_battery_power_kw)

    if not math.isfinite(prepared_net_load_kw) or not math.isfinite(outside_power_kw):
        raise ValueError("Grid physics inputs must all be finite numbers")

    non_negative_net_load_kw = max(0.0, prepared_net_load_kw)
    grid_import_kw = non_negative_net_load_kw - outside_power_kw

    numerical_tolerance = 1e-12
    if grid_import_kw < -numerical_tolerance:
        raise RuntimeError(
            "Outside battery discharge would export power; run Grid Guard before grid physics"
        )
    if grid_import_kw < 0.0:
        return 0.0
    return grid_import_kw


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


@dataclass(frozen=True, slots=True)
class PhysicsStepResult:
    """Everything that physically happened during one BESS timestep.

    All battery-power fields use the battery-side sign convention:
      negative = charge
      positive = discharge

    Outside-world power is deliberately split into non-negative directional fields
    so callers never need to decode a signed "outside battery power" value.
    """

    requested_battery_kw: float
    battery_after_police_kw: float
    final_battery_kw: float
    battery_to_factory_kw: float
    grid_to_battery_kw: float
    conversion_loss_kw: float
    grid_import_kw: float
    battery_throughput_kwh: float
    starting_soc: float
    next_soc: float


def run_physics_step(
    *,
    action: float,
    net_load_kw: float,
    state_of_charge: float,
    minimum_state_of_charge: float,
    maximum_state_of_charge: float,
    battery_capacity_kwh: float,
    battery_power_kw: float,
    timestep_hours: float,
    charge_efficiency: float,
    discharge_efficiency: float,
) -> PhysicsStepResult:
    """Run exactly one physical BESS timestep, with no billing and no reward.

    Order is intentionally explicit:
      1. Brain action -> requested battery-side power.
      2. Battery Police -> SOC-safe battery-side power.
      3. Grid Guard -> no-export-safe final battery-side power.
      4. Outside conversion -> directional power flows and conversion loss.
      5. Battery physics -> next SOC from final battery-side power.
      6. Grid physics -> final grid import.

    This function is only the boss/orchestrator. The individual physical laws stay
    inside the existing helper functions so there is one source of truth per rule.
    """
    requested_battery_kw = action_to_requested_battery_power_kw(
        action,
        battery_power_kw,
    )

    battery_after_police_kw = police_battery_power(
        requested_battery_power_kw=requested_battery_kw,
        state_of_charge=state_of_charge,
        minimum_state_of_charge=minimum_state_of_charge,
        maximum_state_of_charge=maximum_state_of_charge,
        battery_capacity_kwh=battery_capacity_kwh,
        timestep_hours=timestep_hours,
    )

    final_battery_kw = grid_guard_no_export(
        battery_power_kw=battery_after_police_kw,
        net_load_kw=net_load_kw,
        discharge_efficiency=discharge_efficiency,
    )

    outside_battery_power_kw = battery_power_to_outside_power_kw(
        battery_power_kw=final_battery_kw,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
    )

    if final_battery_kw > 0.0:
        battery_to_factory_kw = outside_battery_power_kw
        grid_to_battery_kw = 0.0
        conversion_loss_kw = final_battery_kw - battery_to_factory_kw
    elif final_battery_kw < 0.0:
        battery_to_factory_kw = 0.0
        grid_to_battery_kw = -outside_battery_power_kw
        battery_charge_kw = -final_battery_kw
        conversion_loss_kw = grid_to_battery_kw - battery_charge_kw
    else:
        battery_to_factory_kw = 0.0
        grid_to_battery_kw = 0.0
        conversion_loss_kw = 0.0

    next_soc = next_battery_state_of_charge(
        state_of_charge=state_of_charge,
        battery_power_kw=final_battery_kw,
        battery_capacity_kwh=battery_capacity_kwh,
        timestep_hours=timestep_hours,
        minimum_state_of_charge=minimum_state_of_charge,
        maximum_state_of_charge=maximum_state_of_charge,
    )

    grid_import_kw = grid_import_from_outside_power_kw(
        net_load_kw=net_load_kw,
        outside_battery_power_kw=outside_battery_power_kw,
    )
    battery_throughput_kwh = abs(final_battery_kw) * float(timestep_hours)

    return PhysicsStepResult(
        requested_battery_kw=requested_battery_kw,
        battery_after_police_kw=battery_after_police_kw,
        final_battery_kw=final_battery_kw,
        battery_to_factory_kw=battery_to_factory_kw,
        grid_to_battery_kw=grid_to_battery_kw,
        conversion_loss_kw=conversion_loss_kw,
        grid_import_kw=grid_import_kw,
        battery_throughput_kwh=battery_throughput_kwh,
        starting_soc=float(state_of_charge),
        next_soc=next_soc,
    )


@dataclass(frozen=True, slots=True)
class ElectricityMeterState:
    """Memory carried by the fixed 30-minute utility demand meter.

    ``block_energy_kwh`` and ``block_elapsed_hours`` describe only the currently
    open 30-minute block. ``monthly_peak_kw`` is the highest COMPLETED block
    demand seen in the current month.
    """

    block_energy_kwh: float = 0.0
    block_elapsed_hours: float = 0.0
    monthly_peak_kw: float = 0.0


@dataclass(frozen=True, slots=True)
class ElectricityMeterStepResult:
    """Named answer to: what did the utility meter see on this timestep?"""

    grid_import_kw: float
    sample_energy_kwh: float
    accumulated_block_energy_kwh: float
    accumulated_block_elapsed_hours: float
    block_completed: bool
    completed_block_demand_kw: float | None
    previous_monthly_peak_kw: float
    monthly_peak_kw: float
    new_monthly_peak: bool
    next_state: ElectricityMeterState


def _validate_meter_timestep(timestep_hours: float) -> int:
    """Return samples per fixed 30-minute block, rejecting awkward resolutions."""
    dt_hours = float(timestep_hours)
    if not math.isfinite(dt_hours) or dt_hours <= 0.0:
        raise ValueError("timestep_hours must be a finite number greater than 0")

    samples_per_block = DEMAND_BLOCK_HOURS / dt_hours
    rounded_samples_per_block = round(samples_per_block)
    if rounded_samples_per_block < 1 or not math.isclose(
        samples_per_block,
        rounded_samples_per_block,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("timestep_hours must divide a fixed 30-minute demand block exactly")
    return int(rounded_samples_per_block)


def run_electricity_meter_step(
    *,
    grid_import_kw: float,
    timestep_hours: float,
    meter_state: ElectricityMeterState,
) -> ElectricityMeterStepResult:
    """Feed one grid-import sample into the fixed, non-overlapping 30-minute meter.

    The meter knows nothing about battery physics, tariff, billing, wear, reward,
    PPO, or episodes. It only integrates grid-import energy inside one fixed
    30-minute block and updates the monthly maximum when that block completes.
    """
    _validate_meter_timestep(timestep_hours)

    grid_kw = float(grid_import_kw)
    dt_hours = float(timestep_hours)
    if not math.isfinite(grid_kw):
        raise ValueError("grid_import_kw must be finite")
    if grid_kw < 0.0:
        raise ValueError("grid_import_kw must not be negative")
    if not isinstance(meter_state, ElectricityMeterState):
        raise TypeError("meter_state must be an ElectricityMeterState")

    previous_block_energy_kwh = float(meter_state.block_energy_kwh)
    previous_block_elapsed_hours = float(meter_state.block_elapsed_hours)
    previous_monthly_peak_kw = float(meter_state.monthly_peak_kw)
    if not all(
        math.isfinite(value)
        for value in (
            previous_block_energy_kwh,
            previous_block_elapsed_hours,
            previous_monthly_peak_kw,
        )
    ):
        raise ValueError("Electricity meter state values must all be finite")
    if previous_block_energy_kwh < 0.0:
        raise ValueError("meter_state.block_energy_kwh must not be negative")
    if previous_block_elapsed_hours < 0.0 or previous_block_elapsed_hours >= DEMAND_BLOCK_HOURS:
        raise ValueError("meter_state.block_elapsed_hours must be inside [0, 0.5)")
    if previous_block_elapsed_hours == 0.0 and previous_block_energy_kwh != 0.0:
        raise ValueError("an empty meter block cannot already contain energy")
    if previous_monthly_peak_kw < 0.0:
        raise ValueError("meter_state.monthly_peak_kw must not be negative")

    # A carried state must sit exactly on this timestep's sample grid. This keeps
    # the meter fixed/clock-aligned instead of silently inventing partial samples.
    elapsed_samples = previous_block_elapsed_hours / dt_hours
    if not math.isclose(elapsed_samples, round(elapsed_samples), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("meter_state.block_elapsed_hours is not aligned to timestep_hours")

    sample_energy_kwh = grid_kw * dt_hours
    accumulated_block_energy_kwh = previous_block_energy_kwh + sample_energy_kwh
    accumulated_block_elapsed_hours = previous_block_elapsed_hours + dt_hours

    numerical_tolerance = 1e-12
    if accumulated_block_elapsed_hours > DEMAND_BLOCK_HOURS + numerical_tolerance:
        raise RuntimeError("meter state would overrun the fixed 30-minute demand block")

    block_completed = math.isclose(
        accumulated_block_elapsed_hours,
        DEMAND_BLOCK_HOURS,
        rel_tol=0.0,
        abs_tol=numerical_tolerance,
    )

    completed_block_demand_kw: float | None = None
    monthly_peak_kw = previous_monthly_peak_kw
    new_monthly_peak = False

    if block_completed:
        completed_block_demand_kw = accumulated_block_energy_kwh / DEMAND_BLOCK_HOURS
        new_monthly_peak = completed_block_demand_kw > previous_monthly_peak_kw
        monthly_peak_kw = max(previous_monthly_peak_kw, completed_block_demand_kw)
        next_state = ElectricityMeterState(monthly_peak_kw=monthly_peak_kw)
    else:
        next_state = ElectricityMeterState(
            block_energy_kwh=accumulated_block_energy_kwh,
            block_elapsed_hours=accumulated_block_elapsed_hours,
            monthly_peak_kw=monthly_peak_kw,
        )

    return ElectricityMeterStepResult(
        grid_import_kw=grid_kw,
        sample_energy_kwh=sample_energy_kwh,
        accumulated_block_energy_kwh=accumulated_block_energy_kwh,
        accumulated_block_elapsed_hours=accumulated_block_elapsed_hours,
        block_completed=block_completed,
        completed_block_demand_kw=completed_block_demand_kw,
        previous_monthly_peak_kw=previous_monthly_peak_kw,
        monthly_peak_kw=monthly_peak_kw,
        new_monthly_peak=new_monthly_peak,
        next_state=next_state,
    )


def reset_electricity_meter_for_new_day(meter_state: ElectricityMeterState) -> ElectricityMeterState:
    """Start a new clock day: clear the open block, preserve the monthly peak."""
    if not isinstance(meter_state, ElectricityMeterState):
        raise TypeError("meter_state must be an ElectricityMeterState")
    monthly_peak_kw = float(meter_state.monthly_peak_kw)
    if not math.isfinite(monthly_peak_kw) or monthly_peak_kw < 0.0:
        raise ValueError("meter_state.monthly_peak_kw must be finite and non-negative")
    return ElectricityMeterState(monthly_peak_kw=monthly_peak_kw)


def reset_electricity_meter_for_new_month() -> ElectricityMeterState:
    """Start a new billing month with an empty block and zero monthly peak."""
    return ElectricityMeterState()


@dataclass(frozen=True, slots=True)
class OperatingCostStepResult:
    """Boring accounting result for one timestep.
    It sees Grid pov! battery throughput! ZERO physic logic aurafarming.
    """
    grid_energy_kwh: float
    battery_throughput_kwh: float
    demand_peak_increase_kw: float
    electricity_energy_cost_vnd: float
    demand_cost_vnd: float
    battery_wear_cost_vnd: float
    operating_cost_vnd: float


def calculate_operating_cost_step(
    *,
    grid_energy_kwh: float,
    battery_throughput_kwh: float,
    previous_monthly_peak_kw: float,
    monthly_peak_kw: float,
    tariff_vnd_per_kwh: float,
    demand_charge_vnd_per_kw: float,
    battery_wear_vnd_per_kwh: float,
) -> OperatingCostStepResult:
    """Price finished accounting facts only: grid energy, peak increase, throughput."""
    grid_kwh = float(grid_energy_kwh)
    throughput_kwh = float(battery_throughput_kwh)
    previous_peak_kw = float(previous_monthly_peak_kw)
    current_peak_kw = float(monthly_peak_kw)
    tariff = float(tariff_vnd_per_kwh)
    demand_fee = float(demand_charge_vnd_per_kw)
    wear_rate = float(battery_wear_vnd_per_kwh)

    values = (
        grid_kwh,
        throughput_kwh,
        previous_peak_kw,
        current_peak_kw,
        tariff,
        demand_fee,
        wear_rate,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Operating-cost inputs must all be finite numbers")
    if grid_kwh < 0.0:
        raise ValueError("grid_energy_kwh must not be negative")
    if throughput_kwh < 0.0:
        raise ValueError("battery_throughput_kwh must not be negative")
    if previous_peak_kw < 0.0 or current_peak_kw < 0.0:
        raise ValueError("monthly demand peaks must not be negative")
    if tariff < 0.0:
        raise ValueError("tariff_vnd_per_kwh must not be negative")
    if demand_fee < 0.0:
        raise ValueError("demand_charge_vnd_per_kw must not be negative")
    if wear_rate < 0.0:
        raise ValueError("battery_wear_vnd_per_kwh must not be negative")

    numerical_tolerance = 1e-12
    if current_peak_kw < previous_peak_kw - numerical_tolerance:
        raise ValueError("monthly_peak_kw must not go backwards")
    demand_peak_increase_kw = max(0.0, current_peak_kw - previous_peak_kw)

    electricity_energy_cost_vnd = grid_kwh * tariff
    demand_cost_vnd = demand_peak_increase_kw * demand_fee
    battery_wear_cost_vnd = throughput_kwh * wear_rate
    operating_cost_vnd = electricity_energy_cost_vnd + demand_cost_vnd + battery_wear_cost_vnd

    return OperatingCostStepResult(
        grid_energy_kwh=grid_kwh,
        battery_throughput_kwh=throughput_kwh,
        demand_peak_increase_kw=demand_peak_increase_kw,
        electricity_energy_cost_vnd=electricity_energy_cost_vnd,
        demand_cost_vnd=demand_cost_vnd,
        battery_wear_cost_vnd=battery_wear_cost_vnd,
        operating_cost_vnd=operating_cost_vnd,
    )


@dataclass(frozen=True, slots=True)
class MonthlyOperatingCostResult:
    """Independent whole-month look-back from grid and battery power histories."""

    total_grid_energy_kwh: float
    battery_throughput_kwh: float
    monthly_peak_kw: float
    electricity_energy_cost_vnd: float
    demand_cost_vnd: float
    battery_wear_cost_vnd: float
    operating_cost_vnd: float


def calculate_month_operating_cost_from_history(
    *,
    grid_import_kw: Sequence[float],
    battery_power_kw: Sequence[float],
    tariff_vnd_per_kwh: Sequence[float],
    timestep_hours: float,
    demand_charge_vnd_per_kw: float,
    battery_wear_vnd_per_kwh: float,
) -> MonthlyOperatingCostResult:
    """Recalculate a complete month directly from raw grid and battery histories."""
    dt_hours = float(timestep_hours)
    samples_per_block = _validate_meter_timestep(dt_hours)
    demand_fee = float(demand_charge_vnd_per_kw)
    wear_rate = float(battery_wear_vnd_per_kwh)

    if not math.isfinite(demand_fee) or demand_fee < 0.0:
        raise ValueError("demand_charge_vnd_per_kw must be finite and non-negative")
    if not math.isfinite(wear_rate) or wear_rate < 0.0:
        raise ValueError("battery_wear_vnd_per_kwh must be finite and non-negative")

    sample_count = len(grid_import_kw)
    if sample_count == 0:
        raise ValueError("month history must contain at least one timestep")
    if len(battery_power_kw) != sample_count or len(tariff_vnd_per_kwh) != sample_count:
        raise ValueError("grid, battery, and tariff month histories must have equal length")
    if sample_count % samples_per_block != 0:
        raise ValueError("complete month history must end on a 30-minute demand-block boundary")

    total_grid_energy_kwh = 0.0
    battery_throughput_kwh = 0.0
    electricity_energy_cost_vnd = 0.0
    monthly_peak_kw = 0.0
    block_grid_energy_kwh = 0.0
    block_sample_count = 0

    # One O(N) pass, no temporary month-sized arrays.
    for grid_value, battery_value, tariff_value in zip(
        grid_import_kw,
        battery_power_kw,
        tariff_vnd_per_kwh,
    ):
        grid_kw = float(grid_value)
        battery_kw = float(battery_value)
        tariff = float(tariff_value)
        if not all(math.isfinite(value) for value in (grid_kw, battery_kw, tariff)):
            raise ValueError("month history values must all be finite")
        if grid_kw < 0.0:
            raise ValueError("grid_import_kw history must not contain negative values")
        if tariff < 0.0:
            raise ValueError("tariff history must not contain negative values")

        grid_energy_kwh = grid_kw * dt_hours
        total_grid_energy_kwh += grid_energy_kwh
        electricity_energy_cost_vnd += grid_energy_kwh * tariff
        battery_throughput_kwh += abs(battery_kw) * dt_hours

        block_grid_energy_kwh += grid_energy_kwh
        block_sample_count += 1
        if block_sample_count == samples_per_block:
            completed_block_demand_kw = block_grid_energy_kwh / DEMAND_BLOCK_HOURS
            monthly_peak_kw = max(monthly_peak_kw, completed_block_demand_kw)
            block_grid_energy_kwh = 0.0
            block_sample_count = 0

    demand_cost_vnd = monthly_peak_kw * demand_fee
    battery_wear_cost_vnd = battery_throughput_kwh * wear_rate
    operating_cost_vnd = electricity_energy_cost_vnd + demand_cost_vnd + battery_wear_cost_vnd

    return MonthlyOperatingCostResult(
        total_grid_energy_kwh=total_grid_energy_kwh,
        battery_throughput_kwh=battery_throughput_kwh,
        monthly_peak_kw=monthly_peak_kw,
        electricity_energy_cost_vnd=electricity_energy_cost_vnd,
        demand_cost_vnd=demand_cost_vnd,
        battery_wear_cost_vnd=battery_wear_cost_vnd,
        operating_cost_vnd=operating_cost_vnd,
    )


@dataclass(frozen=True, slots=True)
class BessWorldStepResult:
    """One timestep in the real world where the battery exists."""

    timestep_index: int
    physics: PhysicsStepResult
    meter: ElectricityMeterStepResult
    cost: OperatingCostStepResult


@dataclass(slots=True)
class BessWorld:
    """Stateful battery universe: battery physics -> meter -> shared accountant."""

    state_of_charge: float
    minimum_state_of_charge: float
    maximum_state_of_charge: float
    battery_capacity_kwh: float
    battery_power_kw: float
    timestep_hours: float
    charge_efficiency: float
    discharge_efficiency: float
    demand_charge_vnd_per_kw: float
    battery_wear_vnd_per_kwh: float
    meter_state: ElectricityMeterState = field(
        default_factory=ElectricityMeterState,
        init=False,
    )
    timestep_index: int = field(default=0, init=False)
    total_electricity_energy_cost_vnd: float = field(default=0.0, init=False)
    total_demand_cost_vnd: float = field(default=0.0, init=False)
    total_battery_wear_cost_vnd: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        values = (
            self.state_of_charge,
            self.minimum_state_of_charge,
            self.maximum_state_of_charge,
            self.battery_capacity_kwh,
            self.battery_power_kw,
            self.timestep_hours,
            self.charge_efficiency,
            self.discharge_efficiency,
            self.demand_charge_vnd_per_kw,
            self.battery_wear_vnd_per_kwh,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("BessWorld configuration values must all be finite")
        if self.battery_power_kw <= 0.0:
            raise ValueError("battery_power_kw must be greater than 0")
        if self.demand_charge_vnd_per_kw < 0.0:
            raise ValueError("demand_charge_vnd_per_kw must not be negative")
        if self.battery_wear_vnd_per_kwh < 0.0:
            raise ValueError("battery_wear_vnd_per_kwh must not be negative")

        # Reuse the existing laws as validators instead of inventing a second set of rules.
        _validate_meter_timestep(self.timestep_hours)
        police_battery_power(
            requested_battery_power_kw=0.0,
            state_of_charge=self.state_of_charge,
            minimum_state_of_charge=self.minimum_state_of_charge,
            maximum_state_of_charge=self.maximum_state_of_charge,
            battery_capacity_kwh=self.battery_capacity_kwh,
            timestep_hours=self.timestep_hours,
        )
        battery_power_to_outside_power_kw(
            battery_power_kw=0.0,
            charge_efficiency=self.charge_efficiency,
            discharge_efficiency=self.discharge_efficiency,
        )

    @property
    def total_operating_cost_vnd(self) -> float:
        return (
            self.total_electricity_energy_cost_vnd
            + self.total_demand_cost_vnd
            + self.total_battery_wear_cost_vnd
        )

    def step(
        self,
        *,
        action: float,
        net_load_kw: float,
        tariff_vnd_per_kwh: float,
    ) -> BessWorldStepResult:
        """Advance the battery universe exactly once, committing only after validation."""
        action_value = float(action)
        if not math.isfinite(action_value):
            raise ValueError("action must be finite")

        physics = run_physics_step(
            action=action_value,
            net_load_kw=net_load_kw,
            state_of_charge=self.state_of_charge,
            minimum_state_of_charge=self.minimum_state_of_charge,
            maximum_state_of_charge=self.maximum_state_of_charge,
            battery_capacity_kwh=self.battery_capacity_kwh,
            battery_power_kw=self.battery_power_kw,
            timestep_hours=self.timestep_hours,
            charge_efficiency=self.charge_efficiency,
            discharge_efficiency=self.discharge_efficiency,
        )
        meter = run_electricity_meter_step(
            grid_import_kw=physics.grid_import_kw,
            timestep_hours=self.timestep_hours,
            meter_state=self.meter_state,
        )
        cost = calculate_operating_cost_step(
            grid_energy_kwh=meter.sample_energy_kwh,
            battery_throughput_kwh=physics.battery_throughput_kwh,
            previous_monthly_peak_kw=meter.previous_monthly_peak_kw,
            monthly_peak_kw=meter.monthly_peak_kw,
            tariff_vnd_per_kwh=tariff_vnd_per_kwh,
            demand_charge_vnd_per_kw=self.demand_charge_vnd_per_kw,
            battery_wear_vnd_per_kwh=self.battery_wear_vnd_per_kwh,
        )

        result = BessWorldStepResult(
            timestep_index=self.timestep_index,
            physics=physics,
            meter=meter,
            cost=cost,
        )

        # Commit only after physics, metering, and money all produced valid results.
        self.state_of_charge = physics.next_soc
        self.meter_state = meter.next_state
        self.total_electricity_energy_cost_vnd += cost.electricity_energy_cost_vnd
        self.total_demand_cost_vnd += cost.demand_cost_vnd
        self.total_battery_wear_cost_vnd += cost.battery_wear_cost_vnd
        self.timestep_index += 1
        return result


@dataclass(frozen=True, slots=True)
class RawWorldStepResult:
    """One timestep in the ghost world where the battery does not exist."""

    timestep_index: int
    grid_import_kw: float
    meter: ElectricityMeterStepResult
    cost: OperatingCostStepResult


@dataclass(slots=True)
class RawWorld:
    """No-battery ghost universe: raw factory net load -> meter -> shared accountant."""

    timestep_hours: float
    demand_charge_vnd_per_kw: float
    meter_state: ElectricityMeterState = field(
        default_factory=ElectricityMeterState,
        init=False,
    )
    timestep_index: int = field(default=0, init=False)
    total_electricity_energy_cost_vnd: float = field(default=0.0, init=False)
    total_demand_cost_vnd: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.demand_charge_vnd_per_kw)):
            raise ValueError("demand_charge_vnd_per_kw must be finite")
        if self.demand_charge_vnd_per_kw < 0.0:
            raise ValueError("demand_charge_vnd_per_kw must not be negative")
        _validate_meter_timestep(self.timestep_hours)

    @property
    def total_operating_cost_vnd(self) -> float:
        return self.total_electricity_energy_cost_vnd + self.total_demand_cost_vnd

    def step(
        self,
        *,
        net_load_kw: float,
        tariff_vnd_per_kwh: float,
    ) -> RawWorldStepResult:
        """Advance the no-battery universe exactly once."""
        prepared_net_load_kw = float(net_load_kw)
        if not math.isfinite(prepared_net_load_kw):
            raise ValueError("net_load_kw must be finite")

        # Ghost-world physics has exactly one fact: without a battery, the grid sees
        # the non-negative prepared factory net load. There is no SOC or efficiency here.
        grid_import_kw = max(prepared_net_load_kw, 0.0)
        meter = run_electricity_meter_step(
            grid_import_kw=grid_import_kw,
            timestep_hours=self.timestep_hours,
            meter_state=self.meter_state,
        )
        cost = calculate_operating_cost_step(
            grid_energy_kwh=meter.sample_energy_kwh,
            battery_throughput_kwh=0.0,
            previous_monthly_peak_kw=meter.previous_monthly_peak_kw,
            monthly_peak_kw=meter.monthly_peak_kw,
            tariff_vnd_per_kwh=tariff_vnd_per_kwh,
            demand_charge_vnd_per_kw=self.demand_charge_vnd_per_kw,
            battery_wear_vnd_per_kwh=0.0,
        )

        result = RawWorldStepResult(
            timestep_index=self.timestep_index,
            grid_import_kw=grid_import_kw,
            meter=meter,
            cost=cost,
        )

        self.meter_state = meter.next_state
        self.total_electricity_energy_cost_vnd += cost.electricity_energy_cost_vnd
        self.total_demand_cost_vnd += cost.demand_cost_vnd
        self.timestep_index += 1
        return result


@dataclass(frozen=True, slots=True)
class BrainStepResult:
    """Bundled answer from the two synchronized universes for one factory timestep."""

    timestep_index: int
    bess: BessWorldStepResult
    raw: RawWorldStepResult
    electricity_energy_savings_vnd: float
    demand_savings_vnd: float
    battery_wear_cost_vnd: float
    net_battery_savings_vnd: float
    cumulative_electricity_energy_savings_vnd: float
    cumulative_demand_savings_vnd: float
    cumulative_battery_wear_cost_vnd: float
    cumulative_net_battery_savings_vnd: float


class BrainEnv:
    """Boss of the standalone environment: advance BESS and no-battery worlds together."""

    def __init__(
        self,
        *,
        initial_state_of_charge: float,
        minimum_state_of_charge: float,
        maximum_state_of_charge: float,
        battery_capacity_kwh: float,
        battery_power_kw: float,
        timestep_hours: float,
        charge_efficiency: float,
        discharge_efficiency: float,
        demand_charge_vnd_per_kw: float,
        battery_wear_vnd_per_kwh: float,
    ) -> None:
        self.bess_world = BessWorld(
            state_of_charge=initial_state_of_charge,
            minimum_state_of_charge=minimum_state_of_charge,
            maximum_state_of_charge=maximum_state_of_charge,
            battery_capacity_kwh=battery_capacity_kwh,
            battery_power_kw=battery_power_kw,
            timestep_hours=timestep_hours,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
            demand_charge_vnd_per_kw=demand_charge_vnd_per_kw,
            battery_wear_vnd_per_kwh=battery_wear_vnd_per_kwh,
        )
        self.raw_world = RawWorld(
            timestep_hours=timestep_hours,
            demand_charge_vnd_per_kw=demand_charge_vnd_per_kw,
        )

    @property
    def electricity_energy_savings_vnd(self) -> float:
        return (
            self.raw_world.total_electricity_energy_cost_vnd
            - self.bess_world.total_electricity_energy_cost_vnd
        )

    @property
    def demand_savings_vnd(self) -> float:
        return self.raw_world.total_demand_cost_vnd - self.bess_world.total_demand_cost_vnd

    @property
    def battery_wear_cost_vnd(self) -> float:
        return self.bess_world.total_battery_wear_cost_vnd

    @property
    def net_battery_savings_vnd(self) -> float:
        return self.raw_world.total_operating_cost_vnd - self.bess_world.total_operating_cost_vnd

    def step(
        self,
        *,
        action: float,
        net_load_kw: float,
        tariff_vnd_per_kwh: float,
    ) -> BrainStepResult:
        """Advance both worlds once and report exactly what the battery changed economically."""
        if self.bess_world.timestep_index != self.raw_world.timestep_index:
            raise RuntimeError("BessWorld and RawWorld timestep indexes must stay in lockstep")

        action_value = float(action)
        net_load_value = float(net_load_kw)
        tariff_value = float(tariff_vnd_per_kwh)
        if not all(math.isfinite(value) for value in (action_value, net_load_value, tariff_value)):
            raise ValueError("BrainEnv step inputs must all be finite")
        if tariff_value < 0.0:
            raise ValueError("tariff_vnd_per_kwh must not be negative")

        timestep_index = self.bess_world.timestep_index

        # BESS goes first. With the shared inputs validated and RawWorld configuration
        # fixed at construction, any BESS failure happens before the ghost world mutates.
        bess_snapshot = (
            self.bess_world.state_of_charge,
            self.bess_world.meter_state,
            self.bess_world.timestep_index,
            self.bess_world.total_electricity_energy_cost_vnd,
            self.bess_world.total_demand_cost_vnd,
            self.bess_world.total_battery_wear_cost_vnd,
        )
        bess_result = self.bess_world.step(
            action=action_value,
            net_load_kw=net_load_value,
            tariff_vnd_per_kwh=tariff_value,
        )
        try:
            raw_result = self.raw_world.step(
                net_load_kw=net_load_value,
                tariff_vnd_per_kwh=tariff_value,
            )
        except Exception:
            (
                self.bess_world.state_of_charge,
                self.bess_world.meter_state,
                self.bess_world.timestep_index,
                self.bess_world.total_electricity_energy_cost_vnd,
                self.bess_world.total_demand_cost_vnd,
                self.bess_world.total_battery_wear_cost_vnd,
            ) = bess_snapshot
            raise

        if self.bess_world.timestep_index != self.raw_world.timestep_index:
            raise RuntimeError("BessWorld and RawWorld advanced out of lockstep")
        if (
            raw_result.cost.battery_throughput_kwh != 0.0
            or raw_result.cost.battery_wear_cost_vnd != 0.0
        ):
            raise RuntimeError("RawWorld must never contain battery throughput or battery wear")

        electricity_energy_savings_vnd = (
            raw_result.cost.electricity_energy_cost_vnd
            - bess_result.cost.electricity_energy_cost_vnd
        )
        demand_savings_vnd = raw_result.cost.demand_cost_vnd - bess_result.cost.demand_cost_vnd
        battery_wear_cost_vnd = bess_result.cost.battery_wear_cost_vnd
        net_battery_savings_vnd = (
            raw_result.cost.operating_cost_vnd - bess_result.cost.operating_cost_vnd
        )

        reconstructed_net_savings_vnd = (
            electricity_energy_savings_vnd + demand_savings_vnd - battery_wear_cost_vnd
        )
        if not math.isclose(
            net_battery_savings_vnd,
            reconstructed_net_savings_vnd,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise RuntimeError("Battery savings accounting invariant failed")

        cumulative_net_savings_vnd = self.net_battery_savings_vnd
        reconstructed_cumulative_net_savings_vnd = (
            self.electricity_energy_savings_vnd
            + self.demand_savings_vnd
            - self.battery_wear_cost_vnd
        )
        if not math.isclose(
            cumulative_net_savings_vnd,
            reconstructed_cumulative_net_savings_vnd,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise RuntimeError("Cumulative battery savings accounting invariant failed")

        return BrainStepResult(
            timestep_index=timestep_index,
            bess=bess_result,
            raw=raw_result,
            electricity_energy_savings_vnd=electricity_energy_savings_vnd,
            demand_savings_vnd=demand_savings_vnd,
            battery_wear_cost_vnd=battery_wear_cost_vnd,
            net_battery_savings_vnd=net_battery_savings_vnd,
            cumulative_electricity_energy_savings_vnd=self.electricity_energy_savings_vnd,
            cumulative_demand_savings_vnd=self.demand_savings_vnd,
            cumulative_battery_wear_cost_vnd=self.battery_wear_cost_vnd,
            cumulative_net_battery_savings_vnd=cumulative_net_savings_vnd,
        )


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
