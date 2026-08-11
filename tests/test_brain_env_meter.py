import pytest

from bess.brain.brain_env import (
    ElectricityMeterState,
    ElectricityMeterStepResult,
    reset_electricity_meter_for_new_day,
    reset_electricity_meter_for_new_month,
    run_electricity_meter_step,
    run_physics_step,
)


def meter_step(
    grid_import_kw: float,
    state: ElectricityMeterState | None = None,
    timestep_hours: float = 0.25,
) -> ElectricityMeterStepResult:
    return run_electricity_meter_step(
        grid_import_kw=grid_import_kw,
        timestep_hours=timestep_hours,
        meter_state=state or ElectricityMeterState(),
    )


def test_first_15_minute_sample_does_not_create_demand_peak():
    result = meter_step(800.0)

    assert isinstance(result, ElectricityMeterStepResult)
    assert result.sample_energy_kwh == pytest.approx(200.0)
    assert result.accumulated_block_energy_kwh == pytest.approx(200.0)
    assert result.accumulated_block_elapsed_hours == pytest.approx(0.25)
    assert result.block_completed is False
    assert result.completed_block_demand_kw is None
    assert result.monthly_peak_kw == pytest.approx(0.0)
    assert result.new_monthly_peak is False


def test_second_15_minute_sample_completes_correct_30_minute_block():
    first = meter_step(800.0)
    second = meter_step(1000.0, first.next_state)

    assert second.accumulated_block_energy_kwh == pytest.approx(450.0)
    assert second.accumulated_block_elapsed_hours == pytest.approx(0.5)
    assert second.block_completed is True
    assert second.completed_block_demand_kw == pytest.approx(900.0)
    assert second.previous_monthly_peak_kw == pytest.approx(0.0)
    assert second.monthly_peak_kw == pytest.approx(900.0)
    assert second.new_monthly_peak is True
    assert second.next_state.block_energy_kwh == pytest.approx(0.0)
    assert second.next_state.block_elapsed_hours == pytest.approx(0.0)
    assert second.next_state.monthly_peak_kw == pytest.approx(900.0)


def test_instantaneous_spike_does_not_become_monthly_peak():
    first = meter_step(1500.0)
    assert first.monthly_peak_kw == pytest.approx(0.0)

    second = meter_step(500.0, first.next_state)
    assert second.completed_block_demand_kw == pytest.approx(1000.0)
    assert second.monthly_peak_kw == pytest.approx(1000.0)
    assert second.monthly_peak_kw != pytest.approx(1500.0)


def test_lower_completed_block_keeps_existing_monthly_peak():
    first = meter_step(800.0)
    first_block = meter_step(1000.0, first.next_state)
    assert first_block.monthly_peak_kw == pytest.approx(900.0)

    third = meter_step(700.0, first_block.next_state)
    second_block = meter_step(900.0, third.next_state)

    assert second_block.completed_block_demand_kw == pytest.approx(800.0)
    assert second_block.previous_monthly_peak_kw == pytest.approx(900.0)
    assert second_block.monthly_peak_kw == pytest.approx(900.0)
    assert second_block.new_monthly_peak is False


def test_higher_completed_block_sets_new_monthly_peak():
    first = meter_step(800.0)
    first_block = meter_step(1000.0, first.next_state)

    third = meter_step(1000.0, first_block.next_state)
    second_block = meter_step(1200.0, third.next_state)

    assert second_block.completed_block_demand_kw == pytest.approx(1100.0)
    assert second_block.previous_monthly_peak_kw == pytest.approx(900.0)
    assert second_block.monthly_peak_kw == pytest.approx(1100.0)
    assert second_block.new_monthly_peak is True


def test_thirty_one_minute_samples_make_one_fixed_block():
    state = ElectricityMeterState()
    result = None

    for _ in range(30):
        result = meter_step(600.0, state, timestep_hours=1.0 / 60.0)
        state = result.next_state

    assert result is not None
    assert result.block_completed is True
    assert result.completed_block_demand_kw == pytest.approx(600.0)
    assert result.monthly_peak_kw == pytest.approx(600.0)
    assert state.block_elapsed_hours == pytest.approx(0.0)


def test_zero_import_is_a_real_completed_zero_demand_block():
    result = meter_step(0.0, timestep_hours=0.5)

    assert result.block_completed is True
    assert result.completed_block_demand_kw == pytest.approx(0.0)
    assert result.monthly_peak_kw == pytest.approx(0.0)
    assert result.new_monthly_peak is False


def test_exact_30_minute_timestep_completes_every_sample():
    first = meter_step(300.0, timestep_hours=0.5)
    second = meter_step(500.0, first.next_state, timestep_hours=0.5)

    assert first.block_completed is True
    assert first.completed_block_demand_kw == pytest.approx(300.0)
    assert first.monthly_peak_kw == pytest.approx(300.0)
    assert second.block_completed is True
    assert second.completed_block_demand_kw == pytest.approx(500.0)
    assert second.monthly_peak_kw == pytest.approx(500.0)


def test_timestep_must_tile_30_minutes_exactly():
    with pytest.raises(ValueError, match="divide a fixed 30-minute demand block exactly"):
        meter_step(300.0, timestep_hours=7.0 / 60.0)


def test_new_day_clears_open_block_but_keeps_monthly_peak():
    state = ElectricityMeterState(
        block_energy_kwh=123.0,
        block_elapsed_hours=0.25,
        monthly_peak_kw=900.0,
    )

    reset = reset_electricity_meter_for_new_day(state)

    assert reset.block_energy_kwh == pytest.approx(0.0)
    assert reset.block_elapsed_hours == pytest.approx(0.0)
    assert reset.monthly_peak_kw == pytest.approx(900.0)


def test_new_month_resets_everything():
    reset = reset_electricity_meter_for_new_month()

    assert reset == ElectricityMeterState()


def test_physics_grid_import_can_feed_meter_without_hidden_coupling():
    physics = run_physics_step(
        action=0.0,
        net_load_kw=800.0,
        state_of_charge=0.5,
        minimum_state_of_charge=0.1,
        maximum_state_of_charge=0.9,
        battery_capacity_kwh=1000.0,
        battery_power_kw=450.0,
        timestep_hours=0.25,
        charge_efficiency=0.9,
        discharge_efficiency=0.9,
    )

    meter = meter_step(physics.grid_import_kw)

    assert physics.grid_import_kw == pytest.approx(800.0)
    assert meter.grid_import_kw == pytest.approx(800.0)
    assert meter.sample_energy_kwh == pytest.approx(200.0)
