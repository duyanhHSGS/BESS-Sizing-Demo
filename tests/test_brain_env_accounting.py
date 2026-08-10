import pytest

from EXPERIMENT_FIELD.brain_env import (
    ElectricityMeterState,
    MonthlyOperatingCostResult,
    OperatingCostStepResult,
    calculate_month_operating_cost_from_history,
    calculate_operating_cost_step,
    run_electricity_meter_step,
)


def test_step_accountant_prices_only_grid_energy_peak_delta_and_battery_throughput():
    result = calculate_operating_cost_step(
        grid_energy_kwh=200.0,
        battery_throughput_kwh=25.0,
        previous_monthly_peak_kw=800.0,
        monthly_peak_kw=850.0,
        tariff_vnd_per_kwh=2_000.0,
        demand_charge_vnd_per_kw=100_000.0,
        battery_wear_vnd_per_kwh=50.0,
    )

    assert isinstance(result, OperatingCostStepResult)
    assert result.electricity_energy_cost_vnd == pytest.approx(400_000.0)
    assert result.demand_peak_increase_kw == pytest.approx(50.0)
    assert result.demand_cost_vnd == pytest.approx(5_000_000.0)
    assert result.battery_wear_cost_vnd == pytest.approx(1_250.0)
    assert result.operating_cost_vnd == pytest.approx(5_401_250.0)


def test_no_new_peak_means_zero_demand_cost():
    result = calculate_operating_cost_step(
        grid_energy_kwh=10.0,
        battery_throughput_kwh=0.0,
        previous_monthly_peak_kw=900.0,
        monthly_peak_kw=900.0,
        tariff_vnd_per_kwh=1_000.0,
        demand_charge_vnd_per_kw=123_456.0,
        battery_wear_vnd_per_kwh=50.0,
    )

    assert result.demand_peak_increase_kw == pytest.approx(0.0)
    assert result.demand_cost_vnd == pytest.approx(0.0)
    assert result.operating_cost_vnd == pytest.approx(10_000.0)


def test_battery_wear_counts_charge_plus_discharge_battery_side_without_efficiency():
    # 200 kW charge for 15 min = 50 kWh moved into battery.
    # 2000 kW discharge for 15 min = 500 kWh moved out.
    # Wear sees 50 + 500 = 550 kWh. Nothing else.
    month = calculate_month_operating_cost_from_history(
        grid_import_kw=[0.0, 0.0],
        battery_power_kw=[-200.0, 2_000.0],
        tariff_vnd_per_kwh=[0.0, 0.0],
        timestep_hours=0.25,
        demand_charge_vnd_per_kw=0.0,
        battery_wear_vnd_per_kwh=50.0,
    )

    assert month.battery_throughput_kwh == pytest.approx(550.0)
    assert month.battery_wear_cost_vnd == pytest.approx(27_500.0)
    assert month.operating_cost_vnd == pytest.approx(27_500.0)


def test_month_lookback_uses_grid_tariff_battery_absolute_power_and_highest_30m_peak():
    result = calculate_month_operating_cost_from_history(
        grid_import_kw=[800.0, 1_000.0, 700.0, 900.0],
        battery_power_kw=[-100.0, 200.0, -300.0, 0.0],
        tariff_vnd_per_kwh=[1.0, 2.0, 3.0, 4.0],
        timestep_hours=0.25,
        demand_charge_vnd_per_kw=10.0,
        battery_wear_vnd_per_kwh=5.0,
    )

    assert isinstance(result, MonthlyOperatingCostResult)
    assert result.total_grid_energy_kwh == pytest.approx(850.0)
    assert result.electricity_energy_cost_vnd == pytest.approx(2_125.0)
    assert result.battery_throughput_kwh == pytest.approx(150.0)
    assert result.battery_wear_cost_vnd == pytest.approx(750.0)
    assert result.monthly_peak_kw == pytest.approx(900.0)
    assert result.demand_cost_vnd == pytest.approx(9_000.0)
    assert result.operating_cost_vnd == pytest.approx(11_875.0)


def test_step_cost_sum_matches_independent_month_lookback():
    grid_history = [800.0, 1_000.0, 700.0, 900.0]
    battery_history = [-100.0, 200.0, -300.0, 0.0]
    tariff_history = [1.0, 2.0, 3.0, 4.0]
    dt = 0.25
    demand_fee = 10.0
    wear_rate = 5.0

    meter_state = ElectricityMeterState()
    step_energy_cost = 0.0
    step_demand_cost = 0.0
    step_wear_cost = 0.0
    step_operating_cost = 0.0

    for grid_kw, battery_kw, tariff in zip(grid_history, battery_history, tariff_history):
        meter = run_electricity_meter_step(
            grid_import_kw=grid_kw,
            timestep_hours=dt,
            meter_state=meter_state,
        )
        meter_state = meter.next_state

        cost = calculate_operating_cost_step(
            grid_energy_kwh=meter.sample_energy_kwh,
            battery_throughput_kwh=abs(battery_kw) * dt,
            previous_monthly_peak_kw=meter.previous_monthly_peak_kw,
            monthly_peak_kw=meter.monthly_peak_kw,
            tariff_vnd_per_kwh=tariff,
            demand_charge_vnd_per_kw=demand_fee,
            battery_wear_vnd_per_kwh=wear_rate,
        )
        step_energy_cost += cost.electricity_energy_cost_vnd
        step_demand_cost += cost.demand_cost_vnd
        step_wear_cost += cost.battery_wear_cost_vnd
        step_operating_cost += cost.operating_cost_vnd

    month = calculate_month_operating_cost_from_history(
        grid_import_kw=grid_history,
        battery_power_kw=battery_history,
        tariff_vnd_per_kwh=tariff_history,
        timestep_hours=dt,
        demand_charge_vnd_per_kw=demand_fee,
        battery_wear_vnd_per_kwh=wear_rate,
    )

    assert step_energy_cost == pytest.approx(month.electricity_energy_cost_vnd)
    assert step_demand_cost == pytest.approx(month.demand_cost_vnd)
    assert step_wear_cost == pytest.approx(month.battery_wear_cost_vnd)
    assert step_operating_cost == pytest.approx(month.operating_cost_vnd)


def test_accountant_rejects_peak_that_goes_backwards():
    with pytest.raises(ValueError, match="must not go backwards"):
        calculate_operating_cost_step(
            grid_energy_kwh=1.0,
            battery_throughput_kwh=1.0,
            previous_monthly_peak_kw=900.0,
            monthly_peak_kw=899.0,
            tariff_vnd_per_kwh=1.0,
            demand_charge_vnd_per_kw=1.0,
            battery_wear_vnd_per_kwh=1.0,
        )


def test_complete_month_lookback_requires_fixed_block_boundary():
    with pytest.raises(ValueError, match="30-minute demand-block boundary"):
        calculate_month_operating_cost_from_history(
            grid_import_kw=[100.0],
            battery_power_kw=[0.0],
            tariff_vnd_per_kwh=[1.0],
            timestep_hours=0.25,
            demand_charge_vnd_per_kw=1.0,
            battery_wear_vnd_per_kwh=1.0,
        )
