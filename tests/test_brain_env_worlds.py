import pytest

from EXPERIMENT_FIELD.brain_env import (
    BrainEnv,
    BrainStepResult,
    ElectricityMeterState,
    RawWorld,
)


BASE_ENV = {
    "initial_state_of_charge": 0.80,
    "minimum_state_of_charge": 0.10,
    "maximum_state_of_charge": 0.90,
    "battery_capacity_kwh": 1000.0,
    "battery_power_kw": 100.0,
    "timestep_hours": 0.50,
    "charge_efficiency": 1.0,
    "discharge_efficiency": 1.0,
    "demand_charge_vnd_per_kw": 100.0,
    "battery_wear_vnd_per_kwh": 1.0,
}


def make_env(**overrides) -> BrainEnv:
    return BrainEnv(**{**BASE_ENV, **overrides})


def test_helpful_battery_world_beats_raw_world_and_reports_exact_savings():
    env = make_env()

    result = env.step(
        action=1.0,
        net_load_kw=200.0,
        tariff_vnd_per_kwh=10.0,
    )

    assert isinstance(result, BrainStepResult)
    assert result.timestep_index == 0

    # Same factory, two universes: the battery supplies 100 kW only in BessWorld.
    assert result.bess.physics.grid_import_kw == pytest.approx(100.0)
    assert result.raw.grid_import_kw == pytest.approx(200.0)

    # Exact 30-minute timestep means each universe completes its own demand block.
    assert result.bess.meter.monthly_peak_kw == pytest.approx(100.0)
    assert result.raw.meter.monthly_peak_kw == pytest.approx(200.0)

    # Energy savings: (100 kWh raw - 50 kWh BESS) * 10 VND/kWh = 500 VND.
    assert result.electricity_energy_savings_vnd == pytest.approx(500.0)
    # Demand savings: (200 - 100) kW * 100 VND/kW = 10,000 VND.
    assert result.demand_savings_vnd == pytest.approx(10_000.0)
    # Battery moved 100 kW * 0.5 h = 50 kWh, at 1 VND/kWh wear.
    assert result.battery_wear_cost_vnd == pytest.approx(50.0)
    assert result.net_battery_savings_vnd == pytest.approx(10_450.0)

    assert result.raw.cost.battery_throughput_kwh == pytest.approx(0.0)
    assert result.raw.cost.battery_wear_cost_vnd == pytest.approx(0.0)
    assert env.bess_world.timestep_index == 1
    assert env.raw_world.timestep_index == 1


def test_idle_battery_is_exact_boundary_where_bess_and_raw_worlds_match():
    env = make_env()

    result = env.step(
        action=0.0,
        net_load_kw=200.0,
        tariff_vnd_per_kwh=10.0,
    )

    assert result.bess.physics.grid_import_kw == pytest.approx(result.raw.grid_import_kw)
    assert result.electricity_energy_savings_vnd == pytest.approx(0.0)
    assert result.demand_savings_vnd == pytest.approx(0.0)
    assert result.battery_wear_cost_vnd == pytest.approx(0.0)
    assert result.net_battery_savings_vnd == pytest.approx(0.0)
    assert result.cumulative_net_battery_savings_vnd == pytest.approx(0.0)


def test_raw_world_has_no_battery_and_clamps_negative_factory_net_load_to_zero_grid_import():
    raw = RawWorld(
        timestep_hours=0.5,
        demand_charge_vnd_per_kw=100.0,
    )

    result = raw.step(
        net_load_kw=-50.0,
        tariff_vnd_per_kwh=10.0,
    )

    assert result.grid_import_kw == pytest.approx(0.0)
    assert result.cost.grid_energy_kwh == pytest.approx(0.0)
    assert result.cost.battery_throughput_kwh == pytest.approx(0.0)
    assert result.cost.battery_wear_cost_vnd == pytest.approx(0.0)
    assert result.cost.operating_cost_vnd == pytest.approx(0.0)


def test_stupid_expensive_charging_can_make_battery_net_savings_negative():
    env = make_env(demand_charge_vnd_per_kw=0.0)

    result = env.step(
        action=-1.0,
        net_load_kw=100.0,
        tariff_vnd_per_kwh=10.0,
    )

    assert result.bess.physics.grid_import_kw == pytest.approx(200.0)
    assert result.raw.grid_import_kw == pytest.approx(100.0)
    assert result.electricity_energy_savings_vnd == pytest.approx(-500.0)
    assert result.demand_savings_vnd == pytest.approx(0.0)
    assert result.battery_wear_cost_vnd == pytest.approx(50.0)
    assert result.net_battery_savings_vnd == pytest.approx(-550.0)


def test_invalid_step_fails_before_either_world_advances_or_accumulates_money():
    env = make_env()

    with pytest.raises(ValueError, match="must all be finite"):
        env.step(
            action=0.0,
            net_load_kw=200.0,
            tariff_vnd_per_kwh=float("nan"),
        )

    assert env.bess_world.timestep_index == 0
    assert env.raw_world.timestep_index == 0
    assert env.bess_world.total_operating_cost_vnd == pytest.approx(0.0)
    assert env.raw_world.total_operating_cost_vnd == pytest.approx(0.0)


def test_worlds_must_never_advance_out_of_lockstep():
    env = make_env()
    env.raw_world.timestep_index = 1

    with pytest.raises(RuntimeError, match="must stay in lockstep"):
        env.step(
            action=0.0,
            net_load_kw=200.0,
            tariff_vnd_per_kwh=10.0,
        )

    assert env.bess_world.timestep_index == 0
    assert env.raw_world.timestep_index == 1


def test_cumulative_savings_have_two_equivalent_accounting_truths():
    env = make_env(demand_charge_vnd_per_kw=0.0)

    env.step(action=1.0, net_load_kw=200.0, tariff_vnd_per_kwh=10.0)
    final = env.step(action=-1.0, net_load_kw=100.0, tariff_vnd_per_kwh=2.0)

    direct_difference = (
        env.raw_world.total_operating_cost_vnd
        - env.bess_world.total_operating_cost_vnd
    )
    component_difference = (
        env.electricity_energy_savings_vnd
        + env.demand_savings_vnd
        - env.battery_wear_cost_vnd
    )

    assert env.net_battery_savings_vnd == pytest.approx(direct_difference)
    assert env.net_battery_savings_vnd == pytest.approx(component_difference)
    assert final.cumulative_net_battery_savings_vnd == pytest.approx(env.net_battery_savings_vnd)
    assert final.cumulative_electricity_energy_savings_vnd == pytest.approx(
        env.electricity_energy_savings_vnd
    )
    assert final.cumulative_demand_savings_vnd == pytest.approx(env.demand_savings_vnd)
    assert final.cumulative_battery_wear_cost_vnd == pytest.approx(env.battery_wear_cost_vnd)


def test_raw_world_failure_rolls_bess_world_back_instead_of_splitting_the_universes():
    env = make_env()
    starting_soc = env.bess_world.state_of_charge
    env.raw_world.meter_state = ElectricityMeterState(
        block_energy_kwh=1.0,
        block_elapsed_hours=0.0,
        monthly_peak_kw=0.0,
    )

    with pytest.raises(ValueError, match="empty meter block cannot already contain energy"):
        env.step(
            action=1.0,
            net_load_kw=200.0,
            tariff_vnd_per_kwh=10.0,
        )

    assert env.bess_world.state_of_charge == pytest.approx(starting_soc)
    assert env.bess_world.timestep_index == 0
    assert env.raw_world.timestep_index == 0
    assert env.bess_world.total_operating_cost_vnd == pytest.approx(0.0)
    assert env.raw_world.total_operating_cost_vnd == pytest.approx(0.0)


def test_world_construction_rejects_invalid_strict_configuration():
    with pytest.raises(ValueError, match="battery_power_kw must be greater than 0"):
        make_env(battery_power_kw=0.0)

    with pytest.raises(ValueError, match="demand_charge_vnd_per_kw must not be negative"):
        RawWorld(
            timestep_hours=0.5,
            demand_charge_vnd_per_kw=-1.0,
        )
