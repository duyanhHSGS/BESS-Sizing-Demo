import pytest

from EXPERIMENT_FIELD.brain_env import PhysicsStepResult, run_physics_step


BASE = {
    "minimum_state_of_charge": 0.10,
    "maximum_state_of_charge": 0.90,
    "battery_capacity_kwh": 1000.0,
    "battery_power_kw": 450.0,
    "timestep_hours": 0.25,
    "charge_efficiency": 0.90,
    "discharge_efficiency": 0.90,
}


def run_step(**overrides) -> PhysicsStepResult:
    arguments = {
        **BASE,
        "action": 0.0,
        "net_load_kw": 300.0,
        "state_of_charge": 0.50,
        **overrides,
    }
    return run_physics_step(**arguments)


def test_idle_keeps_everything_boring():
    result = run_step(action=0.0)

    assert isinstance(result, PhysicsStepResult)
    assert result.requested_battery_kw == pytest.approx(0.0)
    assert result.battery_after_police_kw == pytest.approx(0.0)
    assert result.final_battery_kw == pytest.approx(0.0)
    assert result.battery_to_factory_kw == pytest.approx(0.0)
    assert result.grid_to_battery_kw == pytest.approx(0.0)
    assert result.conversion_loss_kw == pytest.approx(0.0)
    assert result.grid_import_kw == pytest.approx(300.0)
    assert result.battery_throughput_kwh == pytest.approx(0.0)
    assert result.starting_soc == pytest.approx(0.50)
    assert result.next_soc == pytest.approx(0.50)


def test_normal_discharge_tracks_every_kw():
    result = run_step(action=1.0, net_load_kw=1000.0)

    assert result.requested_battery_kw == pytest.approx(450.0)
    assert result.battery_after_police_kw == pytest.approx(450.0)
    assert result.final_battery_kw == pytest.approx(450.0)
    assert result.battery_to_factory_kw == pytest.approx(405.0)
    assert result.grid_to_battery_kw == pytest.approx(0.0)
    assert result.conversion_loss_kw == pytest.approx(45.0)
    assert result.grid_import_kw == pytest.approx(595.0)
    assert result.battery_throughput_kwh == pytest.approx(112.5)
    assert result.next_soc == pytest.approx(0.3875)


def test_normal_charge_tracks_every_kw():
    result = run_step(action=-1.0, net_load_kw=300.0)

    assert result.requested_battery_kw == pytest.approx(-450.0)
    assert result.battery_after_police_kw == pytest.approx(-450.0)
    assert result.final_battery_kw == pytest.approx(-450.0)
    assert result.battery_to_factory_kw == pytest.approx(0.0)
    assert result.grid_to_battery_kw == pytest.approx(500.0)
    assert result.conversion_loss_kw == pytest.approx(50.0)
    assert result.grid_import_kw == pytest.approx(800.0)
    assert result.battery_throughput_kwh == pytest.approx(112.5)
    assert result.next_soc == pytest.approx(0.6125)


def test_discharge_at_minimum_soc_is_stopped_by_battery_police():
    result = run_step(action=1.0, state_of_charge=0.10, net_load_kw=1000.0)

    assert result.requested_battery_kw == pytest.approx(450.0)
    assert result.battery_after_police_kw == pytest.approx(0.0)
    assert result.final_battery_kw == pytest.approx(0.0)
    assert result.battery_to_factory_kw == pytest.approx(0.0)
    assert result.grid_import_kw == pytest.approx(1000.0)
    assert result.next_soc == pytest.approx(0.10)


def test_tiny_load_clips_discharge_at_grid_guard_and_never_exports():
    result = run_step(action=1.0, net_load_kw=100.0)

    assert result.requested_battery_kw == pytest.approx(450.0)
    assert result.battery_after_police_kw == pytest.approx(450.0)
    assert result.final_battery_kw == pytest.approx(100.0 / 0.90)
    assert result.battery_to_factory_kw == pytest.approx(100.0)
    assert result.conversion_loss_kw == pytest.approx((100.0 / 0.90) - 100.0)
    assert result.grid_import_kw == pytest.approx(0.0)
    assert result.next_soc == pytest.approx(0.50 - ((100.0 / 0.90) * 0.25 / 1000.0))


def test_charge_near_maximum_soc_lands_exactly_on_soc_wall():
    result = run_step(action=-1.0, state_of_charge=0.89)

    assert result.requested_battery_kw == pytest.approx(-450.0)
    assert result.battery_after_police_kw == pytest.approx(-40.0)
    assert result.final_battery_kw == pytest.approx(-40.0)
    assert result.grid_to_battery_kw == pytest.approx(40.0 / 0.90)
    assert result.conversion_loss_kw == pytest.approx((40.0 / 0.90) - 40.0)
    assert result.next_soc == pytest.approx(0.90)


def test_discharge_near_minimum_soc_lands_exactly_on_soc_wall():
    result = run_step(action=1.0, state_of_charge=0.11, net_load_kw=1000.0)

    assert result.requested_battery_kw == pytest.approx(450.0)
    assert result.battery_after_police_kw == pytest.approx(40.0)
    assert result.final_battery_kw == pytest.approx(40.0)
    assert result.battery_to_factory_kw == pytest.approx(36.0)
    assert result.conversion_loss_kw == pytest.approx(4.0)
    assert result.next_soc == pytest.approx(0.10)
