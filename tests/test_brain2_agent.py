import ast
import math
from pathlib import Path

import pytest

from EXPERIMENT_FIELD.brain2_agent import Brain2Agent, Brain2Decision
from EXPERIMENT_FIELD.brain_env import BrainEnv, BrainEpisode, BrainTimestepInput


BASE_AGENT = {
    "battery_capacity_kwh": 1250.0,
    "battery_power_kw": 450.0,
    "minimum_state_of_charge": 0.20,
    "maximum_state_of_charge": 0.90,
    "timestep_minutes": 15.0,
    "cheap_tariff_vnd_per_kwh": 904.0,
    "normal_tariff_vnd_per_kwh": 1332.0,
    "expensive_tariff_vnd_per_kwh": 2251.0,
    "cheap_start_minute": 0.0,
    "cheap_end_minute": 360.0,
    "expensive_start_minute": 1050.0,
    "expensive_end_minute": 1350.0,
}


def make_agent(**overrides) -> Brain2Agent:
    return Brain2Agent(**{**BASE_AGENT, **overrides})


def observation_at(
    minute: float,
    normalized_soc: float,
    *,
    normalized_tariff: float = 0.5,
    normalized_net_load: float = 0.3,
    normalized_peak: float = 0.2,
    working_day: float = 1.0,
):
    angle = 2.0 * math.pi * minute / 1440.0
    return (
        math.sin(angle),
        math.cos(angle),
        normalized_net_load,
        normalized_soc,
        normalized_tariff,
        normalized_peak,
        working_day,
    )


def test_user_example_initial_cheap_charge_is_exact_875_kwh_over_24_steps():
    agent = make_agent()

    decision = agent.decide(observation_at(0.0, 0.0))

    assert isinstance(decision, Brain2Decision)
    assert decision.tariff_period == "cheap"
    assert decision.remaining_cheap_steps == 24
    assert decision.target_battery_energy_kwh == pytest.approx(875.0)
    assert decision.requested_battery_power_kw == pytest.approx(-145.83333333333334)
    assert decision.action == pytest.approx(-145.83333333333334 / 450.0)
    assert decision.reason_code == "adaptive_cheap_fill"


def test_cheap_charge_recalculates_from_actual_soc_and_remaining_steps():
    agent = make_agent()

    on_plan = agent.decide(observation_at(180.0, 0.50))
    behind_plan = agent.decide(observation_at(180.0, 0.40))

    assert on_plan.remaining_cheap_steps == 12
    assert on_plan.action == pytest.approx(-145.83333333333334 / 450.0)
    assert behind_plan.action == pytest.approx(-175.0 / 450.0)
    assert behind_plan.action < on_plan.action


def test_last_cheap_step_requests_exact_remaining_energy_only():
    agent = make_agent()

    decision = agent.decide(observation_at(345.0, 0.98))

    assert decision.remaining_cheap_steps == 1
    assert decision.target_battery_energy_kwh == pytest.approx(17.5)
    assert decision.requested_battery_power_kw == pytest.approx(-70.0)
    assert decision.action == pytest.approx(-70.0 / 450.0)


def test_cheap_but_already_full_idles():
    decision = make_agent().decide(observation_at(120.0, 1.0))

    assert decision.action == 0.0
    assert decision.label == "IDLE"
    assert decision.reason_code == "cheap_but_full"


def test_weighted_discharge_actions_match_user_tariff_ratio_and_energy_budget():
    agent = make_agent()

    assert agent.discharge_normal_steps == 52
    assert agent.discharge_expensive_steps == 20
    assert agent.normal_action == pytest.approx(0.09065135977039655)
    assert agent.expensive_action == pytest.approx(0.15319535348585783)
    assert agent.expensive_action / agent.normal_action == pytest.approx(2251.0 / 1332.0)

    planned_energy_kwh = 450.0 * 0.25 * (
        52 * agent.normal_action + 20 * agent.expensive_action
    )
    assert planned_energy_kwh == pytest.approx(875.0)


def test_normal_window_uses_constant_normal_action():
    agent = make_agent()

    morning = agent.decide(observation_at(360.0, 1.0))
    afternoon = agent.decide(observation_at(900.0, 0.7))
    late = agent.decide(observation_at(1350.0, 0.2))

    assert morning.action == pytest.approx(agent.normal_action)
    assert afternoon.action == pytest.approx(agent.normal_action)
    assert late.action == pytest.approx(agent.normal_action)
    assert morning.reason_code == "weighted_normal_discharge"


def test_expensive_window_uses_larger_constant_expensive_action():
    agent = make_agent()

    first = agent.decide(observation_at(1050.0, 0.8))
    middle = agent.decide(observation_at(1200.0, 0.5))
    last = agent.decide(observation_at(1335.0, 0.1))

    assert first.action == pytest.approx(agent.expensive_action)
    assert middle.action == pytest.approx(agent.expensive_action)
    assert last.action == pytest.approx(agent.expensive_action)
    assert agent.expensive_action > agent.normal_action
    assert first.reason_code == "weighted_expensive_discharge"


def test_exact_tariff_window_boundaries_switch_periods_without_fuzzy_time():
    agent = make_agent()

    assert agent.decide(observation_at(0.0, 0.5)).tariff_period == "cheap"
    assert agent.decide(observation_at(345.0, 0.98)).tariff_period == "cheap"
    assert agent.decide(observation_at(360.0, 0.5)).tariff_period == "normal"
    assert agent.decide(observation_at(1050.0, 0.5)).tariff_period == "expensive"
    assert agent.decide(observation_at(1335.0, 0.5)).tariff_period == "expensive"
    assert agent.decide(observation_at(1350.0, 0.5)).tariff_period == "normal"


def test_noncheap_period_with_empty_usable_battery_idles():
    agent = make_agent()

    normal = agent.decide(observation_at(600.0, 0.0))
    expensive = agent.decide(observation_at(1200.0, 0.0))

    assert normal.action == 0.0
    assert normal.reason_code == "normal_but_empty"
    assert expensive.action == 0.0
    assert expensive.reason_code == "expensive_but_empty"


def test_schedule_rule_ignores_load_peak_working_day_and_tariff_eye_for_action():
    agent = make_agent()
    baseline = observation_at(1200.0, 0.5)
    weird_other_eyes = observation_at(
        1200.0,
        0.5,
        normalized_tariff=0.01,
        normalized_net_load=99.0,
        normalized_peak=77.0,
        working_day=0.0,
    )

    assert agent.act(baseline) == pytest.approx(agent.act(weird_other_eyes))


def test_late_cheap_state_that_cannot_reach_full_fails_loudly_instead_of_clamping():
    agent = make_agent()

    with pytest.raises(RuntimeError, match="cannot reach maximum SOC"):
        agent.act(observation_at(345.0, 0.0))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"battery_capacity_kwh": 0.0}, "capacity"),
        ({"battery_power_kw": 0.0}, "power"),
        ({"minimum_state_of_charge": 0.9, "maximum_state_of_charge": 0.2}, "SOC limits"),
        ({"timestep_minutes": 17.0}, "divide a 24-hour day"),
        ({"cheap_end_minute": 355.0}, "align exactly"),
        ({"cheap_start_minute": 300.0, "cheap_end_minute": 285.0}, "cheap window"),
        ({"expensive_start_minute": 300.0, "expensive_end_minute": 495.0}, "must not overlap"),
        ({"cheap_tariff_vnd_per_kwh": 1400.0}, "0 < cheap < normal < expensive"),
        ({"normal_tariff_vnd_per_kwh": 3000.0}, "0 < cheap < normal < expensive"),
        ({"battery_capacity_kwh": math.inf}, "finite"),
    ],
)
def test_invalid_configuration_fails_loudly(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_agent(**overrides)


def test_configuration_rejects_battery_that_cannot_fill_during_cheap_window():
    with pytest.raises(ValueError, match="cannot fill"):
        make_agent(battery_capacity_kwh=2000.0, battery_power_kw=100.0)


def test_configuration_rejects_weighting_that_would_require_action_above_one():
    with pytest.raises(ValueError, match="action above \+1"):
        make_agent(
            battery_capacity_kwh=1000.0,
            battery_power_kw=100.0,
            minimum_state_of_charge=0.0,
            maximum_state_of_charge=0.60,
            cheap_tariff_vnd_per_kwh=1.0,
            normal_tariff_vnd_per_kwh=2.0,
            expensive_tariff_vnd_per_kwh=200.0,
        )


@pytest.mark.parametrize(
    "observation",
    [
        (0.0,) * 6,
        (0.0,) * 8,
    ],
)
def test_wrong_observation_width_is_rejected(observation):
    with pytest.raises(ValueError, match="exactly 7"):
        make_agent().act(observation)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_observation_is_rejected(bad_value):
    observation = list(observation_at(600.0, 0.5))
    observation[2] = bad_value

    with pytest.raises(ValueError, match="finite"):
        make_agent().act(tuple(observation))


@pytest.mark.parametrize("bad_soc", [-0.0001, 1.0001])
def test_soc_eye_outside_unit_interval_is_rejected(bad_soc):
    with pytest.raises(ValueError, match="SOC eye"):
        make_agent().act(observation_at(600.0, bad_soc))


@pytest.mark.parametrize("bad_tariff", [-0.0001, 1.0001])
def test_tariff_eye_outside_unit_interval_is_rejected(bad_tariff):
    with pytest.raises(ValueError, match="tariff eye"):
        make_agent().act(observation_at(600.0, 0.5, normalized_tariff=bad_tariff))


def test_invalid_time_circle_is_rejected():
    observation = list(observation_at(600.0, 0.5))
    observation[0] = 0.0
    observation[1] = 0.0

    with pytest.raises(ValueError, match="unit circle"):
        make_agent().act(tuple(observation))


def test_time_not_on_native_timestep_is_rejected():
    with pytest.raises(ValueError, match="exact native timestep"):
        make_agent().act(observation_at(607.5, 0.5))


def test_action_is_always_inside_brain_env_action_bounds_for_valid_schedule_states():
    agent = make_agent()

    for minute in range(0, 1440, 15):
        for soc in (0.0, 0.25, 0.5, 0.75, 1.0):
            try:
                action = agent.act(observation_at(float(minute), soc))
            except RuntimeError:
                # A deliberately impossible late-cheap catch-up state must fail loudly,
                # never silently emit an illegal action.
                continue
            assert -1.0 <= action <= 1.0


def test_same_input_is_deterministic_observation_is_not_mutated_and_agent_is_frozen():
    agent = make_agent()
    observation = observation_at(1200.0, 0.5)
    before = tuple(observation)

    first = agent.decide(observation)
    second = agent.decide(observation)

    assert first == second
    assert observation == before
    with pytest.raises((AttributeError, TypeError)):
        agent.normal_action = 0.123


def test_brain2_imports_no_production_bess_or_training_stack():
    path = Path(__file__).resolve().parents[1] / "EXPERIMENT_FIELD" / "brain2_agent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert "EXPERIMENT_FIELD.brain_env" in imported_modules
    assert all(not module.startswith("bess") for module in imported_modules)
    assert all("training" not in module.lower() for module in imported_modules)
    assert all("ppo" not in module.lower() for module in imported_modules)


def test_full_day_unclipped_world_hits_full_at_cheap_end_and_minimum_at_midnight():
    agent = make_agent()
    timesteps = []
    for step in range(96):
        minute = step * 15
        if 0 <= minute < 360:
            tariff = 904.0
        elif 1050 <= minute < 1350:
            tariff = 2251.0
        else:
            tariff = 1332.0
        timesteps.append(
            BrainTimestepInput(
                net_load_kw=2000.0,
                tariff_vnd_per_kwh=tariff,
                is_working_day=True,
            )
        )

    env = BrainEnv(
        initial_state_of_charge=0.20,
        minimum_state_of_charge=0.20,
        maximum_state_of_charge=0.90,
        battery_capacity_kwh=1250.0,
        battery_power_kw=450.0,
        timestep_hours=0.25,
        charge_efficiency=0.90,
        discharge_efficiency=0.90,
        demand_charge_vnd_per_kw=0.0,
        battery_wear_vnd_per_kwh=0.0,
        episode=BrainEpisode(timesteps=tuple(timesteps), steps_per_day=96),
    )

    observation = env.reset()
    actions = []
    soc_after_cheap = None
    final = None
    for step in range(96):
        action = agent.act(observation)
        actions.append(action)
        final = env.step(action)
        if step == 23:
            soc_after_cheap = env.bess_world.state_of_charge
        if not final.done:
            observation = final.next_observation

    assert soc_after_cheap == pytest.approx(0.90, abs=1e-10)
    assert env.bess_world.state_of_charge == pytest.approx(0.20, abs=1e-10)
    assert all(action == pytest.approx(actions[0]) for action in actions[:24])
    assert all(action == pytest.approx(agent.expensive_action) for action in actions[70:90])
    assert actions[24] == pytest.approx(agent.normal_action)
    assert actions[90] == pytest.approx(agent.normal_action)
    assert final is not None and final.done is True
