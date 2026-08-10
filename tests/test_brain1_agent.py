import ast
from pathlib import Path

import pytest

from EXPERIMENT_FIELD.brain1_agent import Brain1Agent
from EXPERIMENT_FIELD.brain_env import BrainEnv, BrainEpisode, BrainTimestepInput


BASE_OBSERVATION = (0.0, 1.0, 0.2, 0.5, 0.5, 0.1, 1.0)


def make_agent() -> Brain1Agent:
    return Brain1Agent(
        cheap_tariff_max_normalized=0.10,
        expensive_tariff_min_normalized=0.80,
    )


def with_eyes(*, soc: float, tariff: float, base=BASE_OBSERVATION):
    values = list(base)
    values[3] = soc
    values[4] = tariff
    return tuple(values)


# ---------------------------------------------------------------------------
# Normal cases: the tiny school answer sheet.
# ---------------------------------------------------------------------------


def test_cheap_with_room_requests_full_charge():
    assert make_agent().act(with_eyes(soc=0.50, tariff=0.05)) == -1.0


def test_normal_tariff_idles():
    assert make_agent().act(with_eyes(soc=0.50, tariff=0.50)) == 0.0


def test_expensive_with_energy_requests_full_discharge():
    assert make_agent().act(with_eyes(soc=0.50, tariff=0.90)) == 1.0


def test_irrelevant_five_eyes_do_not_change_brain1_decision():
    agent = make_agent()
    boring = (0.0, 1.0, 0.2, 0.50, 0.05, 0.1, 1.0)
    wildly_different = (-0.99, -0.01, 123.0, 0.50, 0.05, 456.0, 0.0)

    assert agent.act(boring) == -1.0
    assert agent.act(wildly_different) == -1.0


# ---------------------------------------------------------------------------
# Boundaries: equality rules are deliberate, not accidental.
# ---------------------------------------------------------------------------


def test_cheap_at_exact_threshold_charges():
    assert make_agent().act(with_eyes(soc=0.50, tariff=0.10)) == -1.0


def test_expensive_at_exact_threshold_discharges():
    assert make_agent().act(with_eyes(soc=0.50, tariff=0.80)) == 1.0


def test_cheap_but_exactly_full_idles():
    assert make_agent().act(with_eyes(soc=1.0, tariff=0.05)) == 0.0


def test_expensive_but_exactly_empty_idles():
    assert make_agent().act(with_eyes(soc=0.0, tariff=0.90)) == 0.0


# ---------------------------------------------------------------------------
# Invalid / strict edge cases: malformed brain inputs must die loudly.
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_observation_value_is_rejected(bad_value):
    observation = list(BASE_OBSERVATION)
    observation[0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        make_agent().act(tuple(observation))


@pytest.mark.parametrize("bad_soc", [-0.0001, 1.0001])
def test_soc_eye_outside_unit_interval_is_rejected(bad_soc):
    with pytest.raises(ValueError, match="SOC eye"):
        make_agent().act(with_eyes(soc=bad_soc, tariff=0.50))


@pytest.mark.parametrize("bad_tariff", [-0.0001, 1.0001])
def test_tariff_eye_outside_unit_interval_is_rejected(bad_tariff):
    with pytest.raises(ValueError, match="tariff eye"):
        make_agent().act(with_eyes(soc=0.50, tariff=bad_tariff))


@pytest.mark.parametrize(
    "cheap, expensive, message",
    [
        (-0.01, 0.80, "cheap_tariff"),
        (0.10, 1.01, "expensive_tariff"),
        (0.50, 0.50, "strictly less"),
        (0.80, 0.20, "strictly less"),
        (float("nan"), 0.80, "finite"),
        (0.10, float("inf"), "finite"),
    ],
)
def test_invalid_threshold_configuration_is_rejected(cheap, expensive, message):
    with pytest.raises(ValueError, match=message):
        Brain1Agent(
            cheap_tariff_max_normalized=cheap,
            expensive_tariff_min_normalized=expensive,
        )


# ---------------------------------------------------------------------------
# Invariants: deterministic, stateless, discrete answer-sheet outputs only.
# ---------------------------------------------------------------------------


def test_output_is_always_exactly_one_of_three_school_answer_actions():
    agent = make_agent()
    actions = {
        agent.act(with_eyes(soc=soc, tariff=tariff))
        for soc in (0.0, 0.25, 0.50, 0.75, 1.0)
        for tariff in (0.0, 0.10, 0.50, 0.80, 1.0)
    }

    assert actions <= {-1.0, 0.0, 1.0}
    assert actions == {-1.0, 0.0, 1.0}


def test_same_observation_is_deterministic_and_not_mutated():
    agent = make_agent()
    observation = [0.0, 1.0, 0.2, 0.50, 0.05, 0.1, 1.0]
    original = observation.copy()

    first = agent.act(observation)
    second = agent.act(observation)

    assert first == second == -1.0
    assert observation == original


def test_agent_configuration_does_not_change_after_act():
    agent = make_agent()
    before = (
        agent.cheap_tariff_max_normalized,
        agent.expensive_tariff_min_normalized,
    )

    agent.act(with_eyes(soc=0.50, tariff=0.05))

    assert (
        agent.cheap_tariff_max_normalized,
        agent.expensive_tariff_min_normalized,
    ) == before


def test_brain1_imports_no_production_bess_or_training_stack():
    source_path = Path(__file__).parents[1] / "EXPERIMENT_FIELD" / "brain1_agent.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert "EXPERIMENT_FIELD.brain_env" in imported_modules
    assert not any(module == "bess" or module.startswith("bess.") for module in imported_modules)
    assert not any("training" in module.lower() or "ppo" in module.lower() for module in imported_modules)


# ---------------------------------------------------------------------------
# Integration answer sheet: obvious cheap/expensive arbitrage must stay solved.
# ---------------------------------------------------------------------------


def test_brain1_solves_baby_arbitrage_answer_sheet_for_exact_7920_vnd_savings():
    episode = BrainEpisode(
        timesteps=tuple(
            BrainTimestepInput(
                net_load_kw=200.0,
                tariff_vnd_per_kwh=tariff,
                is_working_day=True,
            )
            for tariff in (1.0, 1.0, 100.0, 100.0)
        ),
        steps_per_day=48,
    )
    env = BrainEnv(
        initial_state_of_charge=0.10,
        minimum_state_of_charge=0.10,
        maximum_state_of_charge=0.90,
        battery_capacity_kwh=100.0,
        battery_power_kw=80.0,
        timestep_hours=0.50,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        demand_charge_vnd_per_kw=0.0,
        battery_wear_vnd_per_kwh=0.0,
        episode=episode,
    )
    agent = Brain1Agent(
        cheap_tariff_max_normalized=0.01,
        expensive_tariff_min_normalized=1.0,
    )

    observation = env.reset()
    actions = []
    final_result = None

    while True:
        action = agent.act(observation)
        actions.append(action)
        result = env.step(action)
        final_result = result
        if result.done:
            break
        assert result.next_observation is not None
        observation = result.next_observation

    assert tuple(actions) == (-1.0, -1.0, 1.0, 1.0)
    assert final_result is not None
    assert final_result.reward.monthly_savings_vnd == pytest.approx(7_920.0)
    assert env.bess_world.state_of_charge == pytest.approx(0.10)
