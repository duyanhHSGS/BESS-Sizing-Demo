import math

import pytest

from EXPERIMENT_FIELD.brain_env import (
    BrainEnvironmentStepResult,
    BrainEnv,
    BrainEpisode,
    BrainRewardResult,
    BrainTimestepInput,
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


def make_episode(*steps: BrainTimestepInput) -> BrainEpisode:
    return BrainEpisode(
        timesteps=steps,
        steps_per_day=48,
    )


def make_env(episode: BrainEpisode) -> BrainEnv:
    return BrainEnv(**BASE_ENV, episode=episode)


def test_reset_returns_first_seven_eye_observation():
    episode = make_episode(
        BrainTimestepInput(net_load_kw=200.0, tariff_vnd_per_kwh=10.0, is_working_day=True),
        BrainTimestepInput(net_load_kw=100.0, tariff_vnd_per_kwh=2.0, is_working_day=False),
    )
    env = make_env(episode)

    observation = env.reset()

    assert len(observation) == 7
    assert observation[0] == pytest.approx(0.0)
    assert observation[1] == pytest.approx(1.0)
    assert observation[2] == pytest.approx(0.2)
    assert observation[3] == pytest.approx((0.80 - 0.10) / (0.90 - 0.10))
    assert observation[4] == pytest.approx(1.0)
    assert observation[5] == pytest.approx(0.0)
    assert observation[6] == pytest.approx(1.0)


def test_environment_step_returns_dt_and_monthly_savings_then_next_observation():
    episode = make_episode(
        BrainTimestepInput(net_load_kw=200.0, tariff_vnd_per_kwh=10.0, is_working_day=True),
        BrainTimestepInput(net_load_kw=100.0, tariff_vnd_per_kwh=2.0, is_working_day=False),
    )
    env = make_env(episode)
    env.reset()

    result = env.step(1.0)

    assert isinstance(result, BrainEnvironmentStepResult)
    assert isinstance(result.reward, BrainRewardResult)
    assert result.info is result
    assert result.done is False
    assert result.reward.timestep_savings_vnd == pytest.approx(10_450.0)
    assert result.reward.monthly_savings_vnd == pytest.approx(10_450.0)
    assert result.reward.timestep_savings_vnd == pytest.approx(result.net_battery_savings_vnd)
    assert result.reward.monthly_savings_vnd == pytest.approx(
        result.cumulative_net_battery_savings_vnd
    )

    assert result.next_observation is not None
    expected_angle = 2.0 * math.pi / 48.0
    assert result.next_observation[0] == pytest.approx(math.sin(expected_angle))
    assert result.next_observation[1] == pytest.approx(math.cos(expected_angle))
    assert result.next_observation[2] == pytest.approx(0.1)
    assert result.next_observation[3] == pytest.approx((0.75 - 0.10) / (0.90 - 0.10))
    assert result.next_observation[4] == pytest.approx(0.2)
    assert result.next_observation[5] == pytest.approx(0.1)
    assert result.next_observation[6] == pytest.approx(0.0)


def test_terminal_reward_is_exact_final_month_saving_and_has_no_fake_next_observation():
    episode = make_episode(
        BrainTimestepInput(net_load_kw=200.0, tariff_vnd_per_kwh=10.0, is_working_day=True),
        BrainTimestepInput(net_load_kw=100.0, tariff_vnd_per_kwh=2.0, is_working_day=False),
    )
    env = make_env(episode)
    env.reset()

    first = env.step(1.0)
    final = env.step(-1.0)

    assert first.done is False
    assert final.done is True
    assert final.next_observation is None
    assert final.reward.timestep_savings_vnd == pytest.approx(-10_150.0)
    assert final.reward.monthly_savings_vnd == pytest.approx(300.0)
    assert first.reward.timestep_savings_vnd + final.reward.timestep_savings_vnd == pytest.approx(
        final.reward.monthly_savings_vnd
    )
    assert final.reward.monthly_savings_vnd == pytest.approx(
        env.raw_world.total_operating_cost_vnd - env.bess_world.total_operating_cost_vnd
    )


def test_one_timestep_episode_is_exact_done_boundary():
    env = make_env(
        make_episode(
            BrainTimestepInput(net_load_kw=200.0, tariff_vnd_per_kwh=10.0, is_working_day=True)
        )
    )
    env.reset()

    result = env.step(0.0)

    assert result.done is True
    assert result.next_observation is None
    assert result.reward.timestep_savings_vnd == pytest.approx(0.0)
    assert result.reward.monthly_savings_vnd == pytest.approx(0.0)


def test_invalid_action_fails_before_either_world_mutates():
    env = make_env(
        make_episode(
            BrainTimestepInput(net_load_kw=200.0, tariff_vnd_per_kwh=10.0, is_working_day=True)
        )
    )
    env.reset()
    starting_soc = env.bess_world.state_of_charge

    with pytest.raises(ValueError, match=r"inside \[-1, 1\]"):
        env.step(1.0001)

    assert env.bess_world.timestep_index == 0
    assert env.raw_world.timestep_index == 0
    assert env.bess_world.state_of_charge == pytest.approx(starting_soc)
    assert env.bess_world.total_operating_cost_vnd == pytest.approx(0.0)
    assert env.raw_world.total_operating_cost_vnd == pytest.approx(0.0)


def test_step_after_done_fails_loudly_without_advancing_again():
    env = make_env(
        make_episode(
            BrainTimestepInput(net_load_kw=200.0, tariff_vnd_per_kwh=10.0, is_working_day=True)
        )
    )
    env.reset()
    env.step(0.0)
    finished_bess_cost = env.bess_world.total_operating_cost_vnd
    finished_raw_cost = env.raw_world.total_operating_cost_vnd

    with pytest.raises(RuntimeError, match="episode is finished"):
        env.step(0.0)

    assert env.bess_world.timestep_index == 1
    assert env.raw_world.timestep_index == 1
    assert env.bess_world.total_operating_cost_vnd == pytest.approx(finished_bess_cost)
    assert env.raw_world.total_operating_cost_vnd == pytest.approx(finished_raw_cost)


def test_reset_restores_initial_month_state_after_finished_episode():
    env = make_env(
        make_episode(
            BrainTimestepInput(net_load_kw=200.0, tariff_vnd_per_kwh=10.0, is_working_day=True)
        )
    )
    initial_observation = env.reset()
    env.step(1.0)

    reset_observation = env.reset()

    assert reset_observation == pytest.approx(initial_observation)
    assert env.bess_world.state_of_charge == pytest.approx(0.80)
    assert env.bess_world.timestep_index == 0
    assert env.raw_world.timestep_index == 0
    assert env.bess_world.total_operating_cost_vnd == pytest.approx(0.0)
    assert env.raw_world.total_operating_cost_vnd == pytest.approx(0.0)
    assert env.net_battery_savings_vnd == pytest.approx(0.0)


def test_episode_contract_rejects_empty_and_invalid_timestep_data():
    with pytest.raises(ValueError, match="at least one timestep"):
        BrainEpisode(timesteps=(), steps_per_day=48)

    with pytest.raises(ValueError, match="must not be negative"):
        BrainTimestepInput(
            net_load_kw=100.0,
            tariff_vnd_per_kwh=-1.0,
            is_working_day=True,
        )

    with pytest.raises(TypeError, match="must be a bool"):
        BrainTimestepInput(
            net_load_kw=100.0,
            tariff_vnd_per_kwh=1.0,
            is_working_day=1,
        )


def test_zero_tariff_month_still_builds_finite_zero_tariff_observation():
    env = make_env(
        make_episode(
            BrainTimestepInput(net_load_kw=100.0, tariff_vnd_per_kwh=0.0, is_working_day=True)
        )
    )

    observation = env.reset()

    assert observation[4] == pytest.approx(0.0)
    assert all(math.isfinite(value) for value in observation)
