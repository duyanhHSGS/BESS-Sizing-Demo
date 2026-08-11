import math

import pytest

import EXPERIMENT_FIELD.brain_env as brain_env_module

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


def test_post_commit_accounting_failure_rolls_both_worlds_back(monkeypatch):
    env = make_env(
        make_episode(
            BrainTimestepInput(net_load_kw=200.0, tariff_vnd_per_kwh=10.0, is_working_day=True)
        )
    )
    env.reset()
    starting_soc = env.bess_world.state_of_charge
    starting_bess_meter = env.bess_world.meter_state
    starting_raw_meter = env.raw_world.meter_state
    monkeypatch.setattr(brain_env_module, "_money_values_close", lambda *_values: False)

    with pytest.raises(RuntimeError, match="Battery savings accounting invariant failed"):
        env.step(1.0)

    assert env.bess_world.state_of_charge == pytest.approx(starting_soc)
    assert env.bess_world.meter_state == starting_bess_meter
    assert env.raw_world.meter_state == starting_raw_meter
    assert env.bess_world.timestep_index == 0
    assert env.raw_world.timestep_index == 0
    assert env.bess_world.total_operating_cost_vnd == pytest.approx(0.0)
    assert env.raw_world.total_operating_cost_vnd == pytest.approx(0.0)


def test_money_comparison_uses_source_total_scale_without_hiding_real_drift():
    assert brain_env_module._money_values_close(0.0, 0.0005, 1_000_000_000.0)
    assert not brain_env_module._money_values_close(0.0, 1.0, 1_000_000.0)


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


def test_episode_can_use_applied_tariff_ruler_larger_than_day_maximum():
    episode = BrainEpisode(
        timesteps=(
            BrainTimestepInput(
                net_load_kw=100.0,
                tariff_vnd_per_kwh=20.0,
                is_working_day=False,
            ),
        ),
        steps_per_day=48,
        tariff_normalization_vnd_per_kwh=30.0,
    )
    env = make_env(episode)

    observation = env.reset()

    assert episode.maximum_tariff_vnd_per_kwh == pytest.approx(20.0)
    assert episode.tariff_normalization_denominator_vnd_per_kwh == pytest.approx(30.0)
    assert observation[4] == pytest.approx(20.0 / 30.0)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_episode_rejects_invalid_explicit_tariff_ruler(bad_value):
    with pytest.raises(ValueError, match="tariff_normalization"):
        BrainEpisode(
            timesteps=(
                BrainTimestepInput(
                    net_load_kw=100.0,
                    tariff_vnd_per_kwh=20.0,
                    is_working_day=True,
                ),
            ),
            steps_per_day=48,
            tariff_normalization_vnd_per_kwh=bad_value,
        )


# ---------------------------------------------------------------------------
# BABY TEST 1: pure energy arbitrage, before any neural-network chaos exists.
#
# Four 30-minute samples, constant 200 kW factory load:
#   cheap, cheap, EXPENSIVE, EXPENSIVE
#      1,     1,       100,       100 VND/kWh
#
# Battery starts at minimum SOC, has exactly 80 kWh usable room, 100% efficiency,
# zero wear, and zero demand charge. Therefore buying 80 kWh for 1 VND/kWh and
# later avoiding 80 kWh at 100 VND/kWh MUST save exactly 7,920 VND.
# ---------------------------------------------------------------------------


def make_baby_1_arbitrage_env() -> BrainEnv:
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
    return BrainEnv(
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


def run_baby_1_actions(actions: tuple[float, float, float, float]):
    env = make_baby_1_arbitrage_env()
    env.reset()
    transitions = tuple(env.step(action) for action in actions)
    return env, transitions


def test_baby_1_cheap_charge_expensive_discharge_has_exact_kw_kwh_and_vnd_trace():
    env, steps = run_baby_1_actions((-1.0, -1.0, 1.0, 1.0))

    expected = (
        # final_battery_kw, grid_import_kw, start_soc, next_soc, sample_kwh,
        # bess_energy_cost, ghost_energy_cost, timestep_savings
        (-80.0, 280.0, 0.10, 0.50, 140.0, 140.0, 100.0, -40.0),
        (-80.0, 280.0, 0.50, 0.90, 140.0, 140.0, 100.0, -40.0),
        (80.0, 120.0, 0.90, 0.50, 60.0, 6_000.0, 10_000.0, 4_000.0),
        (80.0, 120.0, 0.50, 0.10, 60.0, 6_000.0, 10_000.0, 4_000.0),
    )

    for step, (
        final_battery_kw,
        grid_import_kw,
        starting_soc,
        next_soc,
        sample_energy_kwh,
        bess_energy_cost_vnd,
        raw_energy_cost_vnd,
        timestep_savings_vnd,
    ) in zip(steps, expected):
        assert step.bess.physics.final_battery_kw == pytest.approx(final_battery_kw)
        assert step.bess.physics.grid_import_kw == pytest.approx(grid_import_kw)
        assert step.bess.physics.starting_soc == pytest.approx(starting_soc)
        assert step.bess.physics.next_soc == pytest.approx(next_soc)
        assert step.bess.physics.battery_throughput_kwh == pytest.approx(40.0)
        assert step.bess.meter.sample_energy_kwh == pytest.approx(sample_energy_kwh)
        assert step.bess.cost.electricity_energy_cost_vnd == pytest.approx(bess_energy_cost_vnd)
        assert step.raw.cost.electricity_energy_cost_vnd == pytest.approx(raw_energy_cost_vnd)
        assert step.bess.cost.demand_cost_vnd == pytest.approx(0.0)
        assert step.bess.cost.battery_wear_cost_vnd == pytest.approx(0.0)
        assert step.reward.timestep_savings_vnd == pytest.approx(timestep_savings_vnd)

    # Directional outside-world kW must also be boringly exact at 100% efficiency.
    assert steps[0].bess.physics.grid_to_battery_kw == pytest.approx(80.0)
    assert steps[0].bess.physics.battery_to_factory_kw == pytest.approx(0.0)
    assert steps[2].bess.physics.grid_to_battery_kw == pytest.approx(0.0)
    assert steps[2].bess.physics.battery_to_factory_kw == pytest.approx(80.0)
    assert all(step.bess.physics.conversion_loss_kw == pytest.approx(0.0) for step in steps)

    # Whole-episode accounting truth: same starting/ending SOC means no stored-energy free lunch.
    final = steps[-1]
    assert env.bess_world.state_of_charge == pytest.approx(0.10)
    assert env.raw_world.total_operating_cost_vnd == pytest.approx(20_200.0)
    assert env.bess_world.total_operating_cost_vnd == pytest.approx(12_280.0)
    assert final.reward.monthly_savings_vnd == pytest.approx(7_920.0)
    assert sum(step.reward.timestep_savings_vnd for step in steps) == pytest.approx(7_920.0)
    assert final.reward.monthly_savings_vnd == pytest.approx(
        env.raw_world.total_operating_cost_vnd - env.bess_world.total_operating_cost_vnd
    )
    assert final.done is True
    assert final.next_observation is None


def test_baby_1_obvious_arbitrage_beats_all_idiot_manual_actions():
    policies = {
        "always_idle": (0.0, 0.0, 0.0, 0.0),
        "always_charge": (-1.0, -1.0, -1.0, -1.0),
        "always_discharge": (1.0, 1.0, 1.0, 1.0),
        "charge_only_cheap": (-1.0, -1.0, 0.0, 0.0),
        "discharge_only_expensive": (0.0, 0.0, 1.0, 1.0),
        "cheap_charge_expensive_discharge": (-1.0, -1.0, 1.0, 1.0),
    }
    expected_savings_vnd = {
        "always_idle": 0.0,
        "always_charge": -80.0,
        "always_discharge": 0.0,
        "charge_only_cheap": -80.0,
        "discharge_only_expensive": 0.0,
        "cheap_charge_expensive_discharge": 7_920.0,
    }

    actual_savings_vnd = {}
    for name, actions in policies.items():
        env, steps = run_baby_1_actions(actions)
        actual_savings_vnd[name] = steps[-1].reward.monthly_savings_vnd
        assert actual_savings_vnd[name] == pytest.approx(expected_savings_vnd[name])
        assert actual_savings_vnd[name] == pytest.approx(
            env.raw_world.total_operating_cost_vnd - env.bess_world.total_operating_cost_vnd
        )

    obvious = actual_savings_vnd["cheap_charge_expensive_discharge"]
    stupid_best = max(
        savings
        for name, savings in actual_savings_vnd.items()
        if name != "cheap_charge_expensive_discharge"
    )
    assert obvious == pytest.approx(7_920.0)
    assert obvious > stupid_best


def test_baby_1_soc_boundaries_make_stupid_actions_physically_harmless_when_blocked():
    # Empty battery cannot discharge even if the controller screams +1 forever.
    _, empty_discharge_steps = run_baby_1_actions((1.0, 1.0, 1.0, 1.0))
    assert all(step.bess.physics.final_battery_kw == pytest.approx(0.0) for step in empty_discharge_steps)
    assert all(step.bess.physics.grid_import_kw == pytest.approx(200.0) for step in empty_discharge_steps)

    # Always-charge fills on the two cheap samples, then Battery Police blocks further charge.
    _, always_charge_steps = run_baby_1_actions((-1.0, -1.0, -1.0, -1.0))
    assert always_charge_steps[1].bess.physics.next_soc == pytest.approx(0.90)
    assert always_charge_steps[2].bess.physics.requested_battery_kw == pytest.approx(-80.0)
    assert always_charge_steps[2].bess.physics.battery_after_police_kw == pytest.approx(0.0)
    assert always_charge_steps[2].bess.physics.final_battery_kw == pytest.approx(0.0)
    assert always_charge_steps[2].bess.physics.grid_import_kw == pytest.approx(200.0)
