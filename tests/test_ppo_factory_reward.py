from __future__ import annotations

import math

from bess.core.bess_env import BrainEnv, BrainEpisode, BrainTimestepInput

DT_HOURS = 0.25
BATTERY_CAPACITY_KWH = 1250.0
BATTERY_POWER_KW = 450.0
ETA_CHARGE = 0.9
ETA_DISCHARGE = 0.9
SOC_MIN = 0.20
SOC_MAX = 0.90
DEMAND_FEE_VND_PER_KW = 285_000.0
WEAR_VND_PER_KWH = 50.0


def _env(
    *,
    tariff_vnd_per_kwh: float,
    load_kw: float = 100.0,
    initial_soc: float = 0.50,
) -> BrainEnv:
    episode = BrainEpisode(
        timesteps=(
            BrainTimestepInput(load_kw, tariff_vnd_per_kwh, True),
            BrainTimestepInput(load_kw, tariff_vnd_per_kwh, True),
        ),
        steps_per_day=2,
        power_scale_kw=1500.0,
    )
    env = BrainEnv(
        initial_state_of_charge=initial_soc,
        minimum_state_of_charge=SOC_MIN,
        maximum_state_of_charge=SOC_MAX,
        battery_capacity_kwh=BATTERY_CAPACITY_KWH,
        battery_power_kw=BATTERY_POWER_KW,
        timestep_hours=DT_HOURS,
        charge_efficiency=ETA_CHARGE,
        discharge_efficiency=ETA_DISCHARGE,
        demand_charge_vnd_per_kw=DEMAND_FEE_VND_PER_KW,
        battery_wear_vnd_per_kwh=WEAR_VND_PER_KWH,
        episode=episode,
    )
    env.reset()
    return env


def _first_reward(tariff_vnd_per_kwh: float, action: float, *, load_kw: float) -> float:
    env = _env(tariff_vnd_per_kwh=tariff_vnd_per_kwh, load_kw=load_kw)
    result = env.step(action)
    return result.reward.timestep_savings_vnd


def test_brain_reward_follows_tariff_direction() -> None:
    cheap = 904.0
    normal = 1332.0
    expensive = 2251.0

    cheap_charge = _first_reward(cheap, -1.0, load_kw=100.0)
    normal_charge = _first_reward(normal, -1.0, load_kw=100.0)
    expensive_charge = _first_reward(expensive, -1.0, load_kw=100.0)

    cheap_discharge = _first_reward(cheap, +1.0, load_kw=1000.0)
    normal_discharge = _first_reward(normal, +1.0, load_kw=1000.0)
    expensive_discharge = _first_reward(expensive, +1.0, load_kw=1000.0)

    assert cheap_charge < 0.0
    assert normal_charge < cheap_charge
    assert expensive_charge < normal_charge

    assert cheap_discharge > 0.0
    assert normal_discharge > cheap_discharge
    assert expensive_discharge > normal_discharge


def test_brain_reward_subtracts_symmetric_battery_wear() -> None:
    expected_throughput_kwh = BATTERY_POWER_KW * DT_HOURS
    expected_wear_vnd = expected_throughput_kwh * WEAR_VND_PER_KWH

    charge_env = _env(tariff_vnd_per_kwh=1332.0, load_kw=100.0)
    charge = charge_env.step(-1.0)
    discharge_env = _env(tariff_vnd_per_kwh=1332.0, load_kw=1000.0)
    discharge = discharge_env.step(+1.0)

    assert math.isclose(
        charge.bess.cost.battery_wear_cost_vnd,
        expected_wear_vnd,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        discharge.bess.cost.battery_wear_cost_vnd,
        expected_wear_vnd,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def test_brain_reward_penalizes_charging_that_creates_new_demand_peak() -> None:
    env = _env(tariff_vnd_per_kwh=904.0, load_kw=100.0)

    first = env.step(-1.0)
    second = env.step(-1.0)

    assert not first.done
    assert second.done
    assert first.reward.timestep_savings_vnd < 0.0
    assert second.reward.timestep_savings_vnd < first.reward.timestep_savings_vnd
    assert (
        first.reward.timestep_savings_vnd + second.reward.timestep_savings_vnd
        < 0.0
    )
    assert first.bess.cost.demand_peak_increase_kw == 0.0
    assert second.bess.cost.demand_peak_increase_kw > 0.0
    assert second.bess.physics.grid_import_kw > 100.0
    assert second.bess.meter.monthly_peak_kw > second.raw.meter.monthly_peak_kw
