from __future__ import annotations

import numpy as np

from bess.agents.sadrbc import SADRBCConfig
from bess.core.bess_env import BESSEnv
from bess.core.scenario_gen import DayData, MonthData


DT_HOURS = 0.25
STEPS_PER_DAY = 96


def _config() -> SADRBCConfig:
    return SADRBCConfig({
        "E_cap_kWh": 1250.0,
        "P_rated_kW": 450.0,
        "eta_ch": 0.9,
        "eta_dis": 0.9,
        "soc_min": 0.20,
        "soc_max": 0.90,
        "soc_eod": 0.50,
        "dt_hours": DT_HOURS,
        "price_off": 904.0,
        "price_mid": 1332.0,
        "price_peak": 2251.0,
        "T_cap": 285000.0,
        "off_windows": "00:00-06:00",
        "peak_windows": "17:30-22:30",
    })


def _month(load_kw: float) -> MonthData:
    day = DayData(
        load=np.full(STEPS_PER_DAY, load_kw, dtype=np.float64),
        pv=np.zeros(STEPS_PER_DAY, dtype=np.float64),
        day_type="working",
        weather="test",
        day_index=0,
        date_iso="2026-08-10",
    )
    return MonthData(days=[day], source="test")


def _env(*, load_kw: float = 100.0, initial_peak_kw: float = 2000.0) -> BESSEnv:
    env = BESSEnv(
        _config(),
        reference_power_kw=1500.0,
        initial_running_peak_kw=initial_peak_kw,
        discount_factor=0.995,
        control_interval_minutes=15.0,
        degradation_cost_vnd_per_kwh=50.0,
        reward_mode="factory_dispatch_v1",
    )
    env.reset(_month(load_kw), initial_state_of_charge=0.50)
    return env


def _reward_at_step(step: int, action: float, *, load_kw: float = 100.0) -> float:
    env = _env(load_kw=load_kw)
    env.current_timestep_index = step
    _, reward, _, _ = env.step(action)
    return reward


def test_factory_reward_teaches_tariff_direction() -> None:
    # 00:00 cheap, 12:00 normal, 18:00 expensive for this tariff config.
    cheap_step = 0
    normal_step = 48
    expensive_step = 72

    cheap_charge = _reward_at_step(cheap_step, -1.0)
    normal_charge = _reward_at_step(normal_step, -1.0)
    expensive_charge = _reward_at_step(expensive_step, -1.0)

    # Start with enough load and SOC so a +1 action can actually discharge.
    cheap_discharge = _reward_at_step(cheap_step, +1.0, load_kw=1000.0)
    normal_discharge = _reward_at_step(normal_step, +1.0, load_kw=1000.0)
    expensive_discharge = _reward_at_step(expensive_step, +1.0, load_kw=1000.0)

    assert cheap_charge > 0.0
    assert normal_charge < 0.0
    assert expensive_charge < normal_charge

    assert cheap_discharge < 0.0
    assert normal_discharge > 0.0
    assert expensive_discharge > normal_discharge


def test_factory_reward_inventory_value_sits_between_break_even_values() -> None:
    env = _env()
    cfg = env.config
    wear = env.degradation_cost_vnd_per_kwh

    cheap_charge_break_even = (cfg.price_off + wear) / cfg.eta_ch
    normal_discharge_break_even = (cfg.price_mid - wear) * cfg.eta_dis

    assert cheap_charge_break_even < env._inventory_value_vnd_per_stored_kwh
    assert env._inventory_value_vnd_per_stored_kwh < normal_discharge_break_even


def test_factory_reward_crushes_cheap_charging_that_creates_new_demand_peak() -> None:
    env = _env(load_kw=100.0, initial_peak_kw=100.0)

    # A 30-minute demand block is two 15-minute samples. Charging at rated
    # power makes grid import 550 kW for the whole block, raising PMax by
    # 450 kW. The demand-charge punishment must dominate the cheap-charge
    # arbitrage encouragement.
    _, first_reward, first_done, _ = env.step(-1.0)
    _, second_reward, second_done, info = env.step(-1.0)

    assert not first_done
    assert not second_done
    assert first_reward > 0.0
    assert second_reward < 0.0
    assert first_reward + second_reward < 0.0
    assert info["peak_delta"] > 0.0
    assert info["d_run"] > 100.0
