from __future__ import annotations

import numpy as np
import pytest

from ppo2.data import DayData, MonthData
from ppo2.env import PPO2Env
from ppo2.oracle import run_oracle
from ppo2.settings import (
    PPO2_CONTROL_DT_MINUTES,
    PPO2_CONTROL_NATIVE_STEPS,
    PPO2_DECISIONS_PER_DAY,
    PPO2Config,
    ROLLOUT,
)


def _cfg() -> PPO2Config:
    return PPO2Config(
        {
            "E_cap_kWh": 1250.0,
            "P_rated_kW": 450.0,
            "eta_ch": 0.9,
            "eta_dis": 0.9,
            "soc_min": 0.2,
            "soc_max": 0.9,
            "dt_hours": 0.25,
            "price_peak": 2251.0,
            "price_mid": 1332.0,
            "price_off": 904.0,
            "T_cap": 285000.0,
            "peak_windows": "17:30-22:30",
            "off_windows": "00:00-06:00",
        }
    )


def _month(*, load: np.ndarray | None = None, pv: np.ndarray | None = None) -> MonthData:
    load_values = np.full(96, 300.0, dtype=np.float64) if load is None else np.asarray(load, dtype=np.float64)
    pv_values = np.zeros(96, dtype=np.float64) if pv is None else np.asarray(pv, dtype=np.float64)
    return MonthData(
        days=[
            DayData(
                load=load_values,
                pv=pv_values,
                day_type="working",
                weather="test",
                day_index=1,
                date_iso="2026-08-03",
            )
        ],
        source="test",
    )


def _env() -> PPO2Env:
    return PPO2Env(
        _cfg(),
        p_ref_kw=1000.0,
        degradation_cost_per_kwh_discharged=500.0,
        clip_penalty_per_kwh=100.0,
    )


def test_30m_constants_keep_15m_physics_and_one_month_rollout() -> None:
    assert PPO2_CONTROL_NATIVE_STEPS == 2
    assert PPO2_CONTROL_DT_MINUTES == pytest.approx(30.0)
    assert PPO2_DECISIONS_PER_DAY == 48
    assert ROLLOUT == 48 * 30


def test_step_control_holds_one_request_across_both_native_steps() -> None:
    env = _env()
    env.reset(_month())

    # Five native history samples are still required. Three 30-minute decisions
    # cover six samples, so the fourth decision is the first live action.
    for _ in range(3):
        _, _, done, info = env.step_control(-0.5)
        assert done is False
        assert info["action_held"] is True
        assert info["native_steps"] == 2

    _, total_reward, done, info = env.step_control(-0.5)
    assert done is False
    assert info["action_held"] is False
    assert info["native_steps"] == 2
    assert len(info["native_infos"]) == 2
    requested = [native["p_requested_kw"] for native in info["native_infos"]]
    assert requested == pytest.approx([-225.0, -225.0])
    assert info["rew_total"] == pytest.approx(
        sum(native["rew_total"] for native in info["native_infos"])
    )
    assert total_reward == pytest.approx(info["rew_total"])


def test_step_control_rejects_nonpositive_native_step_count() -> None:
    env = _env()
    env.reset(_month())
    with pytest.raises(ValueError, match="native_steps must be > 0"):
        env.step_control(0.0, native_steps=0)


def test_step_control_stops_cleanly_at_episode_end() -> None:
    env = _env()
    env.reset(_month())
    done = False
    decisions = 0
    native_steps = 0
    while not done:
        _, _, done, info = env.step_control(0.0)
        decisions += 1
        native_steps += info["native_steps"]
    assert decisions == 48
    assert native_steps == 96


def test_30m_oracle_holds_net_battery_power_inside_each_control_block() -> None:
    cfg = _cfg()
    # Alternating PV makes the restriction meaningful: the unrestricted Oracle
    # may exploit each 15-minute slot independently, while the 30-minute Oracle
    # must expose one net battery request across each pair.
    pv = np.zeros(96, dtype=np.float64)
    pv[1::2] = 500.0
    month = _month(load=np.full(96, 100.0), pv=pv)
    unrestricted = run_oracle(
        month,
        cfg,
        degradation_cost_per_kwh_discharged=500.0,
        control_steps=1,
    )
    constrained = run_oracle(
        month,
        cfg,
        degradation_cost_per_kwh_discharged=500.0,
        control_steps=2,
    )

    power = np.concatenate(constrained["p_bess_days"])
    assert np.allclose(power[0::2], power[1::2], rtol=0.0, atol=1e-6)
    assert constrained["control_steps"] == 2
    assert constrained["lp_objective_vnd"] >= unrestricted["lp_objective_vnd"] - 1e-4


def test_oracle_rejects_control_step_count_that_cannot_tile_horizon() -> None:
    with pytest.raises(ValueError, match="positive divisor"):
        run_oracle(
            _month(),
            _cfg(),
            degradation_cost_per_kwh_discharged=500.0,
            control_steps=5,
        )


# TODO(PPO2-30M): keep these tests pinned to the single-variable experiment:
# 15-minute physics + fixed 30-minute demand meter + one actor action per meter block.
