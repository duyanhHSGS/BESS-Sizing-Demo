from __future__ import annotations

import numpy as np
import pytest

from bess.core.common import load_system_config
from bess.core.ppo2_env import PPO2Env, PPO2_OBS_DIM
from bess.core.scenario_gen import DayData, MonthData


def _month(*, load_kw: float = 100.0, pv_kw: float = 0.0) -> MonthData:
    cfg = load_system_config()
    n_steps = round(24.0 / cfg.dt)
    day = DayData(
        load=np.full(n_steps, load_kw, dtype=np.float64),
        pv=np.full(n_steps, pv_kw, dtype=np.float64),
        day_type="working",
        weather="test",
        day_index=1,
        date_iso="2026-08-03",
    )
    return MonthData(days=[day], source="test")


def _env() -> PPO2Env:
    cfg = load_system_config()
    return PPO2Env(
        cfg,
        p_ref_kw=500.0,
        degradation_cost_per_kwh_discharged=50.0,
        clip_penalty_per_kwh=100.0,
    )


def test_ppo2_observation_is_senior_style_17d() -> None:
    env = _env()
    obs = env.reset(_month())
    assert obs.shape == (PPO2_OBS_DIM,)
    assert PPO2_OBS_DIM == 17
    assert np.all(np.isfinite(obs))
    assert obs[0] == pytest.approx(0.0)
    assert obs[1] == pytest.approx(1.0)
    assert obs[7] == pytest.approx(env.cfg.price_off / env.cfg.price_peak)


def test_ppo2_tariff_countdown_sees_midnight_boundary() -> None:
    env = _env()
    env.reset(_month())
    # 22:00 at 15-minute data is slot 88. With default tariff, 22:00 is still
    # peak until 22:30; this verifies the observation follows tariff truth.
    env.t = int(round(22.0 / env.dt))
    obs = env._obs()
    assert obs[7] == pytest.approx(env.cfg.price_peak / env.cfg.price_peak)
    assert obs[8] == pytest.approx(0.5 / 24.0)


def test_ppo2_projection_has_no_economic_peak_safe_charge_guard() -> None:
    env = _env()
    env.reset(_month(load_kw=100.0))
    env.soc = 0.50
    env.d_run = 100.0
    mapped = env.project_action(-1.0, load=100.0, pv=0.0)
    # Senior semantics: economics do not clip charge to d_run - net_load (=0).
    assert mapped.charge_grid_kw > 0.0


def test_ppo2_demand_meter_closes_fixed_30m_blocks() -> None:
    env = _env()
    env.reset(_month(load_kw=100.0))
    assert env.block_slots == 2
    _, _, _, first = env.step(0.0)
    _, _, _, second = env.step(0.0)
    assert first["demand_block_closed"] is False
    assert second["demand_block_closed"] is True
    assert second["demand_kw"] == pytest.approx(100.0)


def test_ppo2_holds_actions_until_history_is_ready() -> None:
    env = _env()
    env.reset(_month(load_kw=100.0))
    _, _, _, info = env.step(-1.0)
    assert info["action_held"] is True
    assert info["p_executed_kw"] == pytest.approx(0.0)
