from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from bess.agents.ppo2_agent import (
    PPO2Agent,
    PPO2InferenceAgent,
    RolloutBuffer,
    _adv_share_of_return,
)
from bess.core.common import load_system_config
from bess.core.ppo2_env import PPO2_OBS_DIM, PPO2Env
from bess.core.scenario_gen import DayData, MonthData
from bess.evaluation.baselines import run_drl_policy
from bess.evaluation.oracle.ppo2_oracle import run_oracle, score_month
from bess.training.runners.train_ppo2_dataset import (
    BC_EPOCHS,
    EVAL_EVERY_UPDATES,
    MIN_MONTH_COVERAGE,
    ROLLOUT,
    TEST_MONTHS,
    VAL_MONTHS,
    _fit_test_split,
    _split_months,
)


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
        degradation_cost_per_kwh_discharged=500.0,
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
    assert obs[6] == pytest.approx(env.cfg.SOC_min)
    assert env.n_steps == 96
    assert env.block_slots == 2


def test_ppo2_tariff_countdown_sees_midnight_boundary() -> None:
    env = _env()
    env.reset(_month())
    # 22:00 at 15-minute data is slot 88. With default tariff, 22:00 is still
    # peak until 22:30; this verifies the observation follows tariff truth.
    env.t = round(22.0 / env.dt)
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


def test_ppo2_holds_exactly_five_reference_history_slots() -> None:
    env = _env()
    env.reset(_month(load_kw=100.0))
    for _ in range(5):
        _, _, _, info = env.step(-1.0)
        assert info["action_held"] is True
        assert info["p_executed_kw"] == pytest.approx(0.0)
    _, _, _, info = env.step(-1.0)
    assert info["action_held"] is False
    assert info["p_executed_kw"] < 0.0


def test_ppo2_rejects_non_15_minute_config() -> None:
    cfg = load_system_config()
    cfg.set_dt(0.5)
    with pytest.raises(ValueError, match="15-minute"):
        PPO2Env(
            cfg,
            p_ref_kw=500.0,
            degradation_cost_per_kwh_discharged=500.0,
        )


def test_ppo2_oracle_objective_matches_reference_scorer() -> None:
    cfg = load_system_config()
    month = _month(load_kw=100.0, pv_kw=0.0)
    oracle = run_oracle(
        month,
        cfg,
        degradation_cost_per_kwh_discharged=500.0,
    )
    scored = score_month(
        oracle["p_grid_days"],
        cfg,
        days=month.days,
        p_bess_days=oracle["p_bess_days"],
        soc_days=oracle["soc_days"],
        degradation_cost_per_kwh_discharged=500.0,
    )
    assert oracle["lp_objective_vnd"] == pytest.approx(
        scored["total_cost_vnd"], rel=1e-6
    )


def test_ppo2_reference_training_constants_match_senior() -> None:
    assert ROLLOUT == 96 * 30
    assert MIN_MONTH_COVERAGE == pytest.approx(0.8)
    assert VAL_MONTHS == 2
    assert TEST_MONTHS == 1
    assert EVAL_EVERY_UPDATES == 20
    assert BC_EPOCHS == 10


def test_ppo2_split_uses_calendar_months_not_day_slices() -> None:
    import calendar
    from datetime import date

    days: list[DayData] = []
    day_index = 1
    for month_number in range(1, 5):
        count = calendar.monthrange(2026, month_number)[1]
        for day_number in range(1, count + 1):
            days.append(
                DayData(
                    load=np.full(96, 100.0),
                    pv=np.zeros(96),
                    day_type="working",
                    weather="test",
                    day_index=day_index,
                    date_iso=date(2026, month_number, day_number).isoformat(),
                )
            )
            day_index += 1
    train, val, test = _split_months(days, 0.8)
    assert [month.source for month in train] == ["csv:2026-01"]
    assert [month.source for month in val] == ["csv:2026-02", "csv:2026-03"]
    assert [month.source for month in test] == ["csv:2026-04"]


def test_ppo2_fit_test_reuses_all_supplied_days_for_all_three_sets() -> None:
    from datetime import date, timedelta

    days = [
        DayData(
            load=np.full(96, 100.0 + index),
            pv=np.zeros(96),
            day_type="working",
            weather="test",
            day_index=index + 1,
            date_iso=(date(2026, 1, 1) + timedelta(days=index)).isoformat(),
        )
        for index in range(30)
    ]
    train, val, test = _fit_test_split(days)
    assert train[0] is val[0] is test[0]
    assert train[0].source == "csv:ppo2-fit-test-overlap"
    assert len(train[0].days) == 30
    assert train[0].days[0].date_iso == "2026-01-01"
    assert train[0].days[-1].date_iso == "2026-01-30"


def test_ppo2_inference_checkpoint_is_actor_only_and_matches_training_actor() -> None:
    agent = PPO2Agent(PPO2_OBS_DIM, seed=11, device="cpu")
    agent.meta = {
        "obs_dim": PPO2_OBS_DIM,
        "reference_env": "ppo2_senior_15m_v1",
        "native_dt_minutes": 15.0,
        "control_dt_minutes": 15.0,
        "native_steps_per_action": 1,
    }
    probe = np.linspace(-0.5, 0.5, PPO2_OBS_DIM, dtype=np.float32)
    expected = agent.predict_action(probe)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "policy.pt"
        agent.save(path)
        inference = PPO2InferenceAgent(PPO2_OBS_DIM)
        inference.load(path)
    assert not hasattr(inference.net, "critic_energy")
    assert not hasattr(inference.net, "critic_peak")
    assert inference.predict_action(probe) == pytest.approx(expected)


def test_shared_rollout_uses_ppo2_reference_environment_from_meta() -> None:
    cfg = load_system_config()
    agent = PPO2Agent(PPO2_OBS_DIM, seed=13, device="cpu")
    agent.meta = {
        "reference_env": "ppo2_senior_15m_v1",
        "native_dt_minutes": 15.0,
        "control_dt_minutes": 15.0,
        "native_steps_per_action": 1,
        "degradation_cost_per_kwh_discharged": 500.0,
    }
    result = run_drl_policy(_month(load_kw=100.0), cfg, agent, p_ref_kw=500.0)
    assert result["decision_count"] == 96
    assert result["soc_days"][0][0] == pytest.approx(cfg.SOC_min)


def test_adv_share_of_return_matches_variance_ratio() -> None:
    returns = np.array([1.0, 3.0, 6.0], dtype=np.float32)
    values = np.array([0.5, 2.0, 4.0], dtype=np.float32)
    expected = float(np.var(returns - values) / np.var(returns))
    assert _adv_share_of_return(returns, values) == pytest.approx(expected)


def test_ppo2_update_smoke_populates_advantage_diagnostics() -> None:
    agent = PPO2Agent(obs_dim=PPO2_OBS_DIM, seed=7, device="cpu", epochs=1, minibatch=4)
    buffer = RolloutBuffer(size=4, obs_dim=PPO2_OBS_DIM)
    obs = np.zeros(PPO2_OBS_DIM, dtype=np.float32)
    for step in range(4):
        action, logp, latent, value_energy, value_peak = agent.act(obs)
        buffer.add(
            obs,
            action,
            latent,
            logp,
            0.01 * (step + 1),
            -0.005 * step,
            value_energy,
            value_peak,
            float(step == 3),
        )
    agent.update(buffer, last_val_energy=0.0, last_val_peak=0.0)
    assert np.isfinite(agent.diagnostics["adv_share_energy"])
    assert np.isfinite(agent.diagnostics["adv_share_peak"])
