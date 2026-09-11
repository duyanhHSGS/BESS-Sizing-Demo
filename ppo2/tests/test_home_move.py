from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from bess.core.common import tariff_vector_day as legacy_tariff_vector_day
from bess.training.training_common import build_training_bess_config
from bess.training.training_launcher import PPO2_SCRIPT, TRAINING_MODULES
from ppo2.agent import PPO2Agent, PPO2InferenceAgent
from ppo2.data import DayData, MonthData
from ppo2.env import PPO2_OBS_DIM, PPO2_STEPS_PER_DAY, PPO2Env
from ppo2.settings import (
    PROJECT_ROOT,
    PPO2Config,
    build_training_ppo2_config,
    tariff_vector_day,
)


def _training_config(tmp_path: Path, *, sunday_no_peak: bool = True) -> Path:
    path = tmp_path / "training_config.json"
    path.write_text(
        json.dumps(
            {
                "charge_efficiency": 0.9,
                "discharge_efficiency": 0.9,
                "minimum_soc": 0.2,
                "maximum_soc": 0.9,
                "price_peak": 2251.0,
                "price_mid": 1332.0,
                "price_off": 904.0,
                "t_cap": 285000.0,
                "billing_mode": "2tc",
                "peak_windows": "17:30-22:30",
                "off_windows": "00:00-06:00",
                "sunday_no_peak": sunday_no_peak,
                "battery_wear_cost": 500.0,
            }
        ),
        encoding="utf-8",
    )
    return path


def _month(date_iso: str = "2026-06-01") -> MonthData:
    load = np.linspace(300.0, 850.0, PPO2_STEPS_PER_DAY, dtype=np.float64)
    pv = np.maximum(0.0, 180.0 * np.sin(np.linspace(-1.2, 1.9, PPO2_STEPS_PER_DAY)))
    return MonthData(
        days=[
            DayData(
                load=load,
                pv=pv,
                day_type="working",
                weather="clear",
                day_index=1,
                date_iso=date_iso,
            )
        ],
        source="test",
    )


def _cfg_pair(tmp_path: Path):
    path = _training_config(tmp_path)
    legacy, legacy_billing = build_training_bess_config(1250.0, 450.0, 0.25, path)
    new, new_billing = build_training_ppo2_config(1250.0, 450.0, 0.25, path)
    assert legacy_billing == new_billing == "2tc"
    return legacy, new


def test_root_ppo2_package_has_no_bess_or_gui_imports() -> None:
    package_root = Path(__file__).resolve().parents[1]
    forbidden_roots = {"bess", "web", "main"}
    offenders: list[str] = []
    for path in package_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] in forbidden_roots:
                        offenders.append(f"{path.name}: import {alias.name}")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".", 1)[0] in forbidden_roots
            ):
                offenders.append(f"{path.name}: from {node.module}")
    assert offenders == []


def test_gui_launcher_points_to_root_ppo2_runner() -> None:
    assert TRAINING_MODULES["ppo2"] == "ppo2.runner"
    assert PPO2_SCRIPT == PROJECT_ROOT / "ppo2" / "runner.py"
    assert PPO2_SCRIPT.is_file()


def test_legacy_import_doors_resolve_to_new_implementation() -> None:
    from bess.agents.ppo2_agent import PPO2Agent as LegacyAgent
    from bess.core.ppo2_env import PPO2Env as LegacyEnv
    from bess.evaluation.oracle.ppo2_oracle import run_oracle as legacy_run_oracle
    from bess.training.runners.train_ppo2_dataset import main as legacy_main
    from ppo2.oracle import run_oracle
    from ppo2.runner import main

    assert LegacyAgent is PPO2Agent
    assert LegacyEnv is PPO2Env
    assert legacy_run_oracle is run_oracle
    assert legacy_main is main


def test_ppo2_owned_config_matches_legacy_training_config_contract(tmp_path: Path) -> None:
    legacy, new = _cfg_pair(tmp_path)
    attributes = (
        "E_cap",
        "P_rated_nominal",
        "eta_ch",
        "eta_dis",
        "SOC_min",
        "SOC_max",
        "dt",
        "price_peak",
        "price_mid",
        "price_off",
        "T_cap",
        "peak_windows",
        "off_windows",
    )
    for name in attributes:
        assert getattr(new, name) == getattr(legacy, name)
    assert list(new.W1) == list(legacy.W1)
    assert list(new.W2) == list(legacy.W2)
    assert list(new.OFF) == list(legacy.OFF)
    assert list(new.INTER) == list(legacy.INTER)


def test_ppo2_tariff_matches_legacy_tariff_for_workday_and_sunday(tmp_path: Path) -> None:
    legacy, new = _cfg_pair(tmp_path)
    for date_iso in ("2026-06-01", "2026-06-07"):
        day = _month(date_iso).days[0]
        np.testing.assert_array_equal(
            tariff_vector_day(new, day),
            legacy_tariff_vector_day(legacy, day),
        )


def test_ppo2_owned_config_rejects_non_15_minute_dt(tmp_path: Path) -> None:
    path = _training_config(tmp_path)
    with pytest.raises(ValueError, match="15-minute"):
        build_training_ppo2_config(1250.0, 450.0, 0.5, path)


def test_ppo2_environment_trajectory_matches_legacy_config_contract(tmp_path: Path) -> None:
    legacy_cfg, new_cfg = _cfg_pair(tmp_path)
    month = _month()
    old_contract_env = PPO2Env(legacy_cfg, p_ref_kw=1000.0, degradation_cost_per_kwh_discharged=500.0, clip_penalty_per_kwh=100.0)
    new_home_env = PPO2Env(new_cfg, p_ref_kw=1000.0, degradation_cost_per_kwh_discharged=500.0, clip_penalty_per_kwh=100.0)
    obs_old = old_contract_env.reset(month, soc_init=0.2)
    obs_new = new_home_env.reset(month, soc_init=0.2)
    np.testing.assert_array_equal(obs_new, obs_old)

    actions = np.sin(np.linspace(0.0, 9.0, PPO2_STEPS_PER_DAY))
    for action in actions:
        next_old, reward_old, done_old, info_old = old_contract_env.step(float(action))
        next_new, reward_new, done_new, info_new = new_home_env.step(float(action))
        assert reward_new == pytest.approx(reward_old, rel=0.0, abs=1e-12)
        assert done_new is done_old
        for key in (
            "grid_kw",
            "d_run",
            "d_run_shaping",
            "p_requested_kw",
            "p_executed_kw",
            "rew_energy_delta",
            "rew_peak_delta",
            "rew_deg_cost",
            "rew_clip_cost",
        ):
            assert info_new[key] == pytest.approx(info_old[key], rel=0.0, abs=1e-12)
        if next_old is None:
            assert next_new is None
        else:
            np.testing.assert_array_equal(next_new, next_old)
    assert done_new is True


def test_existing_checkpoint_layout_still_loads_actor_only(tmp_path: Path) -> None:
    training_agent = PPO2Agent(PPO2_OBS_DIM, seed=123, device="cpu")
    training_agent.meta = {"reference_env": "ppo2_senior_15m_v1", "p_ref_kw": 1000.0}
    checkpoint = tmp_path / "policy_ppo2.pt"
    training_agent.save(checkpoint)

    inference = PPO2InferenceAgent(PPO2_OBS_DIM)
    meta = inference.load(str(checkpoint))
    obs = np.linspace(-0.5, 0.5, PPO2_OBS_DIM, dtype=np.float32)
    assert meta["reference_env"] == "ppo2_senior_15m_v1"
    assert inference.predict_action(obs) == pytest.approx(training_agent.predict_action(obs), abs=1e-7)


def test_ppo2_config_rejects_unaligned_tariff_boundary() -> None:
    with pytest.raises(ValueError, match="align to 15 minutes"):
        PPO2Config(
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
                "peak_windows": "17:37-22:30",
                "off_windows": "00:00-06:00",
            }
        )
