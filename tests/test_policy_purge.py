from __future__ import annotations

from unittest.mock import patch

import pytest

from bess.core.settings import DEFAULT_PARAMETERS
from bess.evaluation.benchmarking import _selected_checkpoints


def _checkpoint(algo: str) -> dict:
    reference_env = "ppo2_senior_15m_v1" if algo == "ppo2" else None
    meta = {
        "algo": algo,
        "obs_dim": 17 if algo == "ppo2" else 8,
        "obs_variant": "base" if algo == "ppo2" else "brain8",
        "native_dt_minutes": 15.0,
        "control_dt_minutes": 15.0,
        "native_steps_per_action": 1,
    }
    if reference_env:
        meta["reference_env"] = reference_env
    return {
        "name": f"policy_{algo}.pt",
        "algo": algo,
        "meta": meta,
        "error": None,
    }


def _parameters(dt_hours: float) -> dict:
    return {
        **DEFAULT_PARAMETERS,
        "dt": str(dt_hours),
        "billing_mode": "2tc",
    }


def test_benchmark_admits_ppo2_on_its_15_minute_playing_field() -> None:
    checkpoint = _checkpoint("ppo2")
    with patch("bess.evaluation.benchmarking.list_checkpoints", return_value=[checkpoint]):
        selected = _selected_checkpoints([checkpoint["name"]], _parameters(0.25))
    assert selected == [checkpoint]


def test_benchmark_marks_ppo2_incompatible_off_its_playing_field() -> None:
    checkpoint = _checkpoint("ppo2")
    with (
        patch("bess.evaluation.benchmarking.list_checkpoints", return_value=[checkpoint]),
        pytest.raises(ValueError, match="PPO2 requires 15-minute data"),
    ):
        _selected_checkpoints([checkpoint["name"]], _parameters(5.0 / 60.0))


def test_benchmark_rejects_removed_checkpoint_algorithms() -> None:
    checkpoint = _checkpoint("grepo")
    with (
        patch("bess.evaluation.benchmarking.list_checkpoints", return_value=[checkpoint]),
        pytest.raises(ValueError, match="removed/unsupported algorithm"),
    ):
        _selected_checkpoints([checkpoint["name"]], _parameters(0.25))
