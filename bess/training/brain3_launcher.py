"""Validate and launch the single Brain 3 training subprocess."""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from bess.paths import PROJECT_ROOT
from bess.training.training_datasets import detect_resolution_minutes, get_dataset_path
from bess.training.training_jobs import JobManager

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TRAINING_MODULES = {"brain3_dqn": "bess.training.runners.train_brain3"}


class TrainingLaunchError(ValueError):
    pass


class UnsupportedAlgorithm(TrainingLaunchError):
    pass


def _tag(value: Any) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "brain3")).strip("_").lower()
    if not clean:
        raise TrainingLaunchError("training tag is empty after sanitization")
    return clean


def _int(payload: dict, key: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise TrainingLaunchError(f"{key} must be an integer") from exc
    if value < minimum:
        raise TrainingLaunchError(f"{key} must be at least {minimum}")
    return value


def _float(payload: dict, key: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise TrainingLaunchError(f"{key} must be numeric") from exc
    if not math.isfinite(value) or value < minimum:
        raise TrainingLaunchError(f"{key} must be finite and at least {minimum}")
    return value


def build_training_command(payload: dict, parameters: dict, *, python_executable: str = sys.executable):
    algorithm = str(payload.get("algo", "brain3_dqn")).lower()
    if algorithm != "brain3_dqn":
        raise UnsupportedAlgorithm("Brain 3 DQN is the only supported training algorithm")
    csv_path = get_dataset_path(str(payload.get("dataset_id") or ""))
    native_minutes = float(detect_resolution_minutes(csv_path))
    control_minutes = _float(payload, "control_dt_minutes", native_minutes, native_minutes)
    ratio = control_minutes / native_minutes
    if not math.isclose(ratio, round(ratio), abs_tol=1e-9) or not math.isclose(
        1440.0 / control_minutes, round(1440.0 / control_minutes), abs_tol=1e-9
    ):
        raise TrainingLaunchError("control_dt_minutes must be a native multiple that divides 24 hours")
    device = str(payload.get("device", "cpu")).lower()
    if device not in {"cpu", "cuda"}:
        raise TrainingLaunchError("device must be cpu or cuda")
    tag = _tag(payload.get("tag"))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_parameters = {**parameters, "selected_data_csv": csv_path.name, "dt": str(native_minutes / 60.0)}
    parameters_path = CHECKPOINT_DIR / f"brain3_parameters_{tag}.json"
    parameters_path.write_text(json.dumps(frozen_parameters, indent=2), encoding="utf-8")
    values = {
        "steps": _int(payload, "steps", 100_000),
        "eval_every": _int(payload, "eval_every", 10_000),
        "validation_periods": _int(payload, "validation_periods", 1),
        "test_periods": _int(payload, "test_periods", 1),
        "hidden_dim": _int(payload, "hidden_dim", 128),
        "batch_size": _int(payload, "batch_size", 256),
        "replay_capacity": _int(payload, "replay_capacity", 100_000),
        "learning_starts": _int(payload, "learning_starts", 1_024, 0),
        "target_sync_interval": _int(payload, "target_sync_interval", 1_000),
        "epsilon_decay_steps": _int(payload, "epsilon_decay_steps", 100_000),
        "seed": _int(payload, "seed", 0, 0),
        "gamma": _float(payload, "gamma", 0.99),
        "learning_rate": _float(payload, "learning_rate", 1e-3, 1e-12),
        "epsilon_start": _float(payload, "epsilon_start", 1.0),
        "epsilon_end": _float(payload, "epsilon_end", 0.05),
        "reward_divisor_vnd": _float(payload, "reward_divisor_vnd", 1_000_000.0, 1e-12),
        "gradient_clip_norm": _float(payload, "gradient_clip_norm", 10.0, 1e-12),
    }
    if not 0 <= values["gamma"] <= 1 or not 0 <= values["epsilon_end"] <= values["epsilon_start"] <= 1:
        raise TrainingLaunchError("gamma and epsilon values must be inside [0, 1]")
    command = [
        python_executable, "-X", "utf8", "-u", "-m", TRAINING_MODULES[algorithm],
        "--csv", str(csv_path), "--parameters", str(parameters_path),
        "--output-dir", str(CHECKPOINT_DIR), "--tag", tag,
        "--control-dt-minutes", str(control_minutes), "--device", device,
    ]
    for key, value in values.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    resume = str(payload.get("resume") or "").strip()
    if resume:
        resume_name = Path(resume).name
        resume_path = CHECKPOINT_DIR / resume_name
        if resume_name != resume or not resume_name.startswith("brain3_resume_") or not resume_path.is_file():
            raise TrainingLaunchError("resume must name a local Brain 3 resume checkpoint")
        command.extend(["--resume", str(resume_path)])
    settings = {
        "schema_version": 1,
        "algorithm": algorithm,
        "source_csv": str(csv_path),
        "native_dt_minutes": native_minutes,
        "control_dt_minutes": control_minutes,
        "payload": payload,
        "parameters": frozen_parameters,
        "command": command,
    }
    (CHECKPOINT_DIR / f"training_settings_{tag}.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    return command, settings


def start_training(payload: dict, parameters: dict, manager: JobManager):
    command, settings = build_training_command(payload, parameters)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    job = manager.start_subprocess("brain3_dqn", command, cwd=str(PROJECT_ROOT), env=environment)
    return job, {"job_id": job.id, "algorithm": "brain3_dqn", "settings": settings}
