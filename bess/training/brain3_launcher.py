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
from bess.brain.runtime import load_csv_days, split_billing_periods
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


def _training_values(payload: dict) -> dict[str, int | float]:
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
    return values


def _resolved_training_core(payload: dict, parameters: dict) -> dict[str, Any]:
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
    return {
        "algorithm": algorithm,
        "csv_path": csv_path,
        "native_minutes": native_minutes,
        "control_minutes": control_minutes,
        "native_steps_per_action": int(round(ratio)),
        "decisions_per_day": int(round(1440.0 / control_minutes)),
        "device": device,
        "tag": _tag(payload.get("tag")),
        "values": _training_values(payload),
        "frozen_parameters": {
            **parameters,
            "selected_data_csv": csv_path.name,
            "dt": str(native_minutes / 60.0),
        },
    }


def _period_preview(period, decisions_per_day: int) -> dict[str, Any]:
    first = period.days[0]
    last = period.days[-1]
    return {
        "key": period.key,
        "days": len(period.days),
        "start": first.date_iso or f"day {first.day_index}",
        "end": last.date_iso or f"day {last.day_index}",
        "decisions": len(period.days) * decisions_per_day,
    }


def preview_training(payload: dict, parameters: dict) -> dict[str, Any]:
    resolved = _resolved_training_core(payload, parameters)
    values = resolved["values"]
    warnings: list[str] = []
    periods = split_billing_periods(
        load_csv_days(resolved["csv_path"]), reject_leftover=True, warnings=warnings
    )
    reserve = int(values["validation_periods"]) + int(values["test_periods"])
    if len(periods) <= reserve:
        raise TrainingLaunchError("dataset needs training periods plus positive validation and test periods")
    train_periods = periods[:-reserve]
    validation_periods = periods[-reserve:-int(values["test_periods"])]
    test_periods = periods[-int(values["test_periods"]):]
    decisions_per_day = resolved["decisions_per_day"]

    total_decisions = 0
    cursor = 0
    next_evaluation = int(values["eval_every"])
    evaluation_boundaries: list[dict[str, Any]] = []
    while total_decisions < int(values["steps"]):
        period = train_periods[cursor]
        cursor = (cursor + 1) % len(train_periods)
        total_decisions += len(period.days) * decisions_per_day
        interval_due = total_decisions >= next_evaluation
        budget_due = total_decisions >= int(values["steps"])
        if interval_due or budget_due:
            target = next_evaluation if interval_due else int(values["steps"])
            evaluation_boundaries.append(
                {
                    "trigger": "interval" if interval_due else "final budget",
                    "target_decisions": target,
                    "actual_decisions": total_decisions,
                    "overshoot_decisions": total_decisions - target,
                }
            )
            next_evaluation += int(values["eval_every"])

    hidden_dim = int(values["hidden_dim"])
    effective_learning_start = max(int(values["batch_size"]), int(values["learning_starts"]))
    if int(values["replay_capacity"]) < effective_learning_start:
        warnings.append(
            f"Replay capacity {int(values['replay_capacity']):,} is below effective learning start "
            f"{effective_learning_start:,}; Brain 3 learn() can never run with these settings."
        )
    resume_name = str(payload.get("resume") or "").strip()
    if resume_name:
        warnings.append(
            "Resume selected: dataset split and learner contract are exact, but displayed stop/evaluation "
            "boundaries are the fresh-run schedule; the runner continues saved counters and period cursor."
        )
    return {
        "dataset": {
            "id": str(payload.get("dataset_id") or ""),
            "source": resolved["csv_path"].name,
            "native_dt_minutes": resolved["native_minutes"],
            "control_dt_minutes": resolved["control_minutes"],
            "native_steps_per_action": resolved["native_steps_per_action"],
            "decisions_per_day": decisions_per_day,
            "billing_period_mode": (
                "sequential 30-day fallback"
                if any(period.key.startswith("period-") for period in periods)
                else "complete calendar months"
            ),
            "usable_periods": len(periods),
        },
        "requested_decisions": int(values["steps"]),
        "expected_stop_decisions": total_decisions,
        "effective_learning_start": effective_learning_start,
        "warnings": warnings,
        "split": {
            "training": [_period_preview(period, decisions_per_day) for period in train_periods],
            "validation": [_period_preview(period, decisions_per_day) for period in validation_periods],
            "test": [_period_preview(period, decisions_per_day) for period in test_periods],
        },
        "evaluation_boundaries": evaluation_boundaries,
        "contract": {
            "algorithm": "Classic / vanilla DQN",
            "actions": ["-1 CHARGE", "0 IDLE", "+1 DISCHARGE"],
            "observation_dim": 7,
            "observation_eyes": [
                "time sin",
                "time cos",
                "grid-facing net load / site power scale",
                "SOC normalized inside allowed SOC range",
                "current tariff / maximum configured tariff",
                "running monthly peak / site power scale",
                "working-day flag (0/1)",
            ],
            "network": f"7 -> {hidden_dim} -> {hidden_dim} -> 3",
            "activation": "ReLU after each hidden layer",
            "optimizer": "Adam",
            "loss": "Smooth L1 (Huber)",
            "replay": "Preallocated ring buffer; uniform random sample without replacement inside each batch",
            "updates": "1 gradient update per decision after effective replay warmup",
            "target_network": f"Hard copy every {int(values['target_sync_interval']):,} gradient updates",
            "dqn_features": "Double DQN: no; Dueling: no; Prioritized replay: no; N-step returns: no",
            "epsilon": (
                f"Linear {float(values['epsilon_start']):g} -> {float(values['epsilon_end']):g} "
                f"over {int(values['epsilon_decay_steps']):,} training decisions"
            ),
            "validation": "Greedy argmax; epsilon 0; no exploration; no learning; no replay writes",
            "reward": "One replay reward is the sum of RawWorld - BessWorld VND savings across held native steps",
            "reward_scaling": f"Training target uses reward / {float(values['reward_divisor_vnd']):g}",
            "gamma": f"{float(values['gamma']):g} per decision",
            "checkpoint_rule": "Maximum total validation savings wins the deployment checkpoint",
        },
        "environment": {
            "battery_capacity_kwh": resolved["frozen_parameters"].get("battery_capacity_kWh"),
            "battery_power_kw": resolved["frozen_parameters"].get("battery_power_limit_kW"),
            "minimum_soc": resolved["frozen_parameters"].get("minimum_soc"),
            "maximum_soc": resolved["frozen_parameters"].get("maximum_soc"),
            "required_final_soc": resolved["frozen_parameters"].get("required_final_soc"),
            "charge_efficiency": resolved["frozen_parameters"].get("charge_efficiency"),
            "discharge_efficiency": resolved["frozen_parameters"].get("discharge_efficiency"),
            "battery_wear_vnd_per_kwh": resolved["frozen_parameters"].get("battery_wear_cost"),
            "billing_mode": resolved["frozen_parameters"].get("billing_mode"),
            "cheap_tariff": resolved["frozen_parameters"].get("billing_cheap"),
            "normal_tariff": resolved["frozen_parameters"].get("billing_normal"),
            "expensive_tariff": resolved["frozen_parameters"].get("billing_expensive"),
            "demand_fee_vnd_per_kw": resolved["frozen_parameters"].get("billing_peak_penalty"),
            "cheap_windows": resolved["frozen_parameters"].get("billing_windows_cheap"),
            "expensive_windows": resolved["frozen_parameters"].get("billing_windows_expensive"),
            "sunday_no_expensive": resolved["frozen_parameters"].get("billing_sunday"),
        },
        "device": resolved["device"],
        "resume": resume_name or "Fresh run",
        "schedule_basis": "resume continuation" if resume_name else "fresh run",
        "artifacts": {
            "deployment": f"brain3_{resolved['tag']}.pt",
            "resume": f"brain3_resume_{resolved['tag']}.pt",
            "report": f"brain3_report_{resolved['tag']}.json",
        },
    }


def build_training_command(payload: dict, parameters: dict, *, python_executable: str = sys.executable):
    resolved = _resolved_training_core(payload, parameters)
    algorithm = resolved["algorithm"]
    csv_path = resolved["csv_path"]
    native_minutes = resolved["native_minutes"]
    control_minutes = resolved["control_minutes"]
    device = resolved["device"]
    tag = resolved["tag"]
    values = resolved["values"]
    frozen_parameters = resolved["frozen_parameters"]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    parameters_path = CHECKPOINT_DIR / f"brain3_parameters_{tag}.json"
    parameters_path.write_text(json.dumps(frozen_parameters, indent=2), encoding="utf-8")
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
