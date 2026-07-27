from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from training_datasets import DatasetError, export_training_csv, get_dataset_path, require_min_days
from training_jobs import Job, JobManager


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
USER_DATA_DIR = BASE_DIR / "user_data"
PPO_SCRIPT = BASE_DIR / "train_ppo_dataset.py"
GREPO_SCRIPT = BASE_DIR / "train_grepo.py"
ALGORITHMS = {"ppo", "grepo", "grpo"}


class TrainingLaunchError(ValueError):
    pass


class UnsupportedAlgorithm(TrainingLaunchError):
    pass


def sanitize_tag(raw: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw or "")).strip("_").lower()
    return tag


def ensure_inside_base(path: Path, base_dir: Path = BASE_DIR) -> Path:
    resolved = path.resolve()
    base = base_dir.resolve()
    if resolved != base and base not in resolved.parents:
        raise TrainingLaunchError(f"path escapes Sizing_Demo: {resolved}")
    return resolved


def _float(payload: dict, key: str, default: float | None = None) -> float:
    value = payload.get(key, default)
    if value is None or value == "":
        raise TrainingLaunchError(f"missing {key}")
    return float(value)


def _int(payload: dict, key: str, default: int) -> int:
    value = payload.get(key, default)
    if value is None or value == "":
        return default
    return int(value)


def _split_days(payload: dict) -> tuple[int, int]:
    val_days = _int(payload, "val_days", 30)
    test_days = _int(payload, "test_days", 30)
    if val_days < 1:
        raise TrainingLaunchError("validation days must be at least 1")
    if test_days < 1:
        raise TrainingLaunchError("test days must be at least 1")
    return val_days, test_days


def _training_tag(payload: dict, dataset_id: str, e_cap: float, p_rated: float, algo: str) -> str:
    explicit = sanitize_tag(payload.get("tag", ""))
    if explicit:
        return explicit
    return sanitize_tag(f"{algo}_{dataset_id}_{e_cap:.0f}kwh_{p_rated:.0f}kw")


def write_tariff_config(parameters: dict, output_dir: Path = USER_DATA_DIR) -> Path:
    output_dir = ensure_inside_base(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "tariff_config.json"
    data = {
        "price_peak": float(parameters.get("billing_expensive", 2759)),
        "price_mid": float(parameters.get("billing_normal", 1485)),
        "price_off": float(parameters.get("billing_cheap", 982)),
        "t_cap": float(parameters.get("billing_peak_penalty", 235414)),
        "billing_mode": parameters.get("billing_mode", "2tc"),
        "peak_windows": parameters.get("billing_windows_expensive", "09:30-11:30,17:00-20:00"),
        "off_windows": parameters.get("billing_windows_cheap", "00:00-04:00,22:00-24:00"),
        "sunday_no_peak": bool(parameters.get("billing_sunday", False)),
        "realization": float(parameters.get("billing_real_saving_factor", 0.6)),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def build_training_command(
    payload: dict,
    csv_path: Path,
    tariff_path: Path,
    python_executable: str = sys.executable,
    base_dir: Path = BASE_DIR,
) -> dict:
    algo = str(payload.get("algo", "ppo")).lower()
    if algo not in ALGORITHMS:
        raise TrainingLaunchError(f"unknown algorithm: {algo}")
    if algo == "grpo":
        raise UnsupportedAlgorithm("GRPO is not implemented in this repo yet.")

    e_cap = _float(payload, "e_cap_kwh")
    p_rated = _float(payload, "p_rated_kw")
    val_days, test_days = _split_days(payload)
    dataset_id = str(payload.get("dataset_id", "dataset"))
    tag = _training_tag(payload, dataset_id, e_cap, p_rated, algo)
    checkpoint = ensure_inside_base(base_dir / "checkpoints" / f"policy_{tag}.pt", base_dir)
    script = PPO_SCRIPT if algo == "ppo" else GREPO_SCRIPT

    cmd = [
        python_executable,
        "-X",
        "utf8",
        "-u",
        str(ensure_inside_base(script, base_dir)),
        "--csv",
        str(ensure_inside_base(csv_path, base_dir)),
        "--e-cap",
        str(e_cap),
        "--p-rated",
        str(p_rated),
        "--tag",
        tag,
        "--val-days",
        str(val_days),
        "--test-days",
        str(test_days),
    ]
    if algo == "ppo":
        cmd.extend(["--steps", str(_int(payload, "steps", 400_000)), "--tariff-json", str(tariff_path)])
    else:
        cmd.extend(
            [
                "--iters",
                str(_int(payload, "iters", 400)),
                "--group",
                str(_int(payload, "group", 8)),
                "--beta",
                str(_float(payload, "beta", 0.5)),
                "--std",
                str(_float(payload, "std", 0.30)),
            ]
        )
    return {"cmd": cmd, "tag": tag, "checkpoint": str(checkpoint), "algo": algo}


def start_training(payload: dict, parameters: dict, manager: JobManager) -> tuple[Job, dict]:
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        raise DatasetError("missing dataset_id")
    val_days, test_days = _split_days(payload)
    source = get_dataset_path(dataset_id)
    n_days = require_min_days(source, val_days + test_days + 1)

    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = export_training_csv(dataset_id, USER_DATA_DIR, min_days=val_days + test_days + 1)
    tariff_path = write_tariff_config(parameters, USER_DATA_DIR)
    spec = build_training_command(payload, csv_path, tariff_path)

    env = os.environ.copy()
    env["SIZING_DEMO_CHECKPOINT_DIR"] = str(CHECKPOINT_DIR.resolve())
    job = manager.start_subprocess("train_" + spec["algo"], spec["cmd"], cwd=str(BASE_DIR), env=env)
    return job, {"job_id": job.id, "n_days": n_days, "val_days": val_days, "test_days": test_days, **spec}
