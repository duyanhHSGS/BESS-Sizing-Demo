from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import oracle_cache
from benchmark import detect_dt_hours
from settings import GREPO_GAMMA, GREPRO_GAMMA, PPO_GAMMA, PPO_LAMBDA, PRO_GAMMA
from training_datasets import (
    DatasetError,
    detect_resolution_minutes,
    export_training_csv,
    get_dataset_path,
    require_min_days,
)
from training_jobs import Job, JobManager
from weather_forecast import forecast_artifact_path, weather_path, weather_status


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
USER_DATA_DIR = BASE_DIR / "user_data"
PPO_SCRIPT = BASE_DIR / "train_ppo_dataset.py"
GREPO_SCRIPT = BASE_DIR / "train_grepo.py"
GREPRO_SCRIPT = BASE_DIR / "train_grepro.py"
PRO_SCRIPT = BASE_DIR / "train_pro.py"
ALGORITHMS = {"ppo", "grepo", "grepro", "grpo", "pro"}


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


def _bounded_float(
    payload: dict,
    key: str,
    default: float,
    *,
    minimum: float,
    minimum_inclusive: bool = True,
    maximum: float,
) -> float:
    value = _float(payload, key, default)
    minimum_ok = value >= minimum if minimum_inclusive else value > minimum
    if not math.isfinite(value) or not minimum_ok or value > maximum:
        left = "[" if minimum_inclusive else "("
        raise TrainingLaunchError(
            f"{key} must be finite and in {left}{minimum}, {maximum}]"
        )
    return value


def _split_days(payload: dict) -> tuple[int, int]:
    val_days = _int(payload, "val_days", 30)
    test_days = _int(payload, "test_days", 30)
    if val_days < 1:
        raise TrainingLaunchError("validation days must be at least 1")
    if test_days < 1:
        raise TrainingLaunchError("test days must be at least 1")
    return val_days, test_days


def _control_dt_minutes(payload: dict, csv_path: Path) -> int:
    native_minutes = float(detect_resolution_minutes(csv_path))
    requested = _float(payload, "control_dt_minutes", native_minutes)
    ratio = requested / native_minutes
    if (
        not math.isfinite(requested)
        or requested < native_minutes - 1e-9
        or abs(ratio - round(ratio)) > 1e-9
        or abs(30.0 / requested - round(30.0 / requested)) > 1e-9
        or abs(1440.0 / requested - round(1440.0 / requested)) > 1e-9
    ):
        raise TrainingLaunchError(
            "control_dt_minutes must be a native-or-coarser multiple "
            "that divides both 30 minutes and 24 hours"
        )
    return int(round(requested))


def _training_tag(
    payload: dict,
    dataset_id: str,
    e_cap: float,
    p_rated: float,
    algo: str,
    control_dt_minutes: int,
    obs_variant: str = "base",
) -> str:
    explicit = sanitize_tag(payload.get("tag", ""))
    if explicit:
        return explicit
    return sanitize_tag(
        f"{algo}_{dataset_id}_{obs_variant}_{e_cap:.0f}kwh_{p_rated:.0f}kw_dt{control_dt_minutes}m"
    )


def write_training_config(parameters: dict, output_dir: Path = USER_DATA_DIR) -> Path:
    output_dir = ensure_inside_base(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_config.json"
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
        "charge_efficiency": float(parameters.get("charge_efficiency", 0.95)),
        "discharge_efficiency": float(parameters.get("discharge_efficiency", 0.95)),
        "minimum_soc": float(parameters.get("minimum_soc", 0.10)),
        "maximum_soc": float(parameters.get("maximum_soc", 0.90)),
        "required_final_soc": float(parameters.get("required_final_soc", 0.50)),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def build_training_command(
    payload: dict,
    csv_path: Path,
    config_path: Path,
    oracle_cache_path: Path,
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
    obs_variant = str(payload.get("obs_variant", "base")).strip().lower()
    if obs_variant not in {"base", "fc"}:
        raise TrainingLaunchError("obs_variant must be base or fc")
    device = str(payload.get("device", "auto")).strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise TrainingLaunchError("Training device must be auto, cpu, or cuda")
    control_dt_minutes = _control_dt_minutes(payload, csv_path)
    tag = _training_tag(
        payload, dataset_id, e_cap, p_rated, algo, control_dt_minutes, obs_variant
    )
    checkpoint = ensure_inside_base(base_dir / "checkpoints" / f"policy_{tag}.pt", base_dir)
    script = {
        "ppo": PPO_SCRIPT,
        "grepo": GREPO_SCRIPT,
        "grepro": GREPRO_SCRIPT,
        "pro": PRO_SCRIPT,
    }[algo]

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
        "--training-config",
        str(config_path),
        "--oracle-cache",
        str(oracle_cache_path),
        "--control-dt-minutes",
        str(control_dt_minutes),
        "--obs-variant",
        obs_variant,
        "--device",
        device,
    ]
    if obs_variant == "fc":
        status = weather_status(dataset_id)
        if not status.get("ready"):
            raise TrainingLaunchError(status.get("message") or "real weather is not ready")
        weather_file = ensure_inside_base(weather_path(dataset_id), base_dir)
        artifact = ensure_inside_base(forecast_artifact_path(tag), base_dir)
        cmd.extend(["--weather-data", str(weather_file), "--forecast-artifact", str(artifact)])
    if algo == "ppo":
        gamma = _bounded_float(
            payload,
            "gamma",
            PPO_GAMMA,
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
        )
        lambda_value = _bounded_float(
            payload,
            "lambda",
            PPO_LAMBDA,
            minimum=0.0,
            maximum=1.0,
        )
        cmd.extend(
            [
                "--steps",
                str(_int(payload, "steps", 400_000)),
                "--gamma",
                str(gamma),
                "--lambda",
                str(lambda_value),
            ]
        )
    elif algo == "pro":
        gamma = _bounded_float(
            payload,
            "pro_gamma",
            PRO_GAMMA,
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
        )
        cmd.extend(
            [
                "--iters",
                str(_int(payload, "pro_iters", 400)),
                "--gamma",
                str(gamma),
                "--oracle-coef",
                str(_float(payload, "pro_oracle_coef", 1.0)),
                "--oracle-decay",
                str(_float(payload, "pro_oracle_decay", 0.0)),
            ]
        )
    else:
        gamma_key = "grepro_gamma" if algo == "grepro" else "grepo_gamma"
        gamma_default = GREPRO_GAMMA if algo == "grepro" else GREPO_GAMMA
        gamma = _bounded_float(
            payload,
            gamma_key,
            gamma_default,
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
        )
        cmd.extend(
            [
                "--iters",
                str(_int(payload, "grepro_iters" if algo == "grepro" else "iters", 200 if algo == "grepro" else 400)),
                "--group",
                str(_int(payload, "grepro_group" if algo == "grepro" else "group", 6 if algo == "grepro" else 8)),
                "--beta",
                str(_float(payload, "grepro_beta" if algo == "grepro" else "beta", 0.5)),
                "--std",
                str(_float(payload, "grepro_std" if algo == "grepro" else "std", 0.20 if algo == "grepro" else 0.30)),
                "--gamma",
                str(gamma),
            ]
        )
        if algo == "grepro":
            cmd.extend(
                [
                    "--residual-limit",
                    str(_bounded_float(
                        payload, "grepro_residual_limit", 0.05,
                        minimum=0.0, minimum_inclusive=False, maximum=1.0,
                    )),
                    "--forecast-seed",
                    str(_int(payload, "grepro_forecast_seed", 13_0013)),
                    "--forecast-load-sigma",
                    str(_bounded_float(
                        payload, "grepro_forecast_load_sigma", 0.05,
                        minimum=0.0, maximum=1.0,
                    )),
                    "--forecast-pv-sigma",
                    str(_bounded_float(
                        payload, "grepro_forecast_pv_sigma", 0.15,
                        minimum=0.0, maximum=1.0,
                    )),
                ]
            )
    return {"cmd": cmd, "tag": tag, "checkpoint": str(checkpoint), "algo": algo,
            "obs_variant": obs_variant, "device": device}


def training_oracle_parameters(payload: dict, parameters: dict) -> tuple[Path, dict]:
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        raise DatasetError("missing dataset_id")
    source = get_dataset_path(dataset_id)
    _control_dt_minutes(payload, source)
    return source, {
        **parameters,
        "selected_data_csv": source.name,
        "dt": str(detect_dt_hours(source)),
        "battery_capacity_kWh": str(_float(payload, "e_cap_kwh")),
        "battery_power_limit_kW": str(_float(payload, "p_rated_kw")),
    }


def training_oracle_status(payload: dict, parameters: dict) -> dict:
    source, oracle_parameters = training_oracle_parameters(payload, parameters)
    try:
        cache_path, result = oracle_cache.require_cached_oracle(oracle_parameters)
    except oracle_cache.OracleCacheRequired as exc:
        return {"ready": False, "source": source.name, "message": str(exc)}
    return {
        "ready": True,
        "source": source.name,
        "cache": cache_path.name,
        "solved_days": result.get("summary", {}).get("solved_day_count", 0),
        "message": "Exact month-wide Oracle LP cache is ready.",
    }


def start_training(payload: dict, parameters: dict, manager: JobManager) -> tuple[Job, dict]:
    dataset_id = str(payload.get("dataset_id", "")).strip()
    val_days, test_days = _split_days(payload)
    source, oracle_parameters = training_oracle_parameters(payload, parameters)
    algo = str(payload.get("algo", "ppo")).lower()
    required_train_days = 30 if algo == "grepro" else 1
    n_days = require_min_days(source, val_days + test_days + required_train_days)
    oracle_path, _ = oracle_cache.require_cached_oracle(oracle_parameters)

    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = export_training_csv(
        dataset_id,
        USER_DATA_DIR,
        min_days=val_days + test_days + required_train_days,
    )
    config_path = write_training_config(oracle_parameters, USER_DATA_DIR)
    spec = build_training_command(payload, csv_path, config_path, oracle_path)

    env = os.environ.copy()
    env["SIZING_DEMO_CHECKPOINT_DIR"] = str(CHECKPOINT_DIR.resolve())
    job = manager.start_subprocess("train_" + spec["algo"], spec["cmd"], cwd=str(BASE_DIR), env=env)
    return job, {"job_id": job.id, "n_days": n_days, "val_days": val_days, "test_days": test_days, **spec}
