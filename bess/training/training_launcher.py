from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

from bess.agents import SUPPORTED_POLICY_ALGORITHMS
from bess.core.common import ensure_inside_directory, validate_control_interval_minutes
from bess.core.settings import (
    PPO2_GAMMA,
    PPO2_LAM_ENERGY,
    PPO2_LAM_PEAK,
    PPO_FIT_CONTROL_DT_MINUTES,
    PPO_GAMMA,
    PPO_LAMBDA,
    PPO_TUNABLE_DEFAULTS,
)
from bess.evaluation.benchmark import detect_dt_hours
from bess.evaluation.oracle import oracle_cache
from bess.paths import PROJECT_ROOT
from bess.training.training_datasets import (
    DatasetError,
    detect_resolution_minutes,
    export_training_csv,
    get_dataset_path,
    require_min_days,
)
from bess.training.training_jobs import Job, JobManager

BASE_DIR = PROJECT_ROOT
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
USER_DATA_DIR = BASE_DIR / "user_data"
PPO_SCRIPT = BASE_DIR / "bess" / "training" / "runners" / "train_ppo_dataset.py"
PPO2_SCRIPT = BASE_DIR / "bess" / "training" / "runners" / "train_ppo2_dataset.py"
TRAINING_MODULES = {
    "ppo": "bess.training.runners.train_ppo_dataset",
    "ppo2": "bess.training.runners.train_ppo2_dataset",
}
ALGORITHMS = SUPPORTED_POLICY_ALGORITHMS


class TrainingLaunchError(ValueError):
    pass


class UnsupportedAlgorithm(TrainingLaunchError):
    pass


def sanitize_tag(raw: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw or "")).strip("_").lower()
    return tag


def ensure_inside_base(path: Path, base_dir: Path = BASE_DIR) -> Path:
    try:
        return ensure_inside_directory(path, base_dir, label="Sizing_Demo")
    except ValueError as exc:
        raise TrainingLaunchError(str(exc)) from exc


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


def _bool(payload: dict, key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise TrainingLaunchError(f"{key} must be true or false")


def _bounded_int(
    payload: dict,
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = _int(payload, key, default)
    if value < minimum or (maximum is not None and value > maximum):
        upper = f", {maximum}" if maximum is not None else ""
        raise TrainingLaunchError(f"{key} must be an integer in [{minimum}{upper}]")
    return value


def _bounded_float_list(
    payload: dict,
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> str:
    raw = str(payload.get(key, default) if payload.get(key, default) not in {None, ""} else default)
    values = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError as exc:
            raise TrainingLaunchError(f"{key} must be a comma-separated list of numbers") from exc
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise TrainingLaunchError(f"{key} values must be finite and in [{minimum}, {maximum}]")
        values.append(value)
    if not values:
        raise TrainingLaunchError(f"{key} must contain at least one value")
    return ",".join(format(value, ".12g") for value in values)


def _int_list(payload: dict, key: str) -> str:
    raw = str(payload.get(key, "") or "").strip()
    if not raw:
        return ""
    values = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except ValueError as exc:
            raise TrainingLaunchError(f"{key} must be a comma-separated list of integers") from exc
    if not values:
        return ""
    if len(values) > 64:
        raise TrainingLaunchError(f"{key} supports at most 64 seeds")
    return ",".join(str(value) for value in values)


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


def _split_months(payload: dict) -> tuple[int, int, int]:
    # IQ-53: use a fixed-size 5/1/1 billing-month window, but let the runner slide
    # that window dynamically over the latest complete measured months.
    # TODO(IQ-53): revisit the 5/1/1 counts only after clean-data unseen-month evidence.
    train_months = _int(payload, "train_months", 5)
    val_months = _int(payload, "val_months", 1)
    test_months = _int(payload, "test_months", 1)
    if train_months < 1:
        raise TrainingLaunchError("training months must be at least 1")
    if val_months < 1:
        raise TrainingLaunchError("validation months must be at least 1")
    if test_months < 1:
        raise TrainingLaunchError("test months must be at least 1")
    return train_months, val_months, test_months


def _control_dt_minutes(payload: dict, csv_path: Path) -> int:
    native_minutes = float(detect_resolution_minutes(csv_path))
    algo = str(payload.get("algo", "ppo")).strip().lower()
    requested = (
        PPO_FIT_CONTROL_DT_MINUTES
        if algo == "ppo"
        else _float(payload, "control_dt_minutes", native_minutes)
    )
    try:
        validate_control_interval_minutes(native_minutes, requested)
    except ValueError as exc:
        raise TrainingLaunchError(
            "control_dt_minutes must be a native-or-coarser multiple "
            "that divides both 30 minutes and 24 hours"
        ) from exc
    return round(requested)


def _training_tag(
    payload: dict,
    dataset_id: str,
    e_cap: float,
    p_rated: float,
    algo: str,
    control_dt_minutes: int,
    obs_variant: str = "brain7",
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
        "battery_wear_cost": float(parameters.get("battery_wear_cost", 0.0)),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_training_run_settings(
    spec: dict,
    payload: dict,
    resolved_parameters: dict,
    csv_path: Path,
    config_path: Path,
    output_dir: Path = CHECKPOINT_DIR,
) -> Path:
    """Freeze the exact launch inputs beside the checkpoint/report artifacts."""
    output_dir = ensure_inside_base(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"training_settings_{spec['tag']}.json"
    training_config = json.loads(config_path.read_text(encoding="utf-8"))
    data = {
        "version": 1,
        "algorithm": spec["algo"],
        "tag": spec["tag"],
        "checkpoint": Path(spec["checkpoint"]).name,
        "dataset_export": str(csv_path),
        "training_request": payload,
        "resolved_control_dt_minutes": spec.get("control_dt_minutes"),
        "resolved_training_config": training_config,
        "resolved_sizing_parameters": resolved_parameters,
        "runner_command": spec["cmd"],
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
        raise UnsupportedAlgorithm(
            f"unsupported algorithm: {algo}; this repository supports only PPO and PPO2"
        )

    e_cap = _float(payload, "e_cap_kwh")
    p_rated = _float(payload, "p_rated_kw")
    if algo == "ppo2":
        train_months, val_months, test_months = 0, 0, 0  # PPO2 owns its separate calendar split.
    else:
        train_months, val_months, test_months = _split_months(payload)
    dataset_id = str(payload.get("dataset_id", "dataset"))
    default_obs_variant = "base" if algo == "ppo2" else "brain7"
    obs_variant = str(payload.get("obs_variant", default_obs_variant)).strip().lower()
    if algo == "ppo2":
        if obs_variant != "base":
            raise TrainingLaunchError(
                "PPO2 senior-reference mode requires obs_variant=base"
            )
    elif obs_variant != "brain7":
        raise TrainingLaunchError(
            "The canonical BrainEnv has exactly seven eyes; use obs_variant=brain7. "
            "Legacy base/fc observation contracts were removed."
        )
    device = str(payload.get("device", "auto")).strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise TrainingLaunchError("Training device must be auto, cpu, or cuda")
    native_dt_minutes = float(detect_resolution_minutes(csv_path))
    control_dt_minutes = _control_dt_minutes(payload, csv_path)
    if algo == "ppo2" and (
        not math.isclose(native_dt_minutes, 15.0, rel_tol=0.0, abs_tol=1e-9)
        or not math.isclose(control_dt_minutes, 15.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise TrainingLaunchError(
            "PPO2 senior-reference mode requires the dataset itself and control interval to be exactly 15 minutes"
        )
    if algo == "ppo2" and device == "cuda":
        raise TrainingLaunchError("PPO2 senior-reference mode is CPU-only")
    tag = _training_tag(
        payload, dataset_id, e_cap, p_rated, algo, control_dt_minutes, obs_variant
    )
    checkpoint = ensure_inside_base(base_dir / "checkpoints" / f"policy_{tag}.pt", base_dir)
    module = TRAINING_MODULES[algo]

    cmd = [
        python_executable,
        "-X",
        "utf8",
        "-u",
        "-m",
        module,
        "--csv",
        str(ensure_inside_base(csv_path, base_dir)),
        "--e-cap",
        str(e_cap),
        "--p-rated",
        str(p_rated),
        "--tag",
        tag,
        "--val-months",
        str(val_months),
        "--test-months",
        str(test_months),
        "--training-config",
        str(config_path),
        "--control-dt-minutes",
        str(control_dt_minutes),
        "--obs-variant",
        obs_variant,
        "--device",
        device,
    ]
    if algo != "ppo2":
        cmd.extend(["--oracle-cache", str(oracle_cache_path)])
    if algo == "ppo":
        cmd.extend(["--train-months", str(train_months)])
        defaults = PPO_TUNABLE_DEFAULTS
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
        learning_rate = _bounded_float(
            payload,
            "learning_rate",
            defaults["learning_rate"],
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
        )
        exploration_lr_multiplier = _bounded_float(
            payload,
            "exploration_lr_multiplier",
            defaults["exploration_lr_multiplier"],
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1000.0,
        )
        soc_edge_log_std_penalty = _bounded_float(
            payload,
            "soc_edge_log_std_penalty",
            defaults["soc_edge_log_std_penalty"],
            minimum=0.0,
            maximum=5.0,
        )
        ppo_clip = _bounded_float(
            payload,
            "ppo_clip",
            defaults["ppo_clip"],
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
        )
        entropy_coef = _bounded_float(
            payload,
            "entropy_coef",
            defaults["entropy_coef"],
            minimum=0.0,
            maximum=100.0,
        )
        value_coef = _bounded_float(
            payload,
            "value_coef",
            defaults["value_coef"],
            minimum=0.0,
            maximum=100.0,
        )
        target_kl = _bounded_float(
            payload,
            "target_kl",
            defaults["target_kl"],
            minimum=0.0,
            minimum_inclusive=False,
            maximum=100.0,
        )
        actor_grad_clip = _bounded_float(
            payload,
            "actor_grad_clip",
            defaults["actor_grad_clip"],
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1e6,
        )
        critic_grad_clip = _bounded_float(
            payload,
            "critic_grad_clip",
            defaults["critic_grad_clip"],
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1e6,
        )
        initial_log_std = _bounded_float(
            payload,
            "initial_log_std",
            defaults["initial_log_std"],
            minimum=-20.0,
            maximum=5.0,
        )
        ppo_start_log_std = _bounded_float(
            payload,
            "ppo_start_log_std",
            defaults["ppo_start_log_std"],
            minimum=-20.0,
            maximum=5.0,
        )
        mismatch_scale = _bounded_float(
            payload,
            "action_mismatch_shaping_scale",
            defaults["action_mismatch_shaping_scale"],
            minimum=0.0,
            maximum=100.0,
        )
        oracle_bc_lr = _bounded_float(
            payload,
            "oracle_bc_learning_rate",
            defaults["oracle_bc_learning_rate"],
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
        )
        oracle_bc_target_mse = _bounded_float(
            payload,
            "oracle_bc_target_mse",
            defaults["oracle_bc_target_mse"],
            minimum=0.0,
            maximum=1e9,
        )
        cmd.extend(
            [
                "--steps",
                str(_bounded_int(payload, "steps", defaults["steps"], minimum=1)),
                "--seed",
                str(_int(payload, "seed", defaults["seed"])),
                "--gamma",
                str(gamma),
                "--lambda",
                str(lambda_value),
                "--learning-rate",
                str(learning_rate),
                "--exploration-lr-multiplier",
                str(exploration_lr_multiplier),
                "--soc-edge-log-std-penalty",
                str(soc_edge_log_std_penalty),
                "--ppo-clip",
                str(ppo_clip),
                "--ppo-epochs",
                str(_bounded_int(payload, "ppo_epochs", defaults["ppo_epochs"], minimum=1, maximum=1000)),
                "--minibatch",
                str(_bounded_int(payload, "minibatch", defaults["minibatch"], minimum=1, maximum=1_000_000)),
                "--entropy-coef",
                str(entropy_coef),
                "--value-coef",
                str(value_coef),
                "--target-kl",
                str(target_kl),
                "--actor-grad-clip",
                str(actor_grad_clip),
                "--critic-grad-clip",
                str(critic_grad_clip),
                "--hidden-size",
                str(_bounded_int(payload, "hidden_size", defaults["hidden_size"], minimum=1, maximum=4096)),
                "--recurrent-sequence-length",
                str(_bounded_int(payload, "recurrent_sequence_length", defaults["recurrent_sequence_length"], minimum=1, maximum=1_000_000)),
                "--initial-log-std",
                str(initial_log_std),
                "--ppo-start-log-std",
                str(ppo_start_log_std),
                "--validate-every-updates",
                str(_bounded_int(payload, "validate_every_updates", defaults["validate_every_updates"], minimum=1, maximum=1_000_000)),
                "--challenger-reset-patience",
                str(_bounded_int(payload, "challenger_reset_patience", defaults["challenger_reset_patience"], minimum=1, maximum=1_000_000)),
                "--action-mismatch-shaping-scale",
                str(mismatch_scale),
                "--oracle-actor-bc-max-epochs",
                str(_bounded_int(payload, "oracle_actor_bc_max_epochs", defaults["oracle_actor_bc_max_epochs"], minimum=0, maximum=1_000_000)),
                "--oracle-bc-max-epochs",
                str(_bounded_int(payload, "oracle_bc_max_epochs", defaults["oracle_bc_max_epochs"], minimum=0, maximum=1_000_000)),
                "--oracle-bc-learning-rate",
                str(oracle_bc_lr),
                "--oracle-bc-minibatch",
                str(_bounded_int(payload, "oracle_bc_minibatch", defaults["oracle_bc_minibatch"], minimum=1, maximum=1_000_000)),
                "--oracle-bc-target-mse",
                str(oracle_bc_target_mse),
                "--log-every-updates",
                str(_bounded_int(payload, "log_every_updates", defaults["log_every_updates"], minimum=1, maximum=1_000_000)),
                "--torch-threads",
                str(_bounded_int(payload, "torch_threads", defaults["torch_threads"], minimum=1, maximum=128)),
            ]
        )
        recurrent_enabled = _bool(
            payload,
            "recurrent_enabled",
            defaults["recurrent_enabled"],
        )
        challenger_resets_enabled = _bool(
            payload,
            "challenger_resets_enabled",
            defaults["challenger_resets_enabled"],
        )
        reset_optimizer_on_reanchor = _bool(
            payload,
            "reset_optimizer_on_reanchor",
            defaults["reset_optimizer_on_reanchor"],
        )
        preserve_critic_on_reanchor = _bool(
            payload,
            "preserve_critic_on_reanchor",
            defaults["preserve_critic_on_reanchor"],
        )
        if preserve_critic_on_reanchor and not reset_optimizer_on_reanchor:
            raise TrainingLaunchError(
                "preserve_critic_on_reanchor requires reset_optimizer_on_reanchor"
            )
        cmd.append(
            "--recurrent-enabled"
            if recurrent_enabled
            else "--no-recurrent-enabled"
        )
        cmd.append(
            "--challenger-resets-enabled"
            if challenger_resets_enabled
            else "--no-challenger-resets-enabled"
        )
        cmd.append(
            "--reset-optimizer-on-reanchor"
            if reset_optimizer_on_reanchor
            else "--no-reset-optimizer-on-reanchor"
        )
        cmd.append(
            "--preserve-critic-on-reanchor"
            if preserve_critic_on_reanchor
            else "--no-preserve-critic-on-reanchor"
        )
        cmd.append(
            "--oracle-bc-enabled"
            if _bool(payload, "oracle_bc_enabled", defaults["oracle_bc_enabled"])
            else "--no-oracle-bc-enabled"
        )
    elif algo == "ppo2":
        gamma = _bounded_float(
            payload, "ppo2_gamma", PPO2_GAMMA,
            minimum=0.0, minimum_inclusive=False, maximum=1.0,
        )
        if not math.isclose(gamma, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise TrainingLaunchError("PPO2 senior-reference mode requires ppo2_gamma=1.0")
        lam_energy = _bounded_float(
            payload, "ppo2_lam_energy", PPO2_LAM_ENERGY,
            minimum=0.0, maximum=1.0,
        )
        lam_peak = _bounded_float_list(
            payload, "ppo2_lam_peak", PPO2_LAM_PEAK,
            minimum=0.0, maximum=1.0,
        )
        seeds = _int_list(payload, "ppo2_seeds")
        cmd.extend(
            [
                "--steps", str(_bounded_int(payload, "ppo2_steps", 1_500_000, minimum=1)),
                "--seed", str(_int(payload, "ppo2_seed", 0)),
                "--rollout", str(_bounded_int(payload, "ppo2_rollout", 2_880, minimum=1)),
                "--eval-every", str(_bounded_int(payload, "ppo2_eval_every", 20, minimum=1)),
                "--min-month-coverage", str(_bounded_float(payload, "ppo2_min_month_coverage", 0.8, minimum=0.01, maximum=1.0)),
                "--val-months", str(_bounded_int(payload, "ppo2_val_months", 2, minimum=1, maximum=24)),
                "--test-months", str(_bounded_int(payload, "ppo2_test_months", 1, minimum=1, maximum=24)),
                "--gamma", str(gamma),
                "--lambda-energy", str(lam_energy),
                "--lambda-peak", lam_peak,
                "--actor-lr", str(_bounded_float(payload, "ppo2_actor_lr", 3e-5, minimum=0.0, minimum_inclusive=False, maximum=1.0)),
                "--critic-lr", str(_bounded_float(payload, "ppo2_critic_lr", 3e-4, minimum=0.0, minimum_inclusive=False, maximum=1.0)),
                "--init-std", str(_bounded_float(payload, "ppo2_init_std", 0.15, minimum=0.0, minimum_inclusive=False, maximum=5.0)),
                "--clip-penalty", str(_bounded_float(payload, "ppo2_clip_penalty", 100.0, minimum=0.0, maximum=1e9)),
                "--bc-epochs", str(_bounded_int(payload, "ppo2_bc_epochs", 10, minimum=0, maximum=10_000)),
                "--ppo-clip", str(_bounded_float(payload, "ppo2_clip", 0.2, minimum=0.0, minimum_inclusive=False, maximum=1.0)),
                "--ppo-epochs", str(_bounded_int(payload, "ppo2_epochs", 6, minimum=1, maximum=1_000)),
                "--minibatch", str(_bounded_int(payload, "ppo2_minibatch", 256, minimum=1, maximum=1_000_000)),
                "--entropy-coef", str(_bounded_float(payload, "ppo2_ent_coef", 0.01, minimum=0.0, maximum=100.0)),
                "--value-coef", str(_bounded_float(payload, "ppo2_vf_coef", 0.5, minimum=0.0, maximum=100.0)),
                "--target-kl", str(_bounded_float(payload, "ppo2_target_kl", 0.01, minimum=0.0, minimum_inclusive=False, maximum=100.0)),
                "--shaping-margin", str(_bounded_float(payload, "ppo2_shaping_margin", 0.9, minimum=0.0, maximum=1.0)),
                "--aug-load-sigma", str(_bounded_float(payload, "ppo2_aug_load_sigma", 0.04, minimum=0.0, maximum=2.0)),
                "--aug-pv-sigma", str(_bounded_float(payload, "ppo2_aug_pv_sigma", 0.08, minimum=0.0, maximum=2.0)),
                "--aug-rho-load", str(_bounded_float(payload, "ppo2_aug_rho_load", 0.9, minimum=-0.999999, maximum=0.999999)),
                "--aug-rho-pv", str(_bounded_float(payload, "ppo2_aug_rho_pv", 0.9, minimum=-0.999999, maximum=0.999999)),
                "--bc-lr", str(_bounded_float(payload, "ppo2_bc_lr", 1e-3, minimum=0.0, minimum_inclusive=False, maximum=1.0)),
                "--bc-minibatch", str(_bounded_int(payload, "ppo2_bc_minibatch", 256, minimum=1, maximum=1_000_000)),
                "--bc-action-clip", str(_bounded_float(payload, "ppo2_bc_action_clip", 0.95, minimum=0.0, minimum_inclusive=False, maximum=1.0)),
                "--torch-threads", str(_bounded_int(payload, "ppo2_torch_threads", 2, minimum=1, maximum=128)),
            ]
        )
        if seeds:
            cmd.extend(["--seeds", seeds])
        if payload.get("ppo2_fit_test") is True:
            cmd.append("--fit-test")
    return {
        "cmd": cmd,
        "tag": tag,
        "checkpoint": str(checkpoint),
        "algo": algo,
        "obs_variant": obs_variant,
        "device": device,
        "control_dt_minutes": control_dt_minutes,
    }


def training_oracle_parameters(payload: dict, parameters: dict) -> tuple[Path, dict]:
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        raise DatasetError("missing dataset_id")
    source = get_dataset_path(dataset_id)
    _control_dt_minutes(payload, source)
    oracle_parameters = {
        **parameters,
        "selected_data_csv": source.name,
        "dt": str(detect_dt_hours(source)),
        "battery_capacity_kWh": str(_float(payload, "e_cap_kwh")),
        "battery_power_limit_kW": str(_float(payload, "p_rated_kw")),
    }
    if str(payload.get("algo", "")).lower() == "ppo":
        wear_cost = _float(payload, "battery_wear_cost")
        if not math.isfinite(wear_cost) or wear_cost < 0.0:
            raise TrainingLaunchError("PPO battery_wear_cost must be finite and >= 0")
        oracle_parameters["battery_wear_cost"] = str(wear_cost)
    return source, oracle_parameters


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
    algo = str(payload.get("algo", "ppo")).lower()
    source, oracle_parameters = training_oracle_parameters(payload, parameters)
    if algo == "ppo2":
        train_months, val_months, test_months = 0, 0, 0
        required_train_days = 1
        n_days = require_min_days(source, required_train_days)
        oracle_path = Path("")  # PPO2 builds the senior fixed-block month LP internally.
    else:
        train_months, val_months, test_months = _split_months(payload)
        # Cheap launcher guard only. The runner validates actual date_iso calendar
        # coverage because day count alone cannot prove distinct billing months.
        required_train_days = train_months + val_months + test_months
        n_days = require_min_days(source, required_train_days)
        oracle_path, _ = oracle_cache.require_cached_oracle(oracle_parameters)

    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = export_training_csv(
        dataset_id,
        USER_DATA_DIR,
        # Generic PPO validates whole calendar-month holdouts in the runner;
        # this export guard is only the cheapest impossible-input rejection.
        min_days=required_train_days,
    )
    config_path = write_training_config(oracle_parameters, USER_DATA_DIR)
    spec = build_training_command(payload, csv_path, config_path, oracle_path)
    settings_path = write_training_run_settings(
        spec, payload, oracle_parameters, csv_path, config_path, CHECKPOINT_DIR
    )

    env = os.environ.copy()
    env["SIZING_DEMO_CHECKPOINT_DIR"] = str(CHECKPOINT_DIR.resolve())
    job = manager.start_subprocess("train_" + spec["algo"], spec["cmd"], cwd=str(BASE_DIR), env=env)
    return job, {
        "job_id": job.id,
        "n_days": n_days,
        "train_months": train_months,
        "val_months": val_months,
        "test_months": test_months,
        "settings": str(settings_path),
        **spec,
    }
