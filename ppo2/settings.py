"""PPO2-owned configuration and tariff helpers.

PPO2 is a fixed 15-minute controller family.  This module is intentionally free
of GUI imports and generic-PPO settings so the root ``ppo2`` package can train,
evaluate, and infer on its own.
"""
from __future__ import annotations

import json
from datetime import date as date_cls
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "checkpoints"

PPO2_DT_HOURS = 0.25
PPO2_STEPS_PER_DAY = 96
PPO2_DEMAND_BLOCK_SLOTS = 2
PPO2_OBS_DIM = 16

PPO2_GAMMA = 1.0
PPO2_LAM_ENERGY = 0.97
PPO2_LAM_PEAK = 0.97
ROLLOUT = PPO2_STEPS_PER_DAY * 30
MIN_MONTH_COVERAGE = 0.8
EVAL_MONTH_COVERAGE = MIN_MONTH_COVERAGE
VAL_MONTHS = 2
TEST_MONTHS = 1
EVAL_EVERY_UPDATES = 20
D_RUN_SHAPING_ORACLE_MARGIN = 0.9
LEARNING_RATE = 3e-4
ACTOR_LR = 3e-5
CRITIC_LR = 3e-4
INIT_STD = 0.15
BC_EPOCHS = 10
CLIP_PENALTY_PER_KWH = 100.0
LAMBDA_ENERGY = PPO2_LAM_ENERGY
LAMBDA_PEAK = PPO2_LAM_PEAK
BC_ACTION_CLIP = 0.95
PPO_CLIP = 0.2
PPO_EPOCHS = 6
PPO_MINIBATCH = 256
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
TARGET_KL = 0.01
BC_LR = 1e-3
BC_MINIBATCH = 256
AUG_LOAD_SIGMA = 0.04
AUG_PV_SIGMA = 0.08
AUG_RHO_LOAD = 0.9
AUG_RHO_PV = 0.9
TORCH_THREADS = 2

# The application launcher reads these defaults instead of owning a second copy
# of PPO2's knobs. Keys intentionally match the existing web payload names.
PPO2_LAUNCH_DEFAULTS = {
    "ppo2_steps": 1_500_000,
    "ppo2_seed": 0,
    "ppo2_rollout": ROLLOUT,
    "ppo2_eval_every": EVAL_EVERY_UPDATES,
    "ppo2_min_month_coverage": MIN_MONTH_COVERAGE,
    "ppo2_val_months": VAL_MONTHS,
    "ppo2_test_months": TEST_MONTHS,
    "ppo2_gamma": PPO2_GAMMA,
    "ppo2_lam_energy": PPO2_LAM_ENERGY,
    "ppo2_lam_peak": PPO2_LAM_PEAK,
    "ppo2_actor_lr": ACTOR_LR,
    "ppo2_critic_lr": CRITIC_LR,
    "ppo2_init_std": INIT_STD,
    "ppo2_clip_penalty": CLIP_PENALTY_PER_KWH,
    "ppo2_bc_epochs": BC_EPOCHS,
    "ppo2_clip": PPO_CLIP,
    "ppo2_epochs": PPO_EPOCHS,
    "ppo2_minibatch": PPO_MINIBATCH,
    "ppo2_ent_coef": ENTROPY_COEF,
    "ppo2_vf_coef": VALUE_COEF,
    "ppo2_target_kl": TARGET_KL,
    "ppo2_shaping_margin": D_RUN_SHAPING_ORACLE_MARGIN,
    "ppo2_aug_load_sigma": AUG_LOAD_SIGMA,
    "ppo2_aug_pv_sigma": AUG_PV_SIGMA,
    "ppo2_aug_rho_load": AUG_RHO_LOAD,
    "ppo2_aug_rho_pv": AUG_RHO_PV,
    "ppo2_bc_lr": BC_LR,
    "ppo2_bc_minibatch": BC_MINIBATCH,
    "ppo2_bc_action_clip": BC_ACTION_CLIP,
    "ppo2_torch_threads": TORCH_THREADS,
}

TOU_RULES: dict[str, bool] = {"sunday_no_peak": False}

_DEFAULTS = {
    "charge_efficiency": 0.9,
    "discharge_efficiency": 0.9,
    "minimum_soc": 0.2,
    "maximum_soc": 0.9,
    "price_peak": 2251.0,
    "price_mid": 1332.0,
    "price_off": 904.0,
    "t_cap": 285000.0,
    "peak_windows": "17:30-22:30",
    "off_windows": "00:00-06:00",
    "sunday_no_peak": True,
}


def _clock_to_step(text: str) -> int:
    hour_text, minute_text = text.strip().split(":", 1)
    minutes = int(hour_text) * 60 + int(minute_text)
    if minutes < 0 or minutes > 24 * 60:
        raise ValueError(f"invalid clock time {text!r}")
    if minutes % 15:
        raise ValueError(f"PPO2 tariff boundaries must align to 15 minutes: {text!r}")
    return (minutes // 15) % PPO2_STEPS_PER_DAY


def _range_steps(text: str) -> list[int]:
    start_text, end_text = text.strip().split("-", 1)
    start = _clock_to_step(start_text)
    raw_end_minutes = int(end_text.strip().split(":", 1)[0]) * 60 + int(end_text.strip().split(":", 1)[1])
    if raw_end_minutes == 24 * 60:
        end = PPO2_STEPS_PER_DAY
    else:
        end = _clock_to_step(end_text)
    if end <= start:
        end += PPO2_STEPS_PER_DAY
    return [step % PPO2_STEPS_PER_DAY for step in range(start, end)]


def build_tariff_windows(peak_ranges: str, off_ranges: str) -> dict[str, list[int] | int]:
    peaks = [_range_steps(item) for item in peak_ranges.split(",") if item.strip()]
    off_steps: list[int] = []
    for item in off_ranges.split(","):
        if item.strip():
            off_steps.extend(_range_steps(item))
    peaks.sort(key=lambda window: window[0])
    if not peaks:
        first_peak: list[int] = []
        second_peak: list[int] = []
    elif len(peaks) == 1:
        first_peak, second_peak = [], peaks[0]
    else:
        first_peak, second_peak = peaks[0], peaks[-1]
    off_steps = sorted(set(off_steps))
    peak_steps = set(first_peak) | set(second_peak)
    off_set = set(off_steps)
    intermediate_steps = [
        step for step in range(PPO2_STEPS_PER_DAY)
        if step not in peak_steps and step not in off_set
    ]
    morning_off_steps = [step for step in off_steps if step < PPO2_STEPS_PER_DAY // 2]
    off_peak_end_step = morning_off_steps[-1] + 1 if morning_off_steps else 0
    first_peak_start = first_peak[0] if first_peak else (second_peak[0] if second_peak else 0)
    second_peak_start = second_peak[0] if second_peak else first_peak_start
    return {
        "W1": first_peak,
        "W2": second_peak,
        "OFF": off_steps,
        "INTER": intermediate_steps,
        "W1_START": first_peak_start,
        "W2_START": second_peak_start,
        "OFF_PEAK_END_STEP": off_peak_end_step,
    }


class PPO2Config:
    """Battery/tariff attribute contract consumed by PPO2Env and PPO2 Oracle."""

    def __init__(self, config: dict):
        self.E_cap = float(config["E_cap_kWh"])
        self.P_rated_nominal = float(config["P_rated_kW"])
        self.eta_ch = float(config["eta_ch"])
        self.eta_dis = float(config["eta_dis"])
        self.eta_RT = self.eta_ch * self.eta_dis
        self.SOC_min = float(config["soc_min"])
        self.SOC_max = float(config["soc_max"])
        self.SOC_chg = self.SOC_max
        self.SOC_safety = float(config.get("soc_safety_buffer", 0.05))
        self.SOC_min_emergency = float(config.get("soc_min_emergency", 0.05))
        self.SOC_SAFETY_BUFFER = self.SOC_safety
        self.peak_windows = str(config.get("peak_windows", _DEFAULTS["peak_windows"]))
        self.off_windows = str(config.get("off_windows", _DEFAULTS["off_windows"]))
        self.dt = float(config.get("dt_hours", PPO2_DT_HOURS))
        if abs(self.dt - PPO2_DT_HOURS) > 1e-12:
            raise ValueError("PPO2 senior-reference mode requires exactly 15-minute data (dt=0.25 h)")
        windows = build_tariff_windows(self.peak_windows, self.off_windows)
        self.W1 = windows["W1"]
        self.W2 = windows["W2"]
        self.INTER = windows["INTER"]
        self.OFF = windows["OFF"]
        self.W1_START = windows["W1_START"]
        self.W2_START = windows["W2_START"]
        self.OFF_PEAK_END_STEP = windows["OFF_PEAK_END_STEP"]
        self.price_peak = float(config["price_peak"])
        self.price_mid = float(config["price_mid"])
        self.price_off = float(config["price_off"])
        self.T_cap = float(config["T_cap"])
        self.P_target_user = float(config.get("P_target_user_kW", 350.0))
        self.FIT_PRICE = float(config.get("FIT_PRICE", 1200.0))
        self.ENABLE_EXPORT = bool(config.get("ENABLE_EXPORT", False))
        self.V_NOMINAL = float(config.get("V_NOMINAL", 1.0))
        self.V_BLACKOUT_TH = float(config.get("V_BLACKOUT_TH", 0.85))
        self.T_DERATE = list(config.get("T_DERATE", [[35, 1.0], [42, 0.7], [45, 0.5], [999, 0.0]]))
        self.spread_peak_off = self.price_peak - self.price_off / self.eta_RT
        self.ARB_ENABLED = self.spread_peak_off > 0


def build_training_ppo2_config(
    e_cap_kwh: float,
    p_rated_kw: float,
    dt_hours: float,
    training_config_path: str | Path,
    *,
    default_billing: str = "2tc",
):
    tariff = json.loads(Path(training_config_path).read_text(encoding="utf-8"))
    cfg = PPO2Config({
        "E_cap_kWh": e_cap_kwh,
        "P_rated_kW": p_rated_kw,
        "eta_ch": float(tariff.get("charge_efficiency", _DEFAULTS["charge_efficiency"])),
        "eta_dis": float(tariff.get("discharge_efficiency", _DEFAULTS["discharge_efficiency"])),
        "soc_min": float(tariff.get("minimum_soc", _DEFAULTS["minimum_soc"])),
        "soc_max": float(tariff.get("maximum_soc", _DEFAULTS["maximum_soc"])),
        "dt_hours": float(dt_hours),
        "price_peak": float(tariff.get("price_peak", _DEFAULTS["price_peak"])),
        "price_mid": float(tariff.get("price_mid", _DEFAULTS["price_mid"])),
        "price_off": float(tariff.get("price_off", _DEFAULTS["price_off"])),
        "T_cap": float(tariff.get("t_cap", _DEFAULTS["t_cap"])),
        "peak_windows": str(tariff.get("peak_windows", _DEFAULTS["peak_windows"])),
        "off_windows": str(tariff.get("off_windows", _DEFAULTS["off_windows"])),
    })
    billing = str(tariff.get("billing_mode", default_billing))
    TOU_RULES["sunday_no_peak"] = bool(tariff.get("sunday_no_peak", False))
    if billing == "tou":
        cfg.T_cap = 0.0
    return cfg, billing


def is_sunday(day) -> bool:
    if not getattr(day, "date_iso", None):
        return False
    return date_cls.fromisoformat(str(day.date_iso)).weekday() == 6


def tariff_vector(cfg) -> np.ndarray:
    values = np.empty(PPO2_STEPS_PER_DAY, dtype=np.float64)
    off = set(cfg.OFF)
    peak = set(cfg.W1) | set(cfg.W2)
    for step in range(PPO2_STEPS_PER_DAY):
        if step in off:
            values[step] = cfg.price_off
        elif step in peak:
            values[step] = cfg.price_peak
        else:
            values[step] = cfg.price_mid
    return values


def tariff_vector_day(cfg, day) -> np.ndarray:
    if TOU_RULES.get("sunday_no_peak") and is_sunday(day):
        values = tariff_vector(cfg)
        peak = set(cfg.W1) | set(cfg.W2)
        for step in peak:
            values[step] = cfg.price_mid
        return values
    return tariff_vector(cfg)


# TODO(PPO2-IQ): after the home move is proven equivalent, change PPO2 learning
# behavior one experiment at a time; never mix IQ changes into architecture moves.
