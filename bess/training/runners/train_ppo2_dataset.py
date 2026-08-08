"""Train PPO2 with the senior reference project's algorithm and environment.

PPO2 is intentionally a controlled reference port for A/B comparison against this
repo's original PPO. Algorithmic choices mirror other-project's run_train_dataset:
15-minute 17D environment, fixed 30-minute blocks, month LP oracle, oracle-based
peak shaping, behaviour cloning, decomposed PPO, calendar-month holdouts, and the
same selection protocol. Only repository paths/config field names are adapted.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import shutil
import statistics
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bess.agents.ppo2_agent import PPO2Agent, RolloutBuffer, resolve_ppo2_device
from bess.core.common import RESULTS_DIR
from bess.core.ppo2_env import PPO2Env, PPO2_OBS_DIM, PPO2_STEPS_PER_DAY
from bess.core.scenario_gen import DayData, MonthData
from bess.evaluation.oracle.ppo2_oracle import (
    fixed_pmax_day,
    run_no_bess,
    run_oracle,
    score_month,
)
from bess.training.training_common import build_training_bess_config
from bess.training.training_reports import write_report

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
LAMBDA_ENERGY = 0.97
LAMBDA_PEAK = 0.97
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


def _dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_csv_days_reference(path: Path) -> list[DayData]:
    """Senior loader: fixed 96-slot arrays, sorted dates, sequential day_index."""
    by_day: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            day = by_day.setdefault(
                row["date_iso"],
                {
                    "load": np.zeros(PPO2_STEPS_PER_DAY),
                    "pv": np.zeros(PPO2_STEPS_PER_DAY),
                    "day_type": row["day_type"],
                },
            )
            step = int(row["step"])
            if not 0 <= step < PPO2_STEPS_PER_DAY:
                raise ValueError(f"PPO2 senior-reference step must be 0..95, got {step}")
            day["load"][step] = float(row["P_load_kW"])
            day["pv"][step] = float(row["P_pv_kW"])
    return [
        DayData(
            load=value["load"],
            pv=value["pv"],
            day_type=value["day_type"],
            weather="tb",
            day_index=index + 1,
            date_iso=date_iso,
        )
        for index, (date_iso, value) in enumerate(sorted(by_day.items()))
    ]


def complete_month_blocks(
    days: list[DayData], min_coverage: float = MIN_MONTH_COVERAGE
) -> list[MonthData]:
    grouped: dict[str, list[DayData]] = {}
    for day in days:
        if not day.date_iso:
            raise ValueError("PPO2 senior-reference training requires date_iso on every day")
        grouped.setdefault(day.date_iso[:7], []).append(day)

    months: list[MonthData] = []
    for key, block_days in sorted(grouped.items()):
        year, month = (int(value) for value in key.split("-"))
        expected = calendar.monthrange(year, month)[1]
        actual_days = {int(day.date_iso[-2:]) for day in block_days}
        if len(actual_days) < min_coverage * expected:
            continue
        block = MonthData(source=f"csv:{key}")
        block.days = sorted(block_days, key=lambda item: item.date_iso)
        months.append(block)
    return months


def _split_months(
    all_days: list[DayData],
    train_coverage: float,
    val_months: int = VAL_MONTHS,
    test_months: int = TEST_MONTHS,
) -> tuple[list[MonthData], list[MonthData], list[MonthData]]:
    months = complete_month_blocks(all_days, min_coverage=train_coverage)
    eligible = complete_month_blocks(all_days, min_coverage=train_coverage)
    holdout_count = val_months + test_months
    if len(months) < holdout_count + 1:
        raise SystemExit(
            f"Need at least {holdout_count + 1} calendar months covering "
            f"{train_coverage:.0%} of their days for PPO2 train/validation/test "
            f"(found {len(months)})"
        )
    if len(eligible) < holdout_count:
        raise SystemExit(
            f"Need {holdout_count} months covering {train_coverage:.0%} of "
            f"their days for PPO2 validation/test (found {len(eligible)})"
        )
    holdout = months[-holdout_count:]
    return months[:-holdout_count], holdout[:val_months], holdout[val_months:]


def _flatten(months: list[MonthData]) -> list[DayData]:
    return [day for month in months for day in month.days]


def _augment_month_reference(
    month: MonthData,
    rng: np.random.Generator,
    *,
    load_sigma: float = AUG_LOAD_SIGMA,
    pv_sigma: float = AUG_PV_SIGMA,
    rho_load: float = AUG_RHO_LOAD,
    rho_pv: float = AUG_RHO_PV,
) -> MonthData:
    """Senior's scalar-draw AR(1) augmentation, including RNG consumption order."""
    out = MonthData(source=month.source + ":aug")
    for day in month.days:
        def _ar1(sig: float, rho: float) -> np.ndarray:
            error = np.zeros(PPO2_STEPS_PER_DAY)
            white = sig * np.sqrt(1.0 - rho ** 2)
            for step in range(1, PPO2_STEPS_PER_DAY):
                error[step] = rho * error[step - 1] + white * rng.standard_normal()
            return error

        load = np.maximum(0.0, day.load * (1.0 + _ar1(load_sigma, rho_load)))
        pv = np.maximum(0.0, day.pv * (1.0 + _ar1(pv_sigma, rho_pv)))
        out.days.append(
            DayData(
                load=load,
                pv=pv,
                day_type=day.day_type,
                weather=day.weather,
                day_index=day.day_index,
                date_iso=day.date_iso,
            )
        )
    return out


def _score_result(
    result: dict,
    month: MonthData,
    cfg,
    degradation_cost_per_kwh_discharged: float,
) -> dict:
    return score_month(
        result["p_grid_days"],
        cfg,
        days=month.days,
        p_bess_days=result["p_bess_days"],
        soc_days=result["soc_days"],
        degradation_cost_per_kwh_discharged=degradation_cost_per_kwh_discharged,
    )


def _run_policy(
    month: MonthData,
    cfg,
    agent,
    p_ref_kw: float,
    degradation_cost_per_kwh_discharged: float,
    *,
    deterministic: bool = True,
) -> dict:
    env = PPO2Env(
        cfg,
        p_ref_kw=p_ref_kw,
        degradation_cost_per_kwh_discharged=degradation_cost_per_kwh_discharged,
    )
    obs = env.reset(month)
    done = False
    requested: list[float] = []
    executed: list[float] = []
    clip_reasons: list[str | None] = []
    while not done:
        action = agent.act(obs, deterministic=deterministic)[0]
        obs, _reward, done, info = env.step(action)
        requested.append(info["p_requested_kw"])
        executed.append(info["p_executed_kw"])
        clip_reasons.append(info["clip_reason"])

    def split(values: np.ndarray) -> list[np.ndarray]:
        return [
            values[index:index + PPO2_STEPS_PER_DAY]
            for index in range(0, len(values), PPO2_STEPS_PER_DAY)
        ]

    return {
        "p_grid_days": env.log_grid,
        "soc_days": env.log_soc,
        "p_bess_days": env.log_pbess,
        "p_requested_days": split(np.asarray(requested, dtype=float)),
        "p_executed_days": split(np.asarray(executed, dtype=float)),
        "clip_reason_days": [
            clip_reasons[index:index + PPO2_STEPS_PER_DAY]
            for index in range(0, len(clip_reasons), PPO2_STEPS_PER_DAY)
        ],
        "peak_actual_grid_charge_kwh": env.peak_actual_grid_charge_kwh,
    }


def _baseline_cost(
    months: list[MonthData],
    cfg,
    degradation_cost_per_kwh_discharged: float,
    runner,
) -> float:
    total = 0.0
    for month in months:
        if runner is run_oracle:
            result = runner(
                month,
                cfg,
                degradation_cost_per_kwh_discharged=degradation_cost_per_kwh_discharged,
            )
        else:
            result = runner(month, cfg)
        total += _score_result(
            result, month, cfg, degradation_cost_per_kwh_discharged
        )["total_cost_vnd"]
    return total


def _evaluate_months(
    months: list[MonthData],
    cfg,
    agent,
    p_ref: float,
    degradation_cost_per_kwh_discharged: float,
    *,
    deterministic: bool = True,
) -> dict:
    additive = (
        "energy_cost_vnd",
        "demand_cost_vnd",
        "electricity_bill_vnd",
        "degradation_cost_vnd",
        "terminal_settlement_vnd",
        "total_cost_vnd",
        "throughput_kwh",
        "discharged_kwh",
        "equivalent_full_cycles",
    )
    totals = dict.fromkeys(additive, 0.0)
    peak_kw = 0.0
    terminal_soc = None
    clipped_kwh = 0.0
    for month in months:
        result = _run_policy(
            month,
            cfg,
            agent,
            p_ref,
            degradation_cost_per_kwh_discharged,
            deterministic=deterministic,
        )
        score = _score_result(
            result, month, cfg, degradation_cost_per_kwh_discharged
        )
        for key in additive:
            totals[key] += score[key]
        peak_kw = max(peak_kw, score["pmax_month_kw"])
        terminal_soc = score["terminal_soc_fraction"]
        clipped_kwh += sum(
            float(np.abs(requested - executed).sum()) * 0.25
            for requested, executed in zip(
                result["p_requested_days"], result["p_executed_days"], strict=True
            )
        )
    return {
        **totals,
        "pmax_month_kw": peak_kw,
        "terminal_soc_fraction": terminal_soc,
        "months_scored": len(months),
        "physical_clip_kwh": clipped_kwh,
    }


def _shaping_warm_starts(
    months: list[MonthData],
    cfg,
    oracle_solutions: list[dict] | None = None,
    *,
    margin: float = D_RUN_SHAPING_ORACLE_MARGIN,
) -> list[float]:
    if cfg.T_cap <= 0.0:
        return [0.0] * len(months)
    if oracle_solutions is None:
        raise ValueError("PPO2 shaping warm starts require the month Oracle solutions")
    warm: list[float] = []
    for month, solution in zip(months, oracle_solutions, strict=True):
        ppk = float(solution["ppk_lp_kw"])
        nobess_peak = max(
            fixed_pmax_day(np.maximum(0.0, day.load - day.pv))
            for day in month.days
        )
        value = margin * ppk
        print(
            f"[train-ppo2] shaping {month.source}: W={value:.1f} kW "
            f"(oracle ppk {ppk:.1f}, no-BESS peak {nobess_peak:.1f}, "
            f"ratio {ppk / max(nobess_peak, 1e-9):.2f})",
            flush=True,
        )
        warm.append(value)
    return warm


def _behavior_clone_actor(
    agent: PPO2Agent,
    months: list[MonthData],
    oracle_solutions: list[dict],
    cfg,
    p_ref: float,
    degradation_cost_per_kwh_discharged: float,
    clip_penalty_per_kwh: float,
    epochs: int,
    seed: int,
    *,
    learning_rate: float = BC_LR,
    minibatch: int = BC_MINIBATCH,
    action_clip: float = BC_ACTION_CLIP,
) -> None:
    if epochs <= 0:
        return
    env = PPO2Env(
        cfg,
        p_ref_kw=p_ref,
        degradation_cost_per_kwh_discharged=degradation_cost_per_kwh_discharged,
        clip_penalty_per_kwh=clip_penalty_per_kwh,
    )
    observations: list[np.ndarray] = []
    targets: list[float] = []
    for month, solution in zip(months, oracle_solutions, strict=True):
        obs = env.reset(month)
        done = False
        flat = np.concatenate([
            np.asarray(day_power, dtype=np.float64)
            for day_power in solution["p_bess_days"]
        ])
        for target_power in flat:
            if done:
                raise RuntimeError("PPO2 oracle trajectory is longer than the episode")
            action_star = float(np.clip(
                target_power / cfg.P_rated_nominal, -action_clip, action_clip
            ))
            if env.history_ready:
                observations.append(obs)
                targets.append(action_star)
            obs, _reward, done, _info = env.step(action_star)

    obs_tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    target_tensor = torch.as_tensor(np.asarray(targets, dtype=np.float32))
    optimizer = torch.optim.Adam(agent.net.actor.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(targets))
    loss_value = float("nan")
    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), minibatch):
            batch = indices[start:start + minibatch]
            predicted = torch.tanh(agent.net.actor(obs_tensor[batch])).squeeze(-1)
            loss = ((predicted - target_tensor[batch]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
    print(
        f"[train-ppo2] behaviour cloning: {len(targets)} oracle steps, "
        f"{epochs} epochs, final minibatch MSE {loss_value:.5f}",
        flush=True,
    )


def _split_reward(info: dict, reward_scale_vnd: float) -> tuple[float, float]:
    energy = -(
        info["rew_energy_delta"]
        + info["rew_deg_cost"]
        + info["rew_terminal_cost"]
        + info["rew_clip_cost"]
    ) / reward_scale_vnd
    peak = -info["rew_peak_delta"] / reward_scale_vnd
    return energy, peak


def _write_curve(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _train_seed(
    *,
    cfg,
    csv_path: Path,
    tag: str,
    seed: int,
    total_steps: int,
    p_ref: float,
    train_months: list[MonthData],
    val_months: list[MonthData],
    test_months: list[MonthData],
    degradation_cost_per_kwh_discharged: float,
    val_base: float,
    val_oracle: float,
    train_warm_starts: list[float],
    rollout: int,
    eval_every_updates: int,
    lam_energy: float,
    lam_peak: float,
    actor_lr: float,
    critic_lr: float,
    log_std_init: float,
    clip_penalty_per_kwh: float,
    bc_epochs: int,
    gamma: float,
    ppo_clip: float,
    ppo_epochs: int,
    ppo_minibatch: int,
    entropy_coef: float,
    value_coef: float,
    target_kl: float,
    bc_lr: float,
    bc_minibatch: int,
    bc_action_clip: float,
    shaping_margin: float,
    min_month_coverage: float,
    aug_load_sigma: float,
    aug_pv_sigma: float,
    aug_rho_load: float,
    aug_rho_pv: float,
    torch_threads: int,
    train_oracle: list[dict] | None,
    config_hash: str,
) -> tuple[float, Path, list[dict]]:
    env = PPO2Env(
        cfg,
        p_ref_kw=p_ref,
        degradation_cost_per_kwh_discharged=degradation_cost_per_kwh_discharged,
        clip_penalty_per_kwh=clip_penalty_per_kwh,
    )
    agent = PPO2Agent(
        env.obs_dim,
        seed=seed,
        gamma=gamma,
        lam_energy=lam_energy,
        lam_peak=lam_peak,
        clip=ppo_clip,
        epochs=ppo_epochs,
        minibatch=ppo_minibatch,
        ent_coef=entropy_coef,
        vf_coef=value_coef,
        target_kl=target_kl,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        log_std_init=log_std_init,
        device="cpu",
    )
    if bc_epochs > 0:
        if train_oracle is None:
            raise ValueError("PPO2 behaviour cloning requires Oracle solutions")
        _behavior_clone_actor(
            agent,
            train_months,
            train_oracle,
            cfg,
            p_ref,
            degradation_cost_per_kwh_discharged,
            clip_penalty_per_kwh,
            bc_epochs,
            seed,
            learning_rate=bc_lr,
            minibatch=bc_minibatch,
            action_clip=bc_action_clip,
        )

    agent.meta = {
        "artifact_schema_version": "2.0-reference-port",
        "p_ref_kw": p_ref,
        "e_cap_kwh": cfg.E_cap,
        "p_rated_kw": cfg.P_rated_nominal,
        "obs_dim": PPO2_OBS_DIM,
        "obs_variant": "base",
        "native_dt_minutes": 15.0,
        "control_dt_minutes": 15.0,
        "native_steps_per_action": 1,
        "observation_schema": "causal_block_aware",
        "reference_env": "ppo2_senior_15m_v1",
        "action_distribution": "tanh_squashed_gaussian",
        "action_mapping": "physical_feasible_15m",
        "action_interval_minutes": 15,
        "objective": "energy+demand_fixed30m+degradation_discharged+terminal",
        "demand_window": "fixed_30m_block_v1",
        "billing_demand_window_minutes": 30,
        "advantage_estimator": "decomposed_gae",
        "lambda_energy": lam_energy,
        "lambda_peak": lam_peak,
        "gamma": gamma,
        "ppo_clip": ppo_clip,
        "ppo_epochs": ppo_epochs,
        "ppo_minibatch": ppo_minibatch,
        "entropy_coef": entropy_coef,
        "value_coef": value_coef,
        "target_kl": target_kl,
        "actor_learning_rate": actor_lr,
        "critic_learning_rate": critic_lr,
        "log_std_init": log_std_init,
        "clip_penalty_per_kwh": clip_penalty_per_kwh,
        "bc_epochs": bc_epochs,
        "bc_learning_rate": bc_lr,
        "bc_minibatch": bc_minibatch,
        "bc_action_clip": bc_action_clip,
        "rollout_steps": rollout,
        "total_steps": total_steps,
        "eval_every_updates": eval_every_updates,
        "d_run_shaping_anchor": "oracle_month_peak",
        "d_run_shaping_margin": shaping_margin,
        "train_warm_starts_kw": [round(value, 2) for value in train_warm_starts],
        "min_month_coverage_train": min_month_coverage,
        "min_month_coverage_eval": min_month_coverage,
        "value_normalization": "popart_per_component",
        "billing_mode": "tou" if cfg.T_cap <= 0.0 else "2tc",
        "demand_charge_active": cfg.T_cap > 0.0,
        "reward_scale_vnd": env.reward_scale_vnd,
        "degradation_cost_per_kwh_discharged": degradation_cost_per_kwh_discharged,
        "seed": seed,
        "train_csv": str(csv_path),
        "dataset_hash": _dataset_hash(csv_path),
        "config_hash": config_hash,
        "train_range": [train_months[0].days[0].date_iso, train_months[-1].days[-1].date_iso],
        "validation_range": [val_months[0].days[0].date_iso, val_months[-1].days[-1].date_iso],
        "test_range": [test_months[0].days[0].date_iso, test_months[-1].days[-1].date_iso],
        "train_month_count": len(train_months),
        "validation_month_count": len(val_months),
        "test_month_count": len(test_months),
        "torch_threads": torch_threads,
        "augmentation": {
            "sigmaLoad": aug_load_sigma,
            "sigmaPv": aug_pv_sigma,
            "rhoLoad": aug_rho_load,
            "rhoPv": aug_rho_pv,
        },
    }

    buffer = RolloutBuffer(rollout, env.obs_dim)
    rng = np.random.default_rng(seed)
    month_index = 0
    obs = env.reset(
        _augment_month_reference(
            train_months[0], rng,
            load_sigma=aug_load_sigma, pv_sigma=aug_pv_sigma,
            rho_load=aug_rho_load, rho_pv=aug_rho_pv,
        ),
        d_run_shaping_init_kw=train_warm_starts[0],
    )
    candidate = RESULTS_DIR / f"policy_{tag}_seed{seed}.pt"
    candidate.unlink(missing_ok=True)
    curves: list[dict] = []
    best_val = float("inf")
    steps = 0
    updates = 0
    evaluated_at = -1
    done = False
    started = time.time()

    def evaluate_and_checkpoint() -> None:
        nonlocal best_val
        torch_rng_state = torch.get_rng_state()
        try:
            score = _evaluate_months(
                val_months,
                cfg,
                agent,
                p_ref,
                degradation_cost_per_kwh_discharged,
            )
            stochastic = _evaluate_months(
                val_months,
                cfg,
                agent,
                p_ref,
                degradation_cost_per_kwh_discharged,
                deterministic=False,
            )
        finally:
            torch.set_rng_state(torch_rng_state)
        val_cost = score["total_cost_vnd"]
        saving = (val_base - val_cost) / val_base * 100.0
        gap = (val_cost - val_oracle) / val_oracle * 100.0
        improved = val_cost < best_val
        diagnostics = agent.diagnostics
        curves.append({
            "seed": seed,
            "lambda_peak": lam_peak,
            "steps": steps,
            "val_cost_vnd": val_cost,
            "val_cost_stochastic_vnd": stochastic["total_cost_vnd"],
            "oracle_gap_pct": gap,
            "saving_vs_nobess_pct": saving,
            "physical_clip_kwh": score["physical_clip_kwh"],
            "val_pmax_kw": score["pmax_month_kw"],
            "adv_raw_std": diagnostics.get("adv_raw_std", 0.0),
            "adv_near_zero_pct": diagnostics.get("adv_near_zero_pct", 0.0),
            "approx_kl": diagnostics.get("approx_kl", 0.0),
            "epochs_run": diagnostics.get("epochs_run", 0),
            "log_std": diagnostics.get("log_std", log_std_init),
            "adv_share_energy": diagnostics.get("adv_share_energy", 0.0),
            "adv_share_peak": diagnostics.get("adv_share_peak", 0.0),
            "value_std_energy": diagnostics.get("value_std_energy", 1.0),
            "value_std_peak": diagnostics.get("value_std_peak", 1.0),
            "is_best": improved,
        })
        if improved:
            best_val = val_cost
            agent.meta["validation_cost_vnd"] = val_cost
            agent.meta["validation_steps"] = steps
            agent.save(candidate)
        print(
            f"  seed {seed} lam_p {lam_peak:.2f} step {steps:>7} | "
            f"val {val_cost/1e6:8.1f}M (stoch {stochastic['total_cost_vnd']/1e6:.1f}M) | "
            f"saving {saving:5.1f}% | gap {gap:6.1f}% | "
            f"kl {diagnostics.get('approx_kl', 0.0):6.4f} | "
            f"logstd {diagnostics.get('log_std', log_std_init):6.3f} | "
            f"{steps / max(time.time() - started, 1e-9):,.0f} sps"
            f"{' | best' if improved else ''}",
            flush=True,
        )

    while steps < total_steps:
        action, logp, latent, value_energy, value_peak = agent.act(obs)
        next_obs, _reward, done, info = env.step(action)
        reward_energy, reward_peak = _split_reward(info, env.reward_scale_vnd)
        if not info["action_held"]:
            buffer.add(
                obs,
                action,
                latent,
                logp,
                reward_energy,
                reward_peak,
                value_energy,
                value_peak,
                float(done),
            )
        steps += 1
        if done:
            month_index += 1
            next_index = month_index % len(train_months)
            next_obs = env.reset(
                _augment_month_reference(
                    train_months[next_index], rng,
                    load_sigma=aug_load_sigma, pv_sigma=aug_pv_sigma,
                    rho_load=aug_rho_load, rho_pv=aug_rho_pv,
                ),
                d_run_shaping_init_kw=train_warm_starts[next_index],
            )
        obs = next_obs

        if buffer.full():
            *_, last_energy, last_peak = agent.act(obs)
            agent.anneal_lr(steps / max(1, total_steps))
            agent.update(buffer, last_energy, last_peak)
            updates += 1
            if updates % eval_every_updates == 0:
                evaluate_and_checkpoint()
                evaluated_at = updates

    if buffer.ptr:
        *_, last_energy, last_peak = agent.act(obs)
        agent.update(buffer, last_energy, last_peak)
        updates += 1
    if updates != evaluated_at:
        evaluate_and_checkpoint()

    return best_val, candidate, curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--e-cap", type=float, required=True)
    parser.add_argument("--p-rated", type=float, required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--steps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--rollout", type=int, default=ROLLOUT)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY_UPDATES)
    parser.add_argument("--min-month-coverage", type=float, default=MIN_MONTH_COVERAGE)
    parser.add_argument("--val-months", type=int, default=VAL_MONTHS)
    parser.add_argument("--test-months", type=int, default=TEST_MONTHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--actor-lr", type=float, default=ACTOR_LR)
    parser.add_argument("--critic-lr", type=float, default=CRITIC_LR)
    parser.add_argument("--init-std", type=float, default=INIT_STD)
    parser.add_argument("--clip-penalty", type=float, default=CLIP_PENALTY_PER_KWH)
    parser.add_argument("--bc-epochs", type=int, default=BC_EPOCHS)
    parser.add_argument("--ppo-clip", type=float, default=PPO_CLIP)
    parser.add_argument("--ppo-epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--minibatch", type=int, default=PPO_MINIBATCH)
    parser.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF)
    parser.add_argument("--value-coef", type=float, default=VALUE_COEF)
    parser.add_argument("--target-kl", type=float, default=TARGET_KL)
    parser.add_argument("--shaping-margin", type=float, default=D_RUN_SHAPING_ORACLE_MARGIN)
    parser.add_argument("--aug-load-sigma", type=float, default=AUG_LOAD_SIGMA)
    parser.add_argument("--aug-pv-sigma", type=float, default=AUG_PV_SIGMA)
    parser.add_argument("--aug-rho-load", type=float, default=AUG_RHO_LOAD)
    parser.add_argument("--aug-rho-pv", type=float, default=AUG_RHO_PV)
    parser.add_argument("--bc-lr", type=float, default=BC_LR)
    parser.add_argument("--bc-minibatch", type=int, default=BC_MINIBATCH)
    parser.add_argument("--bc-action-clip", type=float, default=BC_ACTION_CLIP)
    parser.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    parser.add_argument("--lambda-energy", "--lam-energy", dest="lambda_energy", type=float, default=LAMBDA_ENERGY)
    parser.add_argument("--lambda-peak", "--lam-peak", dest="lambda_peak", type=str, default=str(LAMBDA_PEAK))
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--tag", type=str, default="")
    # Compatibility-only args still emitted by this repo's generic launcher.
    parser.add_argument("--oracle-cache", default="")
    parser.add_argument("--val-days", type=int, default=0)
    parser.add_argument("--test-days", type=int, default=0)
    parser.add_argument("--control-dt-minutes", type=float, default=15.0)
    parser.add_argument("--obs-variant", choices=("base", "fc"), default="base")
    parser.add_argument("--weather-data", default="")
    parser.add_argument("--forecast-artifact", default="")
    parser.add_argument("--billing", choices=("2tc", "tou"), default="2tc")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if not math.isclose(args.gamma, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("PPO2 senior-reference mode requires gamma=1.0")
    if args.obs_variant != "base":
        raise SystemExit("PPO2 senior-reference mode is forecast-free and requires --obs-variant base")
    if abs(args.control_dt_minutes - 15.0) > 1e-9:
        raise SystemExit("PPO2 senior-reference mode requires a 15-minute control interval")
    resolve_ppo2_device(args.device)
    numeric_checks = (
        (args.steps > 0, "--steps must be > 0"),
        (args.rollout > 0, "--rollout must be > 0"),
        (args.eval_every > 0, "--eval-every must be > 0"),
        (0.0 < args.min_month_coverage <= 1.0, "--min-month-coverage must be in (0, 1]"),
        (args.val_months > 0 and args.test_months > 0, "--val-months and --test-months must be > 0"),
        (args.actor_lr > 0.0 and args.critic_lr > 0.0, "actor/critic learning rates must be > 0"),
        (args.init_std > 0.0, "--init-std must be > 0"),
        (args.clip_penalty >= 0.0, "--clip-penalty must be >= 0"),
        (args.bc_epochs >= 0, "--bc-epochs must be >= 0"),
        (0.0 < args.ppo_clip <= 1.0, "--ppo-clip must be in (0, 1]"),
        (args.ppo_epochs > 0 and args.minibatch > 0, "PPO epochs/minibatch must be > 0"),
        (args.entropy_coef >= 0.0 and args.value_coef >= 0.0, "entropy/value coefficients must be >= 0"),
        (args.target_kl > 0.0, "--target-kl must be > 0"),
        (0.0 <= args.shaping_margin <= 1.0, "--shaping-margin must be in [0, 1]"),
        (args.aug_load_sigma >= 0.0 and args.aug_pv_sigma >= 0.0, "augmentation sigmas must be >= 0"),
        (abs(args.aug_rho_load) < 1.0 and abs(args.aug_rho_pv) < 1.0, "augmentation rho values must satisfy abs(rho) < 1"),
        (args.bc_lr > 0.0 and args.bc_minibatch > 0, "BC learning rate/minibatch must be > 0"),
        (0.0 < args.bc_action_clip <= 1.0, "--bc-action-clip must be in (0, 1]"),
        (args.torch_threads > 0, "--torch-threads must be > 0"),
        (0.0 <= args.lambda_energy <= 1.0, "--lambda-energy must be in [0, 1]"),
    )
    for valid, message in numeric_checks:
        if not valid:
            raise SystemExit(message)
    torch.set_num_threads(args.torch_threads)

    csv_path = Path(args.csv)
    all_days = _load_csv_days_reference(csv_path)
    train_months, val_months, test_months = _split_months(
        all_days,
        args.min_month_coverage,
        val_months=args.val_months,
        test_months=args.test_months,
    )
    train_days = _flatten(train_months)
    peak = max(float(day.load.max()) for day in train_days)
    p_ref = max(500.0, math.ceil(peak / 500.0) * 500.0)

    cfg, billing = build_training_bess_config(
        args.e_cap,
        args.p_rated,
        0.25,
        args.training_config,
        default_billing=args.billing,
    )
    config_path = Path(args.training_config)
    config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    if "battery_wear_cost" not in config_raw:
        raise SystemExit("PPO2 senior-reference training config requires battery_wear_cost")
    degradation_cost = float(config_raw["battery_wear_cost"])
    config_hash = _dataset_hash(config_path)
    tag = args.tag or f"ppo2_{cfg.E_cap:.0f}kwh_{cfg.P_rated_nominal:.0f}kw"

    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else [args.seed]
    )
    lambda_peaks = [
        float(value) for value in args.lambda_peak.split(",") if value.strip()
    ]
    if not lambda_peaks:
        raise SystemExit("--lambda-peak must list at least one value")
    if any(not 0.0 <= value <= 1.0 for value in lambda_peaks):
        raise SystemExit("--lambda-peak values must be in [0, 1]")

    need_train_oracle = cfg.T_cap > 0.0 or args.bc_epochs > 0
    train_oracle = (
        [
            run_oracle(
                month,
                cfg,
                degradation_cost_per_kwh_discharged=degradation_cost,
            )
            for month in train_months
        ]
        if need_train_oracle
        else None
    )
    train_warm_starts = _shaping_warm_starts(
        train_months, cfg, train_oracle, margin=args.shaping_margin
    )
    log_std_init = math.log(args.init_std)

    if len(train_months) < 6:
        print(
            f"[train-ppo2] WARNING: only {len(train_months)} training month(s); "
            "the demand-charge decision gets very few independent monthly samples.",
            flush=True,
        )
    print(
        f"[train-ppo2] SENIOR REFERENCE | BESS {cfg.E_cap:.0f} kWh / "
        f"{cfg.P_rated_nominal:.0f} kW | months train={len(train_months)} "
        f"val={len(val_months)} test={len(test_months)} "
        f"(coverage >={args.min_month_coverage:.0%}) | p_ref={p_ref:.0f} | "
        f"degradation={degradation_cost:.2f}/kWh | billing={billing} | "
        f"seeds={seeds} | lambda_peak={lambda_peaks}",
        flush=True,
    )

    val_base = _baseline_cost(val_months, cfg, degradation_cost, run_no_bess)
    val_oracle = _baseline_cost(val_months, cfg, degradation_cost, run_oracle)
    curve: list[dict] = []
    arm_results: dict[float, list[tuple[float, Path]]] = {}
    for lam_peak in lambda_peaks:
        arm_results[lam_peak] = []
        for seed in seeds:
            val_cost, candidate, seed_curve = _train_seed(
                cfg=cfg,
                csv_path=csv_path,
                tag=f"{tag}_lp{lam_peak:g}",
                seed=seed,
                total_steps=args.steps,
                p_ref=p_ref,
                train_months=train_months,
                val_months=val_months,
                test_months=test_months,
                degradation_cost_per_kwh_discharged=degradation_cost,
                val_base=val_base,
                val_oracle=val_oracle,
                train_warm_starts=train_warm_starts,
                rollout=args.rollout,
                eval_every_updates=max(1, args.eval_every),
                lam_energy=args.lambda_energy,
                lam_peak=lam_peak,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                log_std_init=log_std_init,
                clip_penalty_per_kwh=args.clip_penalty,
                bc_epochs=args.bc_epochs,
                gamma=args.gamma,
                ppo_clip=args.ppo_clip,
                ppo_epochs=args.ppo_epochs,
                ppo_minibatch=args.minibatch,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                target_kl=args.target_kl,
                bc_lr=args.bc_lr,
                bc_minibatch=args.bc_minibatch,
                bc_action_clip=args.bc_action_clip,
                shaping_margin=args.shaping_margin,
                min_month_coverage=args.min_month_coverage,
                aug_load_sigma=args.aug_load_sigma,
                aug_pv_sigma=args.aug_pv_sigma,
                aug_rho_load=args.aug_rho_load,
                aug_rho_pv=args.aug_rho_pv,
                torch_threads=args.torch_threads,
                train_oracle=train_oracle,
                config_hash=config_hash,
            )
            curve.extend(seed_curve)
            arm_results[lam_peak].append((val_cost, candidate))

    arm_means = {
        lam: statistics.mean(cost for cost, _ in results)
        for lam, results in arm_results.items()
    }
    best_lambda = min(arm_means, key=arm_means.get)
    for lam in lambda_peaks:
        marker = " <- selected" if lam == best_lambda else ""
        print(
            f"[train-ppo2] lambda_peak {lam:.2f}: mean val "
            f"{arm_means[lam]/1e6:.1f}M over {len(seeds)} seed(s){marker}",
            flush=True,
        )
    best_overall, best_candidate = min(arm_results[best_lambda])
    final_path = RESULTS_DIR / f"policy_{tag}.pt"
    shutil.copy2(best_candidate, final_path)

    test_base = _baseline_cost(test_months, cfg, degradation_cost, run_no_bess)
    test_oracle = _baseline_cost(test_months, cfg, degradation_cost, run_oracle)
    savings: list[float] = []
    for _, seed_path in arm_results[best_lambda]:
        seed_agent = PPO2Agent(PPO2_OBS_DIM, device="cpu")
        seed_agent.load(seed_path)
        seed_score = _evaluate_months(
            test_months, cfg, seed_agent, p_ref, degradation_cost
        )
        savings.append((test_base - seed_score["total_cost_vnd"]) / test_base * 100.0)

    best_agent = PPO2Agent(PPO2_OBS_DIM, device="cpu")
    best_agent.load(final_path)
    test_score = _evaluate_months(test_months, cfg, best_agent, p_ref, degradation_cost)
    test_saving = (test_base - test_score["total_cost_vnd"]) / test_base * 100.0
    test_gap = (test_score["total_cost_vnd"] - test_oracle) / test_oracle * 100.0
    best_agent.meta.update({
        "validation_cost_vnd": best_overall,
        "lambda_peak_sweep": lambda_peaks,
        "lambda_peak_selected": best_lambda,
        "lambda_peak_mean_val_cost_vnd": arm_means,
        "selection_protocol": "mean_val_cost_over_seeds_then_best_seed",
        "test_saving_pct": round(test_saving, 2),
        "test_metrics": test_score,
        "test_oracle_gap_pct": round(test_gap, 2),
        "seeds": seeds,
        "test_saving_pct_by_seed": [round(value, 2) for value in savings],
        "trained": date.today().isoformat(),
    })
    best_agent.save(final_path)

    curve_path = RESULTS_DIR / f"training_curve_{tag}.csv"
    _write_curve(curve_path, curve)
    report_path = RESULTS_DIR / f"training_report_{tag}.json"
    report = {
        "version": 2,
        "status": "complete",
        "algorithm": "ppo2-senior-reference",
        "tag": tag,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": str(csv_path),
            "train_months": len(train_months),
            "validation_months": len(val_months),
            "test_months": len(test_months),
            "train_range": [train_months[0].days[0].date_iso, train_months[-1].days[-1].date_iso],
            "validation_range": [val_months[0].days[0].date_iso, val_months[-1].days[-1].date_iso],
            "test_range": [test_months[0].days[0].date_iso, test_months[-1].days[-1].date_iso],
        },
        "training": {
            "steps": args.steps,
            "rollout": args.rollout,
            "eval_every": args.eval_every,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "init_std": args.init_std,
            "bc_epochs": args.bc_epochs,
            "bc_lr": args.bc_lr,
            "bc_minibatch": args.bc_minibatch,
            "bc_action_clip": args.bc_action_clip,
            "ppo_clip": args.ppo_clip,
            "ppo_epochs": args.ppo_epochs,
            "ppo_minibatch": args.minibatch,
            "entropy_coef": args.entropy_coef,
            "value_coef": args.value_coef,
            "target_kl": args.target_kl,
            "shaping_margin": args.shaping_margin,
            "min_month_coverage": args.min_month_coverage,
            "validation_months": args.val_months,
            "test_months": args.test_months,
            "augmentation": {
                "sigmaLoad": args.aug_load_sigma,
                "sigmaPv": args.aug_pv_sigma,
                "rhoLoad": args.aug_rho_load,
                "rhoPv": args.aug_rho_pv,
            },
            "torch_threads": args.torch_threads,
            "gamma": args.gamma,
            "lambda_energy": args.lambda_energy,
            "lambda_peak_sweep": lambda_peaks,
            "lambda_peak_selected": best_lambda,
            "seeds": seeds,
            "degradation_cost_per_kwh_discharged": degradation_cost,
        },
        "validation": {
            "no_bess_vnd": val_base,
            "oracle_vnd": val_oracle,
            "best_vnd": best_overall,
        },
        "test": {
            **test_score,
            "no_bess_vnd": test_base,
            "oracle_vnd": test_oracle,
            "saving_pct": test_saving,
            "oracle_gap_pct": test_gap,
        },
    }
    write_report(report_path, report)

    for results in arm_results.values():
        for _, seed_path in results:
            seed_path.unlink(missing_ok=True)

    spread = ""
    if len(savings) > 1:
        spread = (
            f" | across seeds {statistics.mean(savings):.2f}% "
            f"+/- {statistics.stdev(savings):.2f}%"
        )
    print(
        f"[train-ppo2] === TEST {test_months[0].days[0].date_iso}->"
        f"{test_months[-1].days[-1].date_iso}: "
        f"{test_score['total_cost_vnd']/1e6:.1f}M vs no-BESS "
        f"{test_base/1e6:.1f}M -> saving {test_saving:.2f}% | "
        f"oracle gap {test_gap:.2f}% | peak {test_score['pmax_month_kw']:.0f} kW | "
        f"lambda_peak={best_lambda:g} seeds={seeds}{spread} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
