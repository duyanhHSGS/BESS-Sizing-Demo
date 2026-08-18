from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bess.agents.ppo_agent import (
    PPOAgent,
    RolloutBuffer,
    configure_ppo_determinism,
    resolve_ppo_device,
)
from bess.core.bess_env import OBSERVATION_DIM
from bess.core.brain_runtime import (
    REWARD_SCALE_VND,
    make_brain_env,
    native_steps_per_action,
    observation_array,
    step_brain_control,
)
from bess.core.common import RESULTS_DIR, score_month
from bess.core.scenario_gen import MonthData
from bess.core.settings import PPO_GAMMA, PPO_LAMBDA
from bess.evaluation.baselines import run_drl_policy, run_no_bess
from bess.evaluation.oracle.oracle_cache import (
    load_cached_training_dispatch,
    load_cached_training_grids,
)
from bess.training.training_common import (
    build_training_bess_config,
    load_training_days,
)
from bess.training.training_reports import (
    PPO_CHAMPION_CURVE_FIELDS,
    write_curve,
    write_report,
)

VALIDATE_EVERY_UPDATES = 4
CHALLENGER_RESET_PATIENCE = 3
CHALLENGER_RESETS_ENABLED = True
ACTION_MISMATCH_SHAPING_SCALE = 0.1
PPO_FIT_CONTROL_DT_MINUTES = 30.0
ORACLE_BC_MAX_EPOCHS = 100
ORACLE_BC_LEARNING_RATE = 1e-3
ORACLE_BC_MINIBATCH = 256
ORACLE_BC_TARGET_MSE = 1e-4
LOG_EVERY_UPDATES = 1


def _action_mismatch_penalty_vnd(transition, *, timestep_hours: float, wear_vnd_per_kwh: float) -> float:
    """Penalize requested battery energy that physics refuses to execute.

    PPO's sampled action and stored log-probability stay untouched, preserving
    on-policy correctness. This is trainer-only reward shaping: Champion/test
    scoring still uses the real economic objective with no shaping penalty.
    """
    mismatch_kwh = sum(
        abs(result.requested_battery_kw - result.bess.physics.final_battery_kw)
        * timestep_hours
        for result in transition.native_results
    )
    return mismatch_kwh * wear_vnd_per_kwh


def _load_ui_wear_cost(training_config_path: str | Path) -> float:
    config = json.loads(Path(training_config_path).read_text(encoding="utf-8"))
    if "battery_wear_cost" not in config:
        raise SystemExit("PPO training config requires UI battery_wear_cost")
    wear_cost = float(config["battery_wear_cost"])
    if not math.isfinite(wear_cost) or wear_cost < 0.0:
        raise SystemExit("UI battery_wear_cost must be finite and >= 0")
    return wear_cost


def _score_ppo_operating_month(
    p_grid_days: list[np.ndarray],
    p_bess_days: list[np.ndarray],
    cfg,
    wear_cost_vnd_per_kwh: float,
    days: list,
) -> dict:
    utility = score_month(p_grid_days, cfg, days=days)
    throughput_kwh = sum(
        float(np.sum(np.abs(np.asarray(day, dtype=np.float64))) * cfg.dt)
        for day in p_bess_days
    )
    wear_cost_vnd = throughput_kwh * wear_cost_vnd_per_kwh
    return {
        **utility,
        "throughput_kwh": throughput_kwh,
        "wear_cost_vnd": wear_cost_vnd,
        "total_operating_cost_vnd": utility["total_cost_vnd"] + wear_cost_vnd,
    }


def _oracle_dispatch_wear_cost_vnd(
    dispatch_days: list[dict[str, list[float]]],
    *,
    timestep_hours: float,
    wear_vnd_per_kwh: float,
) -> float:
    """Score Oracle charge/discharge throughput with the same LP wear convention."""
    throughput_kwh = float(timestep_hours) * sum(
        sum(
            discharge + grid_charge + solar_charge
            for discharge, grid_charge, solar_charge in zip(
                day["discharge"],
                day["grid_charge"],
                day["solar_charge"],
                strict=True,
            )
        )
        for day in dispatch_days
    )
    return throughput_kwh * float(wear_vnd_per_kwh)


def _oracle_teacher_action(
    dispatch: dict[str, list[float]],
    start: int,
    stop: int,
    cfg,
) -> float:
    """Map Oracle outside-world flows into one held BrainEnv battery-side action."""
    battery_side_kw = []
    for step in range(start, stop):
        discharge_kw = float(dispatch["discharge"][step]) / float(cfg.eta_dis)
        charge_kw = (
            float(dispatch["grid_charge"][step])
            + float(dispatch["solar_charge"][step])
        ) * float(cfg.eta_ch)
        battery_side_kw.append(discharge_kw - charge_kw)
    mean_battery_kw = float(np.mean(battery_side_kw)) if battery_side_kw else 0.0
    return float(np.clip(mean_battery_kw / float(cfg.P_rated_nominal), -1.0, 1.0))


def _collect_oracle_teacher_samples(
    month: MonthData,
    oracle_dispatch: list[dict[str, list[float]]],
    cfg,
    *,
    power_scale_kw: float,
    battery_wear_cost: float,
    native_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay Oracle dispatch through canonical BrainEnv and collect brain7 lessons."""
    if len(oracle_dispatch) != len(month.days):
        raise ValueError("Oracle dispatch day count must match the teacher month")
    env = make_brain_env(
        month,
        cfg,
        power_scale_kw=power_scale_kw,
        battery_wear_vnd_per_kwh=battery_wear_cost,
    )
    observation = env.reset()
    observations: list[np.ndarray] = []
    targets: list[float] = []

    for day, dispatch in zip(month.days, oracle_dispatch, strict=True):
        native_rows = len(day.load)
        if any(len(values) != native_rows for values in dispatch.values()):
            raise ValueError(
                f"Oracle dispatch length does not match dataset day {day.day_index}"
            )
        for start in range(0, native_rows, native_steps):
            stop = min(start + native_steps, native_rows)
            action = _oracle_teacher_action(dispatch, start, stop, cfg)
            observations.append(observation_array(observation))
            targets.append(action)
            transition = step_brain_control(
                env,
                action,
                native_steps=stop - start,
            )
            if transition.next_observation is not None:
                observation = transition.next_observation

    return np.asarray(observations, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _behavior_clone_actor(
    agent: PPOAgent,
    observations: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int]:
    """Supervised-fit the existing generic PPO actor to Oracle teacher actions."""
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIM:
        raise ValueError("Oracle BC observations must have shape (N, OBSERVATION_DIM)")
    if targets.ndim != 1 or len(targets) != len(observations) or len(targets) == 0:
        raise ValueError("Oracle BC targets must be one non-empty action per observation")

    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    target_t = torch.as_tensor(targets, dtype=torch.float32, device=agent.device)
    optimizer = torch.optim.Adam(agent.net.actor.parameters(), lr=ORACLE_BC_LEARNING_RATE)
    rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def full_mse() -> float:
        prediction = torch.tanh(agent.net.actor(obs_t)).squeeze(-1)
        return float(torch.mean((prediction - target_t) ** 2).cpu())

    initial_mse = full_mse()
    epochs_completed = 0
    for epoch in range(ORACLE_BC_MAX_EPOCHS):
        indexes = rng.permutation(len(observations))
        for start in range(0, len(indexes), ORACLE_BC_MINIBATCH):
            mb = torch.as_tensor(
                indexes[start:start + ORACLE_BC_MINIBATCH],
                dtype=torch.long,
                device=agent.device,
            )
            prediction = torch.tanh(agent.net.actor(obs_t[mb])).squeeze(-1)
            loss = torch.mean((prediction - target_t[mb]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epochs_completed = epoch + 1
        if full_mse() <= ORACLE_BC_TARGET_MSE:
            break

    final_mse = full_mse()
    agent._sync_collector()
    return {
        "samples": len(observations),
        "charge_samples": int(np.count_nonzero(targets < -1e-6)),
        "idle_samples": int(np.count_nonzero(np.abs(targets) <= 1e-6)),
        "discharge_samples": int(np.count_nonzero(targets > 1e-6)),
        "epochs_completed": int(epochs_completed),
        "initial_mse": initial_mse,
        "final_mse": final_mse,
    }


def _initialize_champion(agent: PPOAgent, validate_cost, checkpoint_path: Path) -> float:
    """Validate and persist Champion #0 before any PPO update can run."""
    champion_cost = float(validate_cost())
    agent.save(checkpoint_path)
    return champion_cost


def _resolve_challenger(
    champion_cost: float,
    candidate_cost: float,
) -> tuple[float, bool]:
    """Keep the trusted Champion monotonic without mutating the live learner."""
    if candidate_cost < champion_cost:
        return candidate_cost, True
    return champion_cost, False


def _save_accepted_champion(agent: PPOAgent, checkpoint_path: Path, accepted: bool) -> bool:
    """Persist only accepted learner state; rejected challengers never touch disk."""
    if not accepted:
        return False
    agent.save(checkpoint_path)
    return True


def _champion_curve_point(
    *,
    steps: int,
    candidate_cost: float,
    champion_cost: float,
    accepted: bool,
    val_base: float,
    val_oracle: float,
) -> dict:
    saving = (val_base - champion_cost) / val_base * 100
    gap = (champion_cost - val_oracle) / val_oracle * 100
    return {
        "steps": steps,
        "candidate_val_cost_vnd": candidate_cost,
        "champion_val_cost_vnd": champion_cost,
        "val_cost_vnd": champion_cost,
        "accepted": accepted,
        "oracle_gap_pct": gap,
        "saving_vs_nobess_pct": saving,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--e-cap", type=float, required=True)
    parser.add_argument("--p-rated", type=float, required=True)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--gamma", type=float, default=PPO_GAMMA)
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=PPO_LAMBDA)
    parser.add_argument("--control-dt-minutes", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--billing", choices=("2tc", "tou"), default="2tc")
    parser.add_argument("--training-config", type=str, required=True)
    parser.add_argument("--oracle-cache", required=True)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--obs-variant", choices=("brain7",), default="brain7")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not math.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        raise SystemExit("gamma must be finite and in (0, 1]")
    if not math.isfinite(args.lambda_value) or not 0.0 <= args.lambda_value <= 1.0:
        raise SystemExit("lambda must be finite and in [0, 1]")
    if not math.isclose(
        args.control_dt_minutes,
        PPO_FIT_CONTROL_DT_MINUTES,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise SystemExit(
            "Generic PPO fit mode requires 30-minute control aligned to demand-meter blocks"
        )

    configure_ppo_determinism(args.seed)

    days = load_training_days(args.csv, weather="csv")
    if not days:
        raise SystemExit("Training CSV contains no usable days")
    csv_dt = 24.0 / len(days[0].load)

    # TEMP DEBUG MODE: intentionally leak the full dataset into all three roles.
    # This measures whether PPO can fit the supplied month at all; it is NOT a
    # valid generalization test and must be reverted before production evaluation.
    train_days = list(days)
    val_days = list(days)
    test_days = list(days)
    peak = max(float(day.load.max()) for day in days)
    p_ref = math.ceil(peak / 500.0) * 500.0

    cfg, billing = build_training_bess_config(
        args.e_cap,
        args.p_rated,
        csv_dt,
        args.training_config,
        default_billing=args.billing,
    )
    battery_wear_cost = _load_ui_wear_cost(args.training_config)

    tag = args.tag or f"ds_{args.e_cap:.0f}kwh_{args.p_rated:.0f}kw"
    if billing == "tou" and not tag.endswith("_tou"):
        tag += "_tou"

    # TEMP DEBUG MODE: keep all supplied days in one continuous training
    # episode instead of calendar-month filtering, so every one of the 30 days
    # participates in training as well as validation/test.
    train_months = [MonthData(days=train_days, source="train:full-overlap")]
    val_month = MonthData(days=val_days, source="val:full-overlap")
    gamma = args.gamma
    native_steps = native_steps_per_action(cfg.dt, args.control_dt_minutes)
    decisions_per_day = len(days[0].load) // native_steps
    decisions_per_episode = decisions_per_day * len(train_days)
    learner_device = resolve_ppo_device(args.device)
    agent = PPOAgent(
        OBSERVATION_DIM,
        gamma=gamma,
        lam=args.lambda_value,
        seed=args.seed,
        device=learner_device,
    )
    agent.meta = {
        "p_ref_kw": p_ref,
        "e_cap_kwh": args.e_cap,
        "p_rated_kw": args.p_rated,
        "obs_variant": "brain7",
        "obs_dim": OBSERVATION_DIM,
        "battery_wear_cost": battery_wear_cost,
        "reward_mode": "brain_savings_vnd_v1",
        "training_reward_shaping": "infeasible_request_phantom_wear_scaled_v2",
        "training_reward_shaping_scale": ACTION_MISMATCH_SHAPING_SCALE,
        "initial_soc": float(cfg.SOC_min),
        "gamma": gamma,
        "lambda": args.lambda_value,
        "learning_rate": float(agent.opt.param_groups[0]["lr"]),
        "ppo_epochs": agent.epochs,
        "entropy_coef": agent.ent_coef,
        "target_kl": agent.target_kl,
        "native_dt_minutes": csv_dt * 60.0,
        "control_dt_minutes": args.control_dt_minutes,
        "native_steps_per_action": native_steps,
        "billing_mode": billing,
        "device_requested": args.device,
        "device": learner_device,
        "seed": args.seed,
        "deterministic_training": True,
        "train_csv": str(args.csv),
        "test_range": [test_days[0].date_iso, test_days[-1].date_iso],
        "temporary_full_dataset_overlap": True,
        "data_overlap_note": "TEMP DEBUG: train/validation/test all use the full dataset",
        "training_augmentation_enabled": False,
        "rollout_decisions": decisions_per_episode,
        "validation_every_updates": VALIDATE_EVERY_UPDATES,
        "challenger_reset_patience": CHALLENGER_RESET_PATIENCE,
        "challenger_resets_enabled": CHALLENGER_RESETS_ENABLED,
        "oracle_behavior_cloning_enabled": True,
        "oracle_behavior_cloning_max_epochs": ORACLE_BC_MAX_EPOCHS,
        "oracle_behavior_cloning_learning_rate": ORACLE_BC_LEARNING_RATE,
    }
    buffer = RolloutBuffer(decisions_per_episode, OBSERVATION_DIM)

    val_base = score_month(run_no_bess(val_month, cfg)["p_grid_days"], cfg, days=val_days)["total_cost_vnd"]
    val_day_indexes = [day.day_index for day in val_days]
    train_day_indexes = [day.day_index for day in train_days]
    oracle_grids = load_cached_training_grids(args.oracle_cache, val_day_indexes)
    oracle_dispatch = load_cached_training_dispatch(args.oracle_cache, train_day_indexes)
    val_oracle_dispatch = (
        oracle_dispatch
        if train_day_indexes == val_day_indexes
        else load_cached_training_dispatch(args.oracle_cache, val_day_indexes)
    )
    val_oracle_utility = score_month(oracle_grids, cfg, days=val_days)["total_cost_vnd"]
    val_oracle_wear = _oracle_dispatch_wear_cost_vnd(
        val_oracle_dispatch,
        timestep_hours=cfg.dt,
        wear_vnd_per_kwh=battery_wear_cost,
    )
    val_oracle = val_oracle_utility + val_oracle_wear
    curve_path = RESULTS_DIR / f"training_curve_{tag}.csv"
    report_path = RESULTS_DIR / f"training_report_{tag}.json"
    report = {
        "version": 1,
        "status": "running",
        "algorithm": "ppo",
        "tag": tag,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": str(args.csv),
            "total_days": len(days),
            "train_days": len(train_days),
            "validation_days": len(val_days),
            "test_days": len(test_days),
            "validation_range": [val_days[0].date_iso, val_days[-1].date_iso],
            "test_range": [test_days[0].date_iso, test_days[-1].date_iso],
            "temporary_full_dataset_overlap": True,
        },
        "battery": {"e_cap_kwh": args.e_cap, "p_rated_kw": args.p_rated},
        "economics": {
            "battery_wear_cost": battery_wear_cost,
            "reward_mode": "brain_savings_vnd_v1",
            "training_reward_shaping": "infeasible_request_phantom_wear_scaled_v2",
            "training_reward_shaping_scale": ACTION_MISMATCH_SHAPING_SCALE,
            "champion_scoring_uses_shaping": False,
        },
        "training": {
            "requested_steps": args.steps,
            "seed": args.seed,
            "gamma": gamma,
            "lambda": args.lambda_value,
            "learning_rate": float(agent.opt.param_groups[0]["lr"]),
            "ppo_epochs": agent.epochs,
            "entropy_coef": agent.ent_coef,
            "target_kl": agent.target_kl,
            "initial_soc": float(cfg.SOC_min),
            "rollout_days": len(train_days),
            "rollout_decisions": decisions_per_episode,
            "augmentation_enabled": False,
            "validation_every_updates": VALIDATE_EVERY_UPDATES,
            "challenger_reset_patience": CHALLENGER_RESET_PATIENCE,
            "challenger_resets_enabled": CHALLENGER_RESETS_ENABLED,
            "oracle_behavior_cloning_enabled": True,
            "oracle_behavior_cloning_max_epochs": ORACLE_BC_MAX_EPOCHS,
            "oracle_behavior_cloning_learning_rate": ORACLE_BC_LEARNING_RATE,
            "native_dt_minutes": csv_dt * 60.0,
            "control_dt_minutes": args.control_dt_minutes,
            "native_steps_per_action": native_steps,
            "device_requested": args.device,
            "device": learner_device,
            "updates": 0,
            "candidate_evaluations": 0,
            "accepted_updates": 0,
            "rejected_updates": 0,
            "learner_resets": 0,
            "acceptance_rate_pct": 0.0,
        },
        "billing_mode": billing,
        "p_ref_kw": p_ref,
        "validation": {
            "no_bess_vnd": val_base,
            "oracle_vnd": val_oracle,
            "oracle_utility_vnd": val_oracle_utility,
            "oracle_wear_vnd": val_oracle_wear,
        },
    }
    write_curve(curve_path, [], fields=PPO_CHAMPION_CURVE_FIELDS)
    write_report(report_path, report)
    print(
        f"[train-ds] TEMP FULL-DATASET OVERLAP | {len(days)} days | "
        f"train {len(train_days)} / val {len(val_days)} / test {len(test_days)} | "
        f"gamma {gamma:g} | lambda {args.lambda_value:g} | "
        f"UI wear {battery_wear_cost:g} VND/kWh | "
        f"learner {learner_device} (requested {args.device}) | BrainEnv eyes={OBSERVATION_DIM} | "
        f"native dt {csv_dt * 60:g}m | control dt {args.control_dt_minutes:g}m | "
        f"p_ref {p_ref:.0f} | val no-BESS {val_base/1e6:.0f}M, oracle {val_oracle/1e6:.0f}M",
        flush=True,
    )

    curve = []
    perf = {
        "rollout": 0.0,
        "update": 0.0,
        "validation": 0.0,
        "scoring": 0.0,
        "io": 0.0,
        "checkpoint": 0.0,
        "decisions": 0,
        "native_rows": 0,
    }
    checkpoint_path = RESULTS_DIR / f"policy_{tag}.pt"

    def validate_policy_cost() -> float:
        validation_started = time.perf_counter()
        result = run_drl_policy(val_month, cfg, agent, p_ref_kw=p_ref)
        perf["validation"] += time.perf_counter() - validation_started
        scoring_started = time.perf_counter()
        val_cost = _score_ppo_operating_month(
            result["p_grid_days"],
            result["p_bess_days"],
            cfg,
            battery_wear_cost,
            val_days,
        )["total_operating_cost_vnd"]
        perf["scoring"] += time.perf_counter() - scoring_started
        return val_cost

    def persist_progress() -> None:
        started_io = time.perf_counter()
        write_curve(curve_path, curve, fields=PPO_CHAMPION_CURVE_FIELDS)
        if curve:
            report["latest"] = curve[-1]
            if (
                "best" not in report
                or curve[-1]["val_cost_vnd"] < report["best"]["val_cost_vnd"]
            ):
                report["best"] = curve[-1]
        write_report(report_path, report)
        perf["io"] += time.perf_counter() - started_io

    def print_performance() -> None:
        rollout_seconds = max(perf["rollout"], 1e-9)
        print(
            f"  perf rollout {perf['rollout']:.2f}s | "
            f"ppo {perf['update']:.2f}s | validate {perf['validation']:.2f}s | "
            f"score {perf['scoring']:.2f}s | io {perf['io']:.2f}s | "
            f"checkpoint {perf['checkpoint']:.2f}s | "
            f"{perf['native_rows']/rollout_seconds:,.0f} native rows/s | "
            f"{perf['decisions']/rollout_seconds:,.0f} decisions/s",
            flush=True,
        )
        for key in perf:
            perf[key] = 0 if key in ("decisions", "native_rows") else 0.0

    def make_training_env(month):
        return make_brain_env(
            month,
            cfg,
            power_scale_kw=p_ref,
            battery_wear_vnd_per_kwh=battery_wear_cost,
        )

    raw_initial_cost = validate_policy_cost()
    raw_initial_state = agent.snapshot_training_state()
    teacher_observations, teacher_targets = _collect_oracle_teacher_samples(
        train_months[0],
        oracle_dispatch,
        cfg,
        power_scale_kw=p_ref,
        battery_wear_cost=battery_wear_cost,
        native_steps=native_steps,
    )
    bc_started = time.perf_counter()
    bc_stats = _behavior_clone_actor(
        agent,
        teacher_observations,
        teacher_targets,
        seed=args.seed,
    )
    bc_stats["seconds"] = time.perf_counter() - bc_started
    bc_stats["teacher_mean_abs_action"] = float(np.mean(np.abs(teacher_targets)))
    bc_stats["teacher_nonzero_action_pct"] = float(
        np.mean(np.abs(teacher_targets) > 1e-6) * 100.0
    )
    post_bc_cost = validate_policy_cost()
    use_bc = post_bc_cost < raw_initial_cost
    if not use_bc:
        agent.restore_training_state(raw_initial_state)

    agent.meta["oracle_behavior_cloning"] = {
        **bc_stats,
        "raw_initial_validation_cost_vnd": raw_initial_cost,
        "post_bc_validation_cost_vnd": post_bc_cost,
        "selected_for_ppo": use_bc,
    }
    report["training"]["oracle_behavior_cloning"] = dict(
        agent.meta["oracle_behavior_cloning"]
    )
    write_report(report_path, report)
    print(
        f"[train-ds] ORACLE TEACHER | {bc_stats['samples']} lessons | "
        f"MSE {bc_stats['initial_mse']:.5f}->{bc_stats['final_mse']:.5f} | "
        f"raw {raw_initial_cost/1e6:.1f}M -> BC {post_bc_cost/1e6:.1f}M | "
        f"{'USE BC' if use_bc else 'KEEP RAW'}",
        flush=True,
    )

    champion_cost = _initialize_champion(agent, validate_policy_cost, checkpoint_path)
    champion_state = agent.snapshot_training_state()
    initial_point = _champion_curve_point(
        steps=0,
        candidate_cost=champion_cost,
        champion_cost=champion_cost,
        accepted=True,
        val_base=val_base,
        val_oracle=val_oracle,
    )
    curve.append(initial_point)
    persist_progress()
    print(
        f"  champion #0 | step {0:>7} | val {champion_cost/1e6:8.1f}M | "
        f"saving {initial_point['saving_vs_nobess_pct']:5.1f}% | "
        f"gap {initial_point['oracle_gap_pct']:6.1f}% | initial checkpoint",
        flush=True,
    )

    # Fit-test means fit the exact supplied episode. Do not quietly mutate the
    # load/PV data while asking whether PPO can memorize it.
    env = make_training_env(train_months[0])
    obs = observation_array(env.reset())
    steps = 0
    updates = 0
    candidate_evaluations = 0
    consecutive_rejections = 0
    last_validated_update = 0
    rollout_action_sum = 0.0
    rollout_action_abs_sum = 0.0
    rollout_action_saturation = 0
    rollout_projected = 0
    rollout_mismatch_penalty_vnd = 0.0
    rollout_mismatch_kwh = 0.0
    rollout_count = 0
    started = time.time()
    rollout_started = time.perf_counter()

    def evaluate_live_candidate(*, allow_reset: bool) -> tuple[dict, float, bool, bool]:
        nonlocal champion_cost
        nonlocal champion_state
        nonlocal candidate_evaluations
        nonlocal consecutive_rejections
        nonlocal last_validated_update

        candidate_evaluations += 1
        candidate_cost = validate_policy_cost()
        champion_cost, accepted = _resolve_challenger(champion_cost, candidate_cost)
        reset_to_champion = False
        if accepted:
            report["training"]["accepted_updates"] += 1
            consecutive_rejections = 0
            champion_state = agent.snapshot_training_state()
            checkpoint_started = time.perf_counter()
            if _save_accepted_champion(agent, checkpoint_path, accepted=True):
                perf["checkpoint"] += time.perf_counter() - checkpoint_started
        else:
            report["training"]["rejected_updates"] += 1
            consecutive_rejections += 1
            if (
                CHALLENGER_RESETS_ENABLED
                and allow_reset
                and consecutive_rejections >= CHALLENGER_RESET_PATIENCE
            ):
                agent.restore_training_state(champion_state)
                report["training"]["learner_resets"] += 1
                consecutive_rejections = 0
                reset_to_champion = True

        last_validated_update = updates
        report["training"]["candidate_evaluations"] = candidate_evaluations
        report["training"]["consecutive_rejections"] = consecutive_rejections
        report["training"]["acceptance_rate_pct"] = (
            report["training"]["accepted_updates"] / candidate_evaluations * 100.0
        )
        point = _champion_curve_point(
            steps=steps,
            candidate_cost=candidate_cost,
            champion_cost=champion_cost,
            accepted=accepted,
            val_base=val_base,
            val_oracle=val_oracle,
        )
        curve.append(point)
        persist_progress()
        return point, candidate_cost, accepted, reset_to_champion

    while steps < args.steps:
        action, logp, latent, value = agent.act_with_latent(obs)
        transition = step_brain_control(
            env,
            action,
            native_steps=native_steps,
        )
        done = transition.done
        mismatch_penalty_vnd = _action_mismatch_penalty_vnd(
            transition,
            timestep_hours=cfg.dt,
            wear_vnd_per_kwh=battery_wear_cost,
        )
        mismatch_kwh = (
            mismatch_penalty_vnd / battery_wear_cost
            if battery_wear_cost > 0.0
            else 0.0
        )
        # Keep the PPO action/log-prob pair exact. We shape only the learning
        # reward so repeatedly requesting battery power that physics rejects is
        # no longer free. Champion/test evaluation never includes this penalty.
        reward = (
            transition.reward_million_vnd
            - ACTION_MISMATCH_SHAPING_SCALE * mismatch_penalty_vnd / REWARD_SCALE_VND
        )
        buffer.add(obs, action, logp, reward, value, float(done), latent=latent)
        steps += 1
        perf["decisions"] += 1
        perf["native_rows"] += len(transition.native_results)
        rollout_action_sum += action
        rollout_action_abs_sum += abs(action)
        rollout_action_saturation += int(abs(action) >= 0.98)
        rollout_projected += int(transition.adjusted_action)
        rollout_mismatch_penalty_vnd += mismatch_penalty_vnd
        rollout_mismatch_kwh += mismatch_kwh
        rollout_count += 1

        if done:
            env = make_training_env(train_months[0])
            next_obs = observation_array(env.reset())
        else:
            if transition.next_observation is None:
                raise RuntimeError("BrainEnv omitted next observation before episode end")
            next_obs = observation_array(transition.next_observation)
        obs = next_obs
        if not buffer.full():
            continue

        perf["rollout"] += time.perf_counter() - rollout_started
        updates += 1
        _, _, last_value = agent.act(obs)
        update_started = time.perf_counter()
        update_stats = agent.update(buffer, 0.0 if done else last_value)
        perf["update"] += time.perf_counter() - update_started
        rollout_stats = {
            "mean_action": rollout_action_sum / rollout_count,
            "mean_abs_action": rollout_action_abs_sum / rollout_count,
            "action_saturation_pct": rollout_action_saturation / rollout_count * 100.0,
            "projected_action_pct": rollout_projected / rollout_count * 100.0,
            "action_mismatch_kwh": rollout_mismatch_kwh,
            "action_mismatch_penalty_vnd": rollout_mismatch_penalty_vnd,
            "action_mismatch_shaping_penalty_vnd": (
                ACTION_MISMATCH_SHAPING_SCALE * rollout_mismatch_penalty_vnd
            ),
            "mean_action_mismatch_penalty_vnd": rollout_mismatch_penalty_vnd / rollout_count,
        }
        report["training"]["updates"] = updates
        report["training"]["last_update"] = {**update_stats, **rollout_stats}

        should_validate = updates % VALIDATE_EVERY_UPDATES == 0
        should_log = updates == 1 or updates % LOG_EVERY_UPDATES == 0
        if should_validate:
            point, candidate_cost, accepted, reset_to_champion = evaluate_live_candidate(
                allow_reset=True
            )
            if should_log:
                verdict = "ACCEPTED 👑" if accepted else "REJECTED 💀"
                reset_note = " | RESET→CHAMPION" if reset_to_champion else ""
                print(
                    f"  update {updates:>4} | step {steps:>7} | "
                    f"candidate {candidate_cost/1e6:8.1f}M | champion {champion_cost/1e6:8.1f}M | "
                    f"{verdict}{reset_note} | saving {point['saving_vs_nobess_pct']:5.1f}% | "
                    f"gap {point['oracle_gap_pct']:6.1f}% | {steps/(time.time()-started):,.0f} sps",
                    flush=True,
                )
        elif should_log:
            print(
                f"  update {updates:>4} | step {steps:>7} | learner continues | "
                f"KL {update_stats['approx_kl']:.4f} | clip {update_stats['clip_fraction']*100:4.1f}% | "
                f"EV {update_stats['explained_variance']:+.3f} | "
                f"{steps/(time.time()-started):,.0f} sps",
                flush=True,
            )

        if should_log:
            print(
                f"  champion stats | accepted {report['training']['accepted_updates']} / "
                f"{candidate_evaluations} evals | rejected {report['training']['rejected_updates']} / "
                f"{candidate_evaluations} | resets {report['training']['learner_resets']} | "
                f"rate {report['training']['acceptance_rate_pct']:.1f}%",
                flush=True,
            )
            print_performance()

        rollout_action_sum = 0.0
        rollout_action_abs_sum = 0.0
        rollout_action_saturation = 0
        rollout_projected = 0
        rollout_mismatch_penalty_vnd = 0.0
        rollout_mismatch_kwh = 0.0
        rollout_count = 0
        rollout_started = time.perf_counter()

    # The requested step budget may end after 1-3 unvalidated full PPO updates.
    # Score that final learner once so a late improvement is not thrown away.
    if updates > last_validated_update:
        point, candidate_cost, accepted, _reset_to_champion = evaluate_live_candidate(
            allow_reset=False
        )
        verdict = "ACCEPTED 👑" if accepted else "REJECTED 💀"
        print(
            f"  final candidate | update {updates:>4} | step {steps:>7} | "
            f"candidate {candidate_cost/1e6:8.1f}M | champion {champion_cost/1e6:8.1f}M | "
            f"{verdict} | saving {point['saving_vs_nobess_pct']:5.1f}% | "
            f"gap {point['oracle_gap_pct']:6.1f}%",
            flush=True,
        )

    persist_progress()

    test_month = MonthData(days=test_days, source="test")
    best_agent = PPOAgent(OBSERVATION_DIM, device=learner_device)
    best_agent.load(RESULTS_DIR / f"policy_{tag}.pt")
    result = run_drl_policy(test_month, cfg, best_agent, p_ref_kw=p_ref)
    test_operating = _score_ppo_operating_month(
        result["p_grid_days"], result["p_bess_days"], cfg,
        battery_wear_cost, test_days,
    )
    test_cost = test_operating["total_operating_cost_vnd"]
    no_bess_cost = score_month(run_no_bess(test_month, cfg)["p_grid_days"], cfg, days=test_days)["total_cost_vnd"]
    test_saving = (no_bess_cost - test_cost) / no_bess_cost * 100
    best_agent.meta = {
        **agent.meta,
        "test_saving_pct": round(test_saving, 2),
        "trained": datetime.now(timezone.utc).astimezone().date().isoformat(),
    }
    best_agent.save(RESULTS_DIR / f"policy_{tag}.pt")
    report["status"] = "complete"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["test"] = {
        "policy_cost_vnd": test_cost,
        "no_bess_vnd": no_bess_cost,
        "saving_pct": test_saving,
        "wear_cost_vnd": test_operating["wear_cost_vnd"],
        "throughput_kwh": test_operating["throughput_kwh"],
        "initial_soc": float(cfg.SOC_min),
        "final_soc": float(result["soc_days"][-1][-1]),
    }
    write_report(report_path, report)
    print(
        f"[train-ds] TEST {test_days[0].date_iso}->{test_days[-1].date_iso}: "
        f"{test_cost/1e6:.1f}M vs no-BESS {no_bess_cost/1e6:.1f}M -> saving {test_saving:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
