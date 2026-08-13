from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bess.agents.ppo_agent import (
    PPOAgent,
    RolloutBuffer,
    configure_ppo_determinism,
    resolve_ppo_device,
)
from bess.core.bess_env import OBSERVATION_DIM
from bess.core.brain_runtime import (
    make_brain_env,
    native_steps_per_action,
    observation_array,
    step_brain_control,
)
from bess.core.common import RESULTS_DIR, score_month
from bess.core.scenario_gen import MonthData
from bess.core.settings import PPO_GAMMA, PPO_LAMBDA
from bess.evaluation.baselines import run_drl_policy, run_no_bess
from bess.evaluation.oracle.oracle_cache import load_cached_training_grids
from bess.training.training_common import (
    augment_month,
    build_training_bess_config,
    load_training_days,
)
from bess.training.training_reports import (
    PPO_CHAMPION_CURVE_FIELDS,
    write_curve,
    write_report,
)

ROLLOUT_DAYS = 32
LOG_EVERY_UPDATES = 1


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


def _initialize_champion(agent: PPOAgent, validate_cost, checkpoint_path: Path) -> float:
    """Validate and persist Champion #0 before any PPO update can run."""
    champion_cost = float(validate_cost())
    agent.save(checkpoint_path)
    return champion_cost


def _resolve_challenger(
    agent: PPOAgent,
    champion_state: dict,
    champion_cost: float,
    candidate_cost: float,
) -> tuple[float, bool]:
    """Accept only strict cost improvements; otherwise restore the Champion learner."""
    if candidate_cost < champion_cost:
        return candidate_cost, True
    agent.restore_training_state(champion_state)
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
        "gamma": gamma,
        "lambda": args.lambda_value,
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
    }
    decisions_per_day = len(days[0].load) // native_steps
    buffer = RolloutBuffer(decisions_per_day * ROLLOUT_DAYS, OBSERVATION_DIM)

    val_base = score_month(run_no_bess(val_month, cfg)["p_grid_days"], cfg, days=val_days)["total_cost_vnd"]
    oracle_grids = load_cached_training_grids(
        args.oracle_cache, [day.day_index for day in val_days]
    )
    val_oracle = score_month(oracle_grids, cfg, days=val_days)["total_cost_vnd"]
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
        },
        "training": {
            "requested_steps": args.steps,
            "seed": args.seed,
            "gamma": gamma,
            "lambda": args.lambda_value,
            "rollout_days": ROLLOUT_DAYS,
            "native_dt_minutes": csv_dt * 60.0,
            "control_dt_minutes": args.control_dt_minutes,
            "native_steps_per_action": native_steps,
            "device_requested": args.device,
            "device": learner_device,
            "accepted_updates": 0,
            "rejected_updates": 0,
            "acceptance_rate_pct": 0.0,
        },
        "billing_mode": billing,
        "p_ref_kw": p_ref,
        "validation": {"no_bess_vnd": val_base, "oracle_vnd": val_oracle},
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

    rng = np.random.default_rng(args.seed)
    curve = []
    perf = {
        "augment": 0.0,
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
            f"  perf rollout {perf['rollout']:.2f}s | augment {perf['augment']:.2f}s | "
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

    champion_cost = _initialize_champion(agent, validate_policy_cost, checkpoint_path)
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

    month_index = 0
    augment_started = time.perf_counter()
    first_month = augment_month(train_months[0], rng)
    perf["augment"] += time.perf_counter() - augment_started
    env = make_training_env(first_month)
    obs = observation_array(env.reset())
    steps = 0
    updates = 0
    started = time.time()
    rollout_started = time.perf_counter()
    while steps < args.steps:
        action, logp, latent, value = agent.act_with_latent(obs)
        transition = step_brain_control(
            env,
            action,
            native_steps=native_steps,
        )
        done = transition.done
        reward = transition.reward_million_vnd
        buffer.add(obs, action, logp, reward, value, float(done), latent=latent)
        steps += 1
        perf["decisions"] += 1
        perf["native_rows"] += len(transition.native_results)
        if done:
            month_index += 1
            augment_started = time.perf_counter()
            source_month = train_months[month_index % len(train_months)]
            next_month = augment_month(source_month, rng)
            perf["augment"] += time.perf_counter() - augment_started
            env = make_training_env(next_month)
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
        champion_state = agent.snapshot_training_state()
        update_started = time.perf_counter()
        agent.update(buffer, 0.0 if done else last_value)
        perf["update"] += time.perf_counter() - update_started
        candidate_cost = validate_policy_cost()
        champion_cost, accepted = _resolve_challenger(
            agent,
            champion_state,
            champion_cost,
            candidate_cost,
        )
        if accepted:
            report["training"]["accepted_updates"] += 1
        else:
            report["training"]["rejected_updates"] += 1
        checkpoint_started = time.perf_counter()
        if _save_accepted_champion(agent, checkpoint_path, accepted):
            perf["checkpoint"] += time.perf_counter() - checkpoint_started
        report["training"]["acceptance_rate_pct"] = (
            report["training"]["accepted_updates"] / updates * 100.0
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
        should_log = updates == 1 or updates % LOG_EVERY_UPDATES == 0
        if should_log:
            verdict = "ACCEPTED 👑" if accepted else "REJECTED 💀"
            print(
                f"  update {updates:>4} | step {steps:>7} | "
                f"candidate {candidate_cost/1e6:8.1f}M | champion {champion_cost/1e6:8.1f}M | "
                f"{verdict} | saving {point['saving_vs_nobess_pct']:5.1f}% | "
                f"gap {point['oracle_gap_pct']:6.1f}% | {steps/(time.time()-started):,.0f} sps",
                flush=True,
            )
            print(
                f"  champion stats | accepted {report['training']['accepted_updates']} / {updates} | "
                f"rejected {report['training']['rejected_updates']} / {updates} | "
                f"rate {report['training']['acceptance_rate_pct']:.1f}%",
                flush=True,
            )
            print_performance()
        rollout_started = time.perf_counter()

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
    }
    write_report(report_path, report)
    print(
        f"[train-ds] TEST {test_days[0].date_iso}->{test_days[-1].date_iso}: "
        f"{test_cost/1e6:.1f}M vs no-BESS {no_bess_cost/1e6:.1f}M -> saving {test_saving:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
