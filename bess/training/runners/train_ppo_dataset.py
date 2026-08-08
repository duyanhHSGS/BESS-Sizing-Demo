from __future__ import annotations

import argparse
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from bess.evaluation.baselines import run_drl_policy, run_no_bess
from bess.core.bess_env import BESSEnv
from bess.evaluation.benchmark import _rolling_30_minute_average
from bess.core.common import RESULTS_DIR, score_month
from bess.agents.ppo_agent import PPOAgent, RolloutBuffer, resolve_ppo_device
from bess.core.scenario_gen import MonthData
from bess.evaluation.oracle.oracle_cache import load_cached_training_grids
from bess.training.training_common import (
    augment_month,
    build_training_bess_config,
    load_training_days,
    month_blocks,
)
from bess.training.training_reports import write_curve, write_report
from bess.core.settings import PPO_GAMMA, PPO_LAMBDA
from bess.forecasting.weather_forecast import build_forecast_bundle, fit_attach_forecasts


ROLLOUT_DAYS = 32
LOG_EVERY_UPDATES = 1


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
    parser.add_argument("--obs-variant", choices=("base", "fc"), default="base")
    parser.add_argument("--weather-data", default="")
    parser.add_argument("--forecast-artifact", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not math.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        raise SystemExit("gamma must be finite and in (0, 1]")
    if not math.isfinite(args.lambda_value) or not 0.0 <= args.lambda_value <= 1.0:
        raise SystemExit("lambda must be finite and in [0, 1]")

    days = load_training_days(args.csv, weather="csv")
    if args.val_days < 1 or args.test_days < 1:
        raise SystemExit("Validation days and test days must both be at least 1")
    split_days = args.val_days + args.test_days
    if len(days) <= split_days:
        raise SystemExit(
            f"Need more than {split_days} days for train/val/test split; found {len(days)}"
        )
    csv_dt = 24.0 / len(days[0].load)

    test_days = days[-args.test_days:]
    val_days = days[-split_days:-args.test_days]
    train_days = days[:-split_days]
    peak = max(float(day.load.max()) for day in days)
    p_ref = math.ceil(peak / 500.0) * 500.0
    daily_peaks = [
        max(_rolling_30_minute_average(np.maximum(0.0, day.load - day.pv), 24.0 / len(day.load)), default=0.0)
        for day in train_days
    ]
    d_run0 = 0.5 * float(np.mean(daily_peaks))
    forecast_model = None
    if args.obs_variant == "fc":
        if not args.weather_data or not args.forecast_artifact:
            raise SystemExit("forecast mode requires --weather-data and --forecast-artifact")
        forecast_model = fit_attach_forecasts(
            days, Path(args.weather_data), len(train_days),
            Path(args.forecast_artifact), p_ref,
        )

    cfg, billing = build_training_bess_config(
        args.e_cap,
        args.p_rated,
        csv_dt,
        args.training_config,
        default_billing=args.billing,
    )

    tag = args.tag or f"ds_{args.e_cap:.0f}kwh_{args.p_rated:.0f}kw"
    if billing == "tou" and not tag.endswith("_tou"):
        tag += "_tou"

    train_months = month_blocks(train_days)
    val_month = MonthData(days=val_days, source="val")
    gamma = args.gamma
    env = BESSEnv(
        cfg,
        reference_power_kw=p_ref,
        initial_running_peak_kw=d_run0,
        discount_factor=gamma,
        control_interval_minutes=args.control_dt_minutes,
        forecast_enabled=args.obs_variant == "fc",
    )
    learner_device = resolve_ppo_device(args.device)
    agent = PPOAgent(
        env.observation_dimensions,
        gamma=gamma,
        lam=args.lambda_value,
        seed=args.seed,
        device=learner_device,
    )
    assert env.discount_factor == agent.gamma
    agent.meta = {
        "p_ref_kw": p_ref,
        "e_cap_kwh": args.e_cap,
        "p_rated_kw": args.p_rated,
        "obs_variant": args.obs_variant,
        "obs_dim": env.observation_dimensions,
        "d_run_init_kw": d_run0,
        "gamma": gamma,
        "lambda": args.lambda_value,
        "native_dt_minutes": csv_dt * 60.0,
        "control_dt_minutes": env.control_interval_minutes,
        "native_steps_per_action": env.native_samples_per_action,
        "billing_mode": billing,
        "device_requested": args.device,
        "device": learner_device,
        "train_csv": str(args.csv),
        "test_range": [test_days[0].date_iso, test_days[-1].date_iso],
    }
    if forecast_model:
        agent.meta["forecast_model"] = forecast_model["model"]
        agent.meta["forecast_artifact"] = forecast_model["artifact"]
        agent.meta["forecast_model_artifact"] = forecast_model["model_artifact"]
        agent.meta["weather_data"] = str(Path(args.weather_data))
        agent.meta["forecast_embedded"] = True
        agent.forecast_bundle = build_forecast_bundle(days)
    decisions_per_day = len(days[0].load) // env.native_samples_per_action
    buffer = RolloutBuffer(decisions_per_day * ROLLOUT_DAYS, env.observation_dimensions)

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
        },
        "battery": {"e_cap_kwh": args.e_cap, "p_rated_kw": args.p_rated},
        "training": {
            "requested_steps": args.steps,
            "seed": args.seed,
            "gamma": gamma,
            "lambda": args.lambda_value,
            "rollout_days": ROLLOUT_DAYS,
            "native_dt_minutes": csv_dt * 60.0,
            "control_dt_minutes": env.control_interval_minutes,
            "native_steps_per_action": env.native_samples_per_action,
            "device_requested": args.device,
            "device": learner_device,
        },
        "billing_mode": billing,
        "p_ref_kw": p_ref,
        "d_run_init_kw": d_run0,
        "validation": {"no_bess_vnd": val_base, "oracle_vnd": val_oracle},
    }
    write_curve(curve_path, [])
    write_report(report_path, report)
    print(
        f"[train-ds] {len(days)} days | train {len(train_days)} / "
        f"val {len(val_days)} / test {len(test_days)} | "
        f"gamma {gamma:g} | lambda {args.lambda_value:g} | "
        f"learner {learner_device} (requested {args.device}) | "
        f"native dt {csv_dt * 60:g}m | control dt {env.control_interval_minutes:g}m | "
        f"p_ref {p_ref:.0f} | val no-BESS {val_base/1e6:.0f}M, oracle {val_oracle/1e6:.0f}M",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    best_val = float("inf")
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

    def persist_progress() -> None:
        started_io = time.perf_counter()
        write_curve(curve_path, curve)
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

    month_index = 0
    augment_started = time.perf_counter()
    first_month = train_months[0] if args.obs_variant == "fc" else augment_month(train_months[0], rng)
    perf["augment"] += time.perf_counter() - augment_started
    obs = env.reset(first_month)
    steps = 0
    updates = 0
    started = time.time()
    rollout_started = time.perf_counter()
    while steps < args.steps:
        action, logp, value = agent.act(obs)
        next_obs, reward, done, step_info = env.step(action)
        buffer.add(obs, action, logp, reward, value, float(done))
        steps += 1
        perf["decisions"] += 1
        perf["native_rows"] += int(step_info["native_rows"])
        if done:
            month_index += 1
            augment_started = time.perf_counter()
            source_month = train_months[month_index % len(train_months)]
            next_month = source_month if args.obs_variant == "fc" else augment_month(source_month, rng)
            perf["augment"] += time.perf_counter() - augment_started
            next_obs = env.reset(next_month)
        obs = next_obs
        if not buffer.full():
            continue
        perf["rollout"] += time.perf_counter() - rollout_started
        updates += 1
        update_started = time.perf_counter()
        _, _, last_value = agent.act(obs)
        agent.update(buffer, 0.0 if done else last_value)
        perf["update"] += time.perf_counter() - update_started
        validation_started = time.perf_counter()
        result = run_drl_policy(val_month, cfg, agent, p_ref_kw=p_ref)
        perf["validation"] += time.perf_counter() - validation_started
        scoring_started = time.perf_counter()
        val_cost = score_month(result["p_grid_days"], cfg, days=val_days)["total_cost_vnd"]
        perf["scoring"] += time.perf_counter() - scoring_started
        saving = (val_base - val_cost) / val_base * 100
        gap = (val_cost - val_oracle) / val_oracle * 100
        curve.append({"steps": steps, "val_cost_vnd": val_cost, "oracle_gap_pct": gap, "saving_vs_nobess_pct": saving})
        persist_progress()
        is_best = val_cost < best_val
        if val_cost < best_val:
            best_val = val_cost
            checkpoint_started = time.perf_counter()
            agent.save(RESULTS_DIR / f"policy_{tag}.pt")
            perf["checkpoint"] += time.perf_counter() - checkpoint_started
        should_log = updates == 1 or updates % LOG_EVERY_UPDATES == 0
        if should_log:
            print(
                f"  update {updates:>4} | step {steps:>7} | val {val_cost/1e6:8.1f}M | "
                f"saving {saving:5.1f}% | gap {gap:6.1f}% | {steps/(time.time()-started):,.0f} sps",
                flush=True,
            )
            print_performance()
        if is_best:
            print(
                f"  best   {updates:>4} | step {steps:>7} | val {val_cost/1e6:8.1f}M | "
                f"saving {saving:5.1f}% | gap {gap:6.1f}% | checkpoint updated",
                flush=True,
            )
        elif updates % LOG_EVERY_UPDATES == 0:
            print(
                f"  step {steps:>7} | val {val_cost/1e6:8.1f}M | saving {saving:5.1f}% | "
                f"gap {gap:6.1f}% | no new best",
                flush=True,
            )
        rollout_started = time.perf_counter()

    if not curve:
        result = run_drl_policy(val_month, cfg, agent, p_ref_kw=p_ref)
        val_cost = score_month(result["p_grid_days"], cfg, days=val_days)["total_cost_vnd"]
        saving = (val_base - val_cost) / val_base * 100
        gap = (val_cost - val_oracle) / val_oracle * 100
        curve.append({"steps": steps, "val_cost_vnd": val_cost, "oracle_gap_pct": gap, "saving_vs_nobess_pct": saving})
        persist_progress()
        best_val = val_cost
        print(
            f"  best   {updates:>4} | step {steps:>7} | val {val_cost/1e6:8.1f}M | "
            f"saving {saving:5.1f}% | gap {gap:6.1f}% | final short-run checkpoint",
            flush=True,
        )
        agent.save(RESULTS_DIR / f"policy_{tag}.pt")

    persist_progress()

    test_month = MonthData(days=test_days, source="test")
    best_agent = PPOAgent(env.observation_dimensions, device=learner_device)
    best_agent.load(RESULTS_DIR / f"policy_{tag}.pt")
    result = run_drl_policy(test_month, cfg, best_agent, p_ref_kw=p_ref)
    test_cost = score_month(result["p_grid_days"], cfg, days=test_days)["total_cost_vnd"]
    no_bess_cost = score_month(run_no_bess(test_month, cfg)["p_grid_days"], cfg, days=test_days)["total_cost_vnd"]
    test_saving = (no_bess_cost - test_cost) / no_bess_cost * 100
    best_agent.meta = {**agent.meta, "test_saving_pct": round(test_saving, 2), "trained": date.today().isoformat()}
    best_agent.save(RESULTS_DIR / f"policy_{tag}.pt")
    report["status"] = "complete"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["test"] = {
        "policy_cost_vnd": test_cost,
        "no_bess_vnd": no_bess_cost,
        "saving_pct": test_saving,
    }
    write_report(report_path, report)
    print(
        f"[train-ds] TEST {test_days[0].date_iso}->{test_days[-1].date_iso}: "
        f"{test_cost/1e6:.1f}M vs no-BESS {no_bess_cost/1e6:.1f}M -> saving {test_saving:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
