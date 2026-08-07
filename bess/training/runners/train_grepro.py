"""Train the repo-local GrePRO method on progressive contiguous horizons.

GrePRO is separate from published GREPO. It moves from 3-day to 7-day to
30-day chronological episodes, then learns from both absolute returns and
same-time group-relative returns. Validation remains month-wide.
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from bess.core.common import RESULTS_DIR, check_hard_constraints, score_month
from bess.evaluation.benchmark import _rolling_30_minute_average
from bess.core.scenario_gen import MonthData
from bess.agents.grepo_agent import resolve_grepo_device
from bess.agents.grepro_agent import GREPROAgent
from bess.evaluation.baselines import run_no_bess, run_drl_policy, run_sadrbc
from bess.evaluation.oracle.oracle_cache import load_cached_training_grids
from bess.training.training_common import build_training_bess_config, load_training_days
from bess.training.training_reports import write_curve, write_report
from bess.core.settings import GREPRO_GAMMA
from bess.forecasting.weather_forecast import build_forecast_bundle, fit_attach_forecasts
from bess.forecasting.sadrbc_forecast import (
    DEFAULT_FORECAST_SEED,
    SADRBCForecastSpec,
    SADRBCResidualEnv,
    rollout_activity,
)

VAL_EVERY = 5
LOG_EVERY_ITERS = 4


def _runtime_snapshot(perf):
    rollout_seconds = max(perf["group_rollout_seconds"], 1e-12)
    return {
        **{
            key: float(value)
            for key, value in perf.items()
            if not key.startswith("_")
        },
        "native_rows_per_second": perf["_native_rows"] / rollout_seconds,
        "decisions_per_second": perf["_decisions"] / rollout_seconds,
        "samples_per_second": perf["_samples"] / rollout_seconds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--e-cap", type=float, required=True)
    ap.add_argument("--p-rated", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--csv", type=str, required=True,
                    help="CSV dataset exported by the Sizing Demo launcher.")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="trng s hybrid baseline Eq.28")
    ap.add_argument("--std", type=float, default=0.20,
                    help="std c nh lambda ca policy Gaussian")
    ap.add_argument("--gamma", type=float, default=GREPRO_GAMMA)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"),
                    default="auto",
                    help="GrePRO learner device; collection stays on CPU")
    ap.add_argument("--control-dt-minutes", type=float, required=True)
    ap.add_argument("--training-config", type=str, required=True,
                    help="Sizing Demo canonical training_config.json path.")
    ap.add_argument("--oracle-cache", type=str, required=True,
                    help="Exact cached month-wide Oracle LP result.")
    ap.add_argument("--val-days", type=int, default=30,
                    help="Number of CSV days reserved for validation.")
    ap.add_argument("--test-days", type=int, default=30,
                    help="Number of CSV days reserved for test holdout.")
    ap.add_argument("--obs-variant", choices=("base", "fc"), default="base")
    ap.add_argument("--weather-data", default="")
    ap.add_argument("--forecast-artifact", default="")
    ap.add_argument("--forecast-seed", type=int, default=DEFAULT_FORECAST_SEED)
    ap.add_argument("--forecast-load-sigma", type=float, default=0.05)
    ap.add_argument("--forecast-pv-sigma", type=float, default=0.15)
    ap.add_argument(
        "--residual-limit", type=float, default=0.05,
        help="Constant fraction of rated BESS power available to GrePRO corrections.",
    )
    args = ap.parse_args()
    if not np.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        raise SystemExit("gamma must be finite and in (0, 1]")
    if not np.isfinite(args.residual_limit) or not 0.0 < args.residual_limit <= 1.0:
        raise SystemExit("residual-limit must be finite and in (0, 1]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_days = load_training_days(args.csv, weather="tb")
    csv_dt = 24.0 / len(csv_days[0].load)
    cfg, billing_mode = build_training_bess_config(
        args.e_cap,
        args.p_rated,
        csv_dt,
        args.training_config,
    )
    if args.val_days < 1 or args.test_days < 1:
        raise SystemExit("Validation days and test days must both be at least 1")
    split_days = args.val_days + args.test_days
    if len(csv_days) <= split_days:
        raise SystemExit(f"CSV has {len(csv_days)} days; need more than {split_days}")
    peak = max(float(d.load.max()) for d in csv_days)
    p_ref = math.ceil(peak / 500.0) * 500.0
    peaks = [
        max(_rolling_30_minute_average(np.maximum(0, d.load - d.pv), cfg.dt), default=0.0)
        for d in csv_days[:-split_days]
    ]
    d_run0 = 0.5 * float(np.mean(peaks))
    train_days = csv_days[:-split_days]
    if len(train_days) < 30:
        raise SystemExit(
            f"GrePRO needs at least 30 chronological training days; found {len(train_days)}"
        )
    forecast_model = None
    if args.obs_variant == "fc":
        if not args.weather_data or not args.forecast_artifact:
            raise SystemExit("forecast mode requires --weather-data and --forecast-artifact")
        forecast_model = fit_attach_forecasts(
            csv_days, Path(args.weather_data), len(train_days),
            Path(args.forecast_artifact), p_ref,
        )
    val_month = MonthData(source="csv_val")
    val_month.days = csv_days[-split_days:-args.test_days]
    tag = args.tag or f"grepro_{cfg.E_cap:.0f}kwh_{cfg.P_rated_nominal:.0f}kw"

    forecast_spec = SADRBCForecastSpec(
        seed=args.forecast_seed,
        load_sigma=args.forecast_load_sigma,
        pv_sigma=args.forecast_pv_sigma,
    )
    make_env = lambda: SADRBCResidualEnv(   # noqa: E731
        cfg,
        p_ref_kw=p_ref,
        gamma=args.gamma,
        control_dt_minutes=args.control_dt_minutes,
        use_forecast=args.obs_variant == "fc",
        record_trajectory=False,
        residual_limit=args.residual_limit,
        forecast_spec=forecast_spec,
    )
    control_probe = make_env()
    learner_device = resolve_grepo_device(args.device)
    agent = GREPROAgent(
        control_probe.obs_dim, n_group=args.group, seed=args.seed,
        gamma=args.gamma, beta=args.beta, std=args.std,
        device=learner_device,
    )
    # meta trc validation u tin  env val dng ng floor/p_ref
    agent.meta = {
        "p_ref_kw": p_ref,
        "algo": "grepro",
        "method": "sadrbc-residual-group-relative-progressive-horizon-v3",
        "controller": "sadrbc_residual",
        "e_cap_kwh": cfg.E_cap,
        "p_rated_kw": cfg.P_rated_nominal,
        "billing_mode": billing_mode,
        "group": args.group,
        "beta": args.beta,
        "std": args.std,
        "gamma": args.gamma,
        "obs_variant": args.obs_variant,
        "obs_dim": control_probe.obs_dim,
        "native_dt_minutes": cfg.dt * 60.0,
        "control_dt_minutes": args.control_dt_minutes,
        "residual_limit": args.residual_limit,
        "sadrbc_forecast": forecast_spec.public(
            "real_weather_causal_plus_declared_noisy_tail"
            if forecast_model else "declared_noisy_ar1"
        ),
    }
    if forecast_model:
        agent.meta["forecast_model"] = forecast_model["model"]
        agent.meta["forecast_artifact"] = forecast_model["artifact"]
        agent.meta["forecast_model_artifact"] = forecast_model["model_artifact"]
        agent.meta["weather_data"] = str(Path(args.weather_data))
        agent.meta["forecast_embedded"] = True
        agent.forecast_bundle = build_forecast_bundle(csv_days)
    if d_run0 is not None:
        agent.meta["d_run_init_kw"] = d_run0
    agent.meta["native_steps_per_action"] = control_probe.native_steps_per_action
    val_base = score_month(
        run_no_bess(val_month, cfg)["p_grid_days"], cfg, days=val_month.days
    )["total_cost_vnd"]
    val_sadrbc_result = run_sadrbc(
        val_month, cfg, forecast_spec=forecast_spec, p_ref_kw=p_ref
    )
    val_sadrbc = score_month(
        val_sadrbc_result["p_grid_days"], cfg, days=val_month.days
    )["total_cost_vnd"]
    val_sadrbc_activity = rollout_activity(val_sadrbc_result, cfg.dt)
    oracle_grids = load_cached_training_grids(
        args.oracle_cache, [day.day_index for day in val_month.days]
    )
    val_oracle = score_month(oracle_grids, cfg, days=val_month.days)["total_cost_vnd"]
    curve_path = RESULTS_DIR / f"training_curve_{tag}.csv"
    report_path = RESULTS_DIR / f"training_report_{tag}.json"
    report = {
        "version": 1,
        "status": "running",
        "algorithm": "grepro",
        "tag": tag,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": str(args.csv),
            "total_days": len(csv_days),
            "train_days": len(train_days),
            "validation_days": len(val_month.days),
            "test_days": args.test_days,
            "validation_range": [
                val_month.days[0].date_iso,
                val_month.days[-1].date_iso,
            ],
            "test_range": [
                csv_days[-args.test_days].date_iso,
                csv_days[-1].date_iso,
            ],
        },
        "battery": {"e_cap_kwh": cfg.E_cap, "p_rated_kw": cfg.P_rated_nominal},
        "training": {
            "requested_iterations": args.iters,
            "seed": args.seed,
            "group": args.group,
            "beta": args.beta,
            "std": args.std,
            "gamma": args.gamma,
            "native_dt_minutes": cfg.dt * 60.0,
            "control_dt_minutes": control_probe.control_dt_minutes,
            "native_steps_per_action": control_probe.native_steps_per_action,
            "device_requested": args.device,
            "device": learner_device,
            "horizon_curriculum_days": [3, 7, 30],
            "residual_limit": args.residual_limit,
            "sadrbc_forecast": forecast_spec.public(
                "real_weather_causal_plus_declared_noisy_tail"
                if forecast_model else "declared_noisy_ar1"
            ),
        },
        "billing_mode": billing_mode,
        "p_ref_kw": p_ref,
        "d_run_init_kw": d_run0,
        "validation": {
            "no_bess_vnd": val_base,
            "sadrbc_vnd": val_sadrbc,
            "sadrbc_activity": val_sadrbc_activity,
            "oracle_vnd": val_oracle,
        },
    }
    perf = {
        "group_rollout_seconds": 0.0,
        "return_preparation_seconds": 0.0,
        "batch_transfer_seconds": 0.0,
        "actor_critic_update_seconds": 0.0,
        "cpu_actor_sync_seconds": 0.0,
        "validation_seconds": 0.0,
        "scoring_seconds": 0.0,
        "checkpoint_report_io_seconds": 0.0,
        "_native_rows": 0,
        "_decisions": 0,
        "_samples": 0,
    }
    io_started = time.perf_counter()
    write_curve(curve_path, [])
    write_report(report_path, report)
    perf["checkpoint_report_io_seconds"] += (
        time.perf_counter() - io_started
    )
    print(f"[grepro] config {tag} | group={args.group} | gamma={args.gamma:g} | "
          f"learner={learner_device} (requested {args.device}) | "
          f"native dt {cfg.dt * 60:g}m | control dt {control_probe.control_dt_minutes:g}m | "
          f"fixed residual {args.residual_limit * 100:g}% | "
          f"SADRBC forecast seed={forecast_spec.seed}, load sigma={forecast_spec.load_sigma:g}, "
          f"PV sigma={forecast_spec.pv_sigma:g} | val no-BESS {val_base/1e6:.1f}M, "
          f"causal SADRBC {val_sadrbc/1e6:.1f}M, oracle {val_oracle/1e6:.1f}M VND", flush=True)

    curve = []
    best_val = float("inf")
    t0 = time.time()
    steps = 0
    rng = np.random.default_rng(args.seed)
    for it in range(args.iters):
        progress = (it + 1) / max(1, args.iters)
        horizon_days = 3 if progress <= 0.20 else (7 if progress <= 0.50 else 30)
        residual_limit = args.residual_limit
        max_start = len(train_days) - horizon_days
        start = int(rng.integers(max_start + 1))
        episode_days = train_days[start:start + horizon_days]
        episode = MonthData(
            days=episode_days,
            source=f"grepro_{horizon_days}d",
        )
        soc_init = float(rng.uniform(cfg.SOC_min + cfg.SOC_safety,
                                     cfg.SOC_max))
        if d_run0 is not None:      # site tht: floor data-driven  jitter
            d_run_init = float(d_run0 * rng.uniform(0.8, 1.5))
        else:
            d_run_init = float(rng.uniform(0.5, 0.9) * p_ref)
        batch = agent.collect_group(
            make_env,
            episode,
            soc_init=soc_init,
            d_run_init=d_run_init,
            residual_limit=residual_limit,
        )
        collect_stats = agent.last_collect_stats
        perf["group_rollout_seconds"] += collect_stats[
            "group_rollout_seconds"
        ]
        perf["_native_rows"] += collect_stats["native_rows"]
        perf["_decisions"] += collect_stats["decisions"]
        perf["_samples"] += collect_stats["samples"]
        steps += batch[3].size
        losses = agent.update(*batch)
        for key, value in agent.last_update_stats.items():
            perf[key] += value
        if (it + 1) % VAL_EVERY != 0 and it + 1 != args.iters:
            continue
        validation_started = time.perf_counter()
        agent.meta["residual_limit"] = residual_limit
        res = run_drl_policy(val_month, cfg, agent, p_ref_kw=p_ref)
        perf["validation_seconds"] += (
            time.perf_counter() - validation_started
        )
        scoring_started = time.perf_counter()
        val_cost = score_month(res["p_grid_days"], cfg, days=val_month.days)["total_cost_vnd"]
        gap = (val_cost - val_oracle) / val_oracle * 100
        sav = (val_base - val_cost) / val_base * 100
        sav_sadrbc = (val_sadrbc - val_cost) / val_sadrbc * 100
        activity = rollout_activity(res, cfg.dt)
        constraints = check_hard_constraints(
            res["p_grid_days"], res["soc_days"], cfg
        )
        safe = sum(constraints.values()) == 0
        min_throughput = 0.05 * val_sadrbc_activity["throughput_kwh"]
        active = (
            val_sadrbc_activity["throughput_kwh"] <= 1e-6
            or activity["throughput_kwh"] >= min_throughput
        )
        perf["scoring_seconds"] += time.perf_counter() - scoring_started
        curve.append({
            "steps": steps,
            "val_cost_vnd": val_cost,
            "oracle_gap_pct": gap,
            "saving_vs_nobess_pct": sav,
            "saving_vs_sadrbc_pct": sav_sadrbc,
            **activity,
            "blocked_action_pct": res.get("blocked_action_pct", 0.0),
            "residual_limit": residual_limit,
            "active_gate": int(active),
            "zero_export_violation_days": constraints["zero_export_violation_days"],
            "soc_violation_days": constraints["soc_violation_days"],
        })
        report["runtime"] = _runtime_snapshot(perf)
        io_started = time.perf_counter()
        write_curve(curve_path, curve)
        report["latest"] = curve[-1]
        eligible_curve = [
            point for point in curve
            if point["active_gate"]
            and point["zero_export_violation_days"] == 0
            and point["soc_violation_days"] == 0
        ]
        report["best"] = (
            min(eligible_curve, key=lambda point: point["val_cost_vnd"])
            if eligible_curve else {}
        )
        write_report(report_path, report)
        perf["checkpoint_report_io_seconds"] += (
            time.perf_counter() - io_started
        )
        if (it + 1) == 1 or (it + 1) % LOG_EVERY_ITERS == 0:
            print(f"  iter {it+1:>3}/{args.iters} | horizon {horizon_days:>2}d | steps {steps:>7} | "
                  f"val {val_cost/1e6:8.1f}M | saving {sav:5.1f}% | "
                  f"gap {gap:6.1f}% | pi {losses['pi_loss']:+.3f} | "
                  f"{steps/(time.time()-t0):,.0f} sps", flush=True)
        if val_cost < best_val and safe and active:
            best_val = val_cost
            agent.meta = {
                **agent.meta,
                "beta": args.beta,
                "std": args.std,
                "e_cap_kwh": cfg.E_cap,
                "p_rated_kw": cfg.P_rated_nominal,
                "controller": "sadrbc_residual",
                "residual_limit": residual_limit,
                "sadrbc_forecast": forecast_spec.public(
                    "real_weather_causal_plus_declared_noisy_tail"
                    if forecast_model else "declared_noisy_ar1"
                ),
            }
            if d_run0 is not None:
                agent.meta["d_run_init_kw"] = d_run0
            io_started = time.perf_counter()
            agent.save(RESULTS_DIR / f"policy_{tag}.pt")
            perf["checkpoint_report_io_seconds"] += (
                time.perf_counter() - io_started
            )
            print(f"  best {it+1:>4} | steps {steps:>7} | "
                  f"val {val_cost/1e6:8.1f}M | saving {sav:5.1f}% | "
                  f"vs SADRBC {sav_sadrbc:+5.1f}% | gap {gap:6.1f}% | "
                  f"throughput {activity['throughput_kwh']:.0f} kWh | checkpoint updated", flush=True)
        elif not safe or not active:
            reason = "safety violations" if not safe else "idle-policy gate"
            print(f"  iter {it+1:>3} | checkpoint REJECTED: {reason}", flush=True)
        elif (it + 1) % LOG_EVERY_ITERS == 0:
            print(f"  iter {it+1:>3}/{args.iters} | steps {steps:>7} | "
                  f"val {val_cost/1e6:8.1f}M | saving {sav:5.1f}% | "
                  f"gap {gap:6.1f}% | no new best", flush=True)

    test_days = csv_days[-args.test_days:]
    test_month = MonthData(days=test_days, source="csv_test")
    best_agent = GREPROAgent(
        control_probe.obs_dim,
        n_group=args.group,
        gamma=args.gamma,
        std=args.std,
        beta=args.beta,
        seed=args.seed,
        device=learner_device,
    )
    best_agent.load(RESULTS_DIR / f"policy_{tag}.pt")
    test_result = run_drl_policy(test_month, cfg, best_agent, p_ref_kw=p_ref)
    test_policy = score_month(
        test_result["p_grid_days"], cfg, days=test_days
    )
    test_no_bess = score_month(
        run_no_bess(test_month, cfg)["p_grid_days"], cfg, days=test_days
    )
    test_sadrbc_result = run_sadrbc(
        test_month, cfg, forecast_spec=forecast_spec, p_ref_kw=p_ref
    )
    test_sadrbc = score_month(
        test_sadrbc_result["p_grid_days"], cfg, days=test_days
    )
    test_oracle_grids = load_cached_training_grids(
        args.oracle_cache, [day.day_index for day in test_days]
    )
    test_oracle = score_month(test_oracle_grids, cfg, days=test_days)
    test_saving = (
        (test_no_bess["total_cost_vnd"] - test_policy["total_cost_vnd"])
        / test_no_bess["total_cost_vnd"] * 100
    )
    test_oracle_gap = (
        (test_policy["total_cost_vnd"] - test_oracle["total_cost_vnd"])
        / test_oracle["total_cost_vnd"] * 100
    )
    test_sadrbc_saving = (
        (test_sadrbc["total_cost_vnd"] - test_policy["total_cost_vnd"])
        / test_sadrbc["total_cost_vnd"] * 100
    )
    test_activity = rollout_activity(test_result, cfg.dt)
    test_sadrbc_activity = rollout_activity(test_sadrbc_result, cfg.dt)
    best_agent.meta = {
        **best_agent.meta,
        "test_saving_pct": round(test_saving, 2),
        "test_oracle_gap_pct": round(test_oracle_gap, 2),
        "test_peak_kw": round(test_policy["pmax_month_kw"], 2),
        "test_saving_vs_sadrbc_pct": round(test_sadrbc_saving, 2),
        "test_throughput_kwh": round(test_activity["throughput_kwh"], 2),
        "trained": date.today().isoformat(),
    }
    best_agent.save(RESULTS_DIR / f"policy_{tag}.pt")

    report["status"] = "complete"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["runtime"] = _runtime_snapshot(perf)
    report["test"] = {
        "policy_cost_vnd": test_policy["total_cost_vnd"],
        "no_bess_vnd": test_no_bess["total_cost_vnd"],
        "sadrbc_vnd": test_sadrbc["total_cost_vnd"],
        "oracle_vnd": test_oracle["total_cost_vnd"],
        "saving_pct": test_saving,
        "saving_vs_sadrbc_pct": test_sadrbc_saving,
        "oracle_gap_pct": test_oracle_gap,
        "energy_cost_vnd": test_policy["energy_cost_vnd"],
        "demand_cost_vnd": test_policy["demand_cost_vnd"],
        "peak_kw": test_policy["pmax_month_kw"],
        "activity": test_activity,
        "sadrbc_activity": test_sadrbc_activity,
        "blocked_action_pct": test_result.get("blocked_action_pct", 0.0),
    }
    io_started = time.perf_counter()
    write_curve(curve_path, curve)
    write_report(report_path, report)
    perf["checkpoint_report_io_seconds"] += time.perf_counter() - io_started
    print(
        f"[grepro] TEST {test_days[0].date_iso}->{test_days[-1].date_iso}: "
        f"{test_policy['total_cost_vnd']/1e6:.1f}M vs no-BESS "
        f"{test_no_bess['total_cost_vnd']/1e6:.1f}M -> saving {test_saving:.2f}% | "
        f"vs SADRBC {test_sadrbc_saving:+.2f}% | "
        f"oracle {test_oracle['total_cost_vnd']/1e6:.1f}M -> gap {test_oracle_gap:.2f}% | "
        f"peak {test_policy['pmax_month_kw']:.1f} kW | "
        f"throughput {test_activity['throughput_kwh']:.0f} kWh",
        flush=True,
    )
    print(f"[grepro] done in {time.time()-t0:.0f}s. Best val "
          f"{best_val/1e6:.1f}M VND -> policy_{tag}.pt", flush=True)


if __name__ == "__main__":
    main()
