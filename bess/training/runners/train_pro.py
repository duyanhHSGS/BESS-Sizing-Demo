"""Train PRO (PPO with Oracle Regularization) on CSV data.

Each iteration samples a random day from the training split, runs a full
day-length rollout while computing the oracle LP's implied action at each
decision step, then performs one PPO update with the auxiliary oracle-
imitation loss.  Validation uses the cached, month-wide Oracle LP traces
already wired into the GREPO training script.

Run: python train_pro.py --e-cap 750 --p-rated 350 --csv data.csv
                         --oracle-cache user_data/oracle_lp_cache/xxx.json
                         --training-config training_config.json
                         --control-dt-minutes 15
                         [--oracle-coef 1.0] [--oracle-decay 0.002]
                         [--iters 400] [--seed 0]
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from bess.core.common import RESULTS_DIR, score_month, score_operating_month
from bess.evaluation.benchmark import _rolling_30_minute_average
from bess.core.scenario_gen import MonthData, DayData
from bess.core.bess_env import BESSEnv, NORMAL_OBSERVATION_SCHEMA
from bess.agents.pro_agent import PROAgent, PROBuffer
from bess.evaluation.baselines import run_no_bess, run_drl_policy
from bess.evaluation.oracle.oracle_cache import load_cached_training_grids
from bess.evaluation.policy_diagnostics import monthly_policy_diagnostics
from bess.training.training_common import build_training_bess_config, load_training_days, score_cached_oracle
from bess.training.training_reports import write_curve, write_report
from bess.core.settings import PPO_GAMMA
from bess.forecasting.weather_forecast import build_forecast_bundle, fit_attach_forecasts

VAL_EVERY = 10
LOG_EVERY_ITERS = 4


def _runtime_snapshot(perf):
    return {
        key: float(value)
        for key, value in perf.items()
        if not key.startswith("_")
    }


def _build_oracle_lookup(oracle_cache_path: str | Path,
                         day_indexes: list[int]) -> dict[int, list[float]]:
    """Return {day_index: oracle_grid_trajectory} for all requested days."""
    grids = load_cached_training_grids(oracle_cache_path, day_indexes)
    return {idx: grid for idx, grid in zip(day_indexes, grids)}


def _oracle_action(day: DayData, oracle_grid: list[float],
                   dt: float, p_ref: float,
                   cfg) -> np.ndarray:
    """Compute oracle-implied normalized battery action for each native step.

    p_bess = eff_load - grid_oracle   (positive = discharge to load)
    a_oracle = clip(p_bess / P_rated, -1, 1)

    Returns array of shape (n_native_steps,).
    """
    load = np.asarray(day.load)
    pv = np.asarray(day.pv)
    eff = np.maximum(0.0, load - pv)
    grid_oracle = np.asarray(oracle_grid, dtype=np.float64)
    # The oracle grid should already be constrained to >= 0 by the LP;
    # p_bess > 0 means discharge, < 0 means charge.
    p_bess = eff - grid_oracle
    a = np.clip(p_bess / cfg.P_rated_nominal, -1.0, 1.0)
    return a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--e-cap", type=float, required=True)
    ap.add_argument("--p-rated", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--csv", type=str, required=True,
                    help="CSV dataset exported by the Sizing Demo launcher.")
    ap.add_argument("--oracle-coef", type=float, default=1.0,
                    help="Initial weight on oracle imitation loss (default 1.0).")
    ap.add_argument("--oracle-decay", type=float, default=0.0,
                    help="Linear decay per update subtracted from oracle_coef "
                         "(default 0 = constant weight). "
                         "e.g. 0.002 reaches zero after 500 updates.")
    ap.add_argument("--gamma", type=float, default=PPO_GAMMA)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"),
                    default="auto",
                    help="Learner device; rollout stays on CPU.")
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
    args = ap.parse_args()
    if not np.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        raise SystemExit("gamma must be finite and in (0, 1]")

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
        raise SystemExit(
            f"CSV has {len(csv_days)} days; need more than {split_days}"
        )

    peak = max(float(d.load.max()) for d in csv_days)
    p_ref = math.ceil(peak / 500.0) * 500.0
    peaks = [
        max(_rolling_30_minute_average(
            np.maximum(0, d.load - d.pv), cfg.dt
        ), default=0.0)
        for d in csv_days[:-split_days]
    ]
    d_run0 = 0.5 * float(np.mean(peaks))
    train_days = csv_days[:-split_days]

    # --- forecast model (optional) -----------------------------------------
    forecast_model = None
    if args.obs_variant == "fc":
        if not args.weather_data or not args.forecast_artifact:
            raise SystemExit(
                "forecast mode requires --weather-data and --forecast-artifact"
            )
        forecast_model = fit_attach_forecasts(
            csv_days, Path(args.weather_data), len(train_days),
            Path(args.forecast_artifact), p_ref,
        )

    # --- build oracle lookup -----------------------------------------------
    all_train_indexes = [int(d.day_index) for d in train_days]
    oracle_lookup = _build_oracle_lookup(args.oracle_cache, all_train_indexes)

    val_month = MonthData(source="csv_val")
    val_month.days = csv_days[-split_days:-args.test_days]
    tag = args.tag or f"pro_{cfg.E_cap:.0f}kwh_{cfg.P_rated_nominal:.0f}kw"

    # --- environment & agent -----------------------------------------------
    make_env = lambda: BESSEnv(  # noqa: E731
        cfg,
        reference_power_kw=p_ref,
        discount_factor=args.gamma,
        control_interval_minutes=args.control_dt_minutes,
        forecast_enabled=args.obs_variant == "fc",
        record_trajectory=False,
        degradation_cost_vnd_per_kwh=cfg.battery_wear_cost_vnd_per_kwh,
    )
    control_probe = make_env()
    native_steps = len(train_days[0].load)
    decisions_per_day = native_steps // control_probe.native_samples_per_action
    agent = PROAgent(
        control_probe.observation_dimensions,
        oracle_coef=args.oracle_coef,
        oracle_coef_decay=args.oracle_decay,
        seed=args.seed,
        gamma=args.gamma,
        device=args.device,
    )

    agent.meta = {
        "p_ref_kw": p_ref,
        "algo": "pro",
        "e_cap_kwh": cfg.E_cap,
        "p_rated_kw": cfg.P_rated_nominal,
        "billing_mode": billing_mode,
        "gamma": args.gamma,
        "obs_variant": args.obs_variant,
        "obs_dim": control_probe.observation_dimensions,
        "battery_wear_cost": cfg.battery_wear_cost_vnd_per_kwh,
        "observation_schema": NORMAL_OBSERVATION_SCHEMA,
        "native_dt_minutes": cfg.dt * 60.0,
        "control_dt_minutes": args.control_dt_minutes,
        "oracle_coef": args.oracle_coef,
        "oracle_coef_decay": args.oracle_decay,
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
    agent.meta["native_steps_per_action"] = control_probe.native_samples_per_action

    # --- validation baselines ----------------------------------------------
    val_base = score_month(
        run_no_bess(val_month, cfg)["p_grid_days"], cfg, days=val_month.days
    )["total_cost_vnd"]
    val_oracle_result = score_cached_oracle(
        args.oracle_cache,
        [day.day_index for day in val_month.days],
        cfg,
        val_month.days,
    )
    val_oracle = val_oracle_result["total_operating_cost_vnd"]

    curve_path = RESULTS_DIR / f"training_curve_{tag}.csv"
    report_path = RESULTS_DIR / f"training_report_{tag}.json"
    report = {
        "version": 1,
        "status": "running",
        "algorithm": "pro",
        "observation_schema": NORMAL_OBSERVATION_SCHEMA,
        "obs_dim": control_probe.observation_dimensions,
        "battery_wear_cost": cfg.battery_wear_cost_vnd_per_kwh,
        "economics": {"battery_wear_cost": cfg.battery_wear_cost_vnd_per_kwh},
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
            "gamma": args.gamma,
            "native_dt_minutes": cfg.dt * 60.0,
            "control_dt_minutes": control_probe.control_interval_minutes,
            "native_steps_per_action": control_probe.native_samples_per_action,
            "device_requested": args.device,
            "device": str(agent.device),
            "oracle_coef": args.oracle_coef,
            "oracle_coef_decay": args.oracle_decay,
        },
        "billing_mode": billing_mode,
        "p_ref_kw": p_ref,
        "d_run_init_kw": d_run0,
        "validation": {"no_bess_vnd": val_base, "oracle_vnd": val_oracle},
    }

    perf = {
        "rollout_seconds": 0.0,
        "update_seconds": 0.0,
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
    perf["checkpoint_report_io_seconds"] += time.perf_counter() - io_started
    print(
        f"[pro] config {tag} | gamma={args.gamma:g} | "
        f"oracle_coef={args.oracle_coef:g} decay={args.oracle_decay:g} | "
        f"learner={agent.device} (requested {args.device}) | "
        f"native dt {cfg.dt * 60:g}m | control dt {control_probe.control_interval_minutes:g}m | "
        f"val no-BESS {val_base/1e6:.1f}M, oracle {val_oracle/1e6:.1f}M VND",
        flush=True,
    )

    curve = []
    best_val = float("inf")
    t0 = time.time()
    steps = 0
    rng = np.random.default_rng(args.seed)
    buf = PROBuffer(decisions_per_day, control_probe.observation_dimensions)

    for it in range(args.iters):
        # Sample a random training day
        day = train_days[rng.integers(len(train_days))]
        day_index = int(day.day_index)

        # Oracle implied action for every native step of this day
        oracle_grid = oracle_lookup.get(day_index)
        if oracle_grid is not None:
            oracle_actions_native = _oracle_action(
                day, oracle_grid, cfg.dt, p_ref, cfg
            )
        else:
            # Day missing from oracle cache — treat as having no oracle signal.
            oracle_actions_native = np.zeros(native_steps, dtype=np.float32)

        # Down-sample oracle actions to match the control interval.
        # The oracle action for a control step is the mean over its native sub-steps
        # (all sub-steps inside one control interval share the same decision).
        interval = control_probe.native_samples_per_action
        oracle_actions = np.array([
            float(np.clip(oracle_actions_native[i * interval:(i + 1) * interval].mean(),
                          -1.0, 1.0))
            for i in range(decisions_per_day)
        ], dtype=np.float32)

        # --- rollout -------------------------------------------------------
        rollout_started = time.perf_counter()
        env = make_env()
        episode = MonthData(days=[day], source="pro_day")
        soc_init = float(rng.uniform(cfg.SOC_min + cfg.SOC_safety,
                                     cfg.SOC_max))
        if d_run0 is not None:
            d_run_init = float(d_run0 * rng.uniform(0.8, 1.5))
        else:
            d_run_init = float(rng.uniform(0.5, 0.9) * p_ref)
        env.initial_running_peak_kw = d_run_init
        obs = env.reset(episode, soc_init=soc_init)
        done = False
        decision_idx = 0
        buf.clear()
        while not done:
            a, logp, v = agent.act(obs, deterministic=False)
            a_oracle = float(oracle_actions[decision_idx])
            obs, reward, done, info = env.step(a)
            native_rows = int(info.get("native_rows", 1))
            # Only store one entry per decision; step() handles sub-stepping
            buf.add(obs if not done else np.zeros_like(obs),
                    np.array([a], dtype=np.float32),
                    logp, reward, v,
                    1.0 if done else 0.0,
                    a_oracle)
            decision_idx += 1
            steps += 1
            perf["_native_rows"] += native_rows
            perf["_decisions"] += 1
            perf["_samples"] += 1
        last_val = 0.0  # episode terminated at buf.ptr
        perf["rollout_seconds"] += time.perf_counter() - rollout_started

        # --- update --------------------------------------------------------
        update_started = time.perf_counter()
        losses = agent.update(buf, last_val)
        perf["update_seconds"] += time.perf_counter() - update_started

        # --- validation ----------------------------------------------------
        if (it + 1) % VAL_EVERY != 0 and it + 1 != args.iters:
            if (it + 1) == 1 or (it + 1) % LOG_EVERY_ITERS == 0:
                print(
                    f"  iter {it+1:>3}/{args.iters} | steps {steps:>5} | "
                    f"pi {losses['pi_loss']:+.3f} | vf {losses['vf_loss']:.4f} | "
                    f"oracle_loss {losses['oracle_loss']:.4f} "
                    f"(coef={losses['oracle_coef']:.4f}) | "
                    f"ent {losses['ent']:.4f}",
                    flush=True,
                )
            continue

        validation_started = time.perf_counter()
        res = run_drl_policy(val_month, cfg, agent, p_ref_kw=p_ref)
        perf["validation_seconds"] += time.perf_counter() - validation_started

        scoring_started = time.perf_counter()
        val_cost = score_operating_month(
            res["p_grid_days"], res["p_bess_days"], cfg, days=val_month.days
        )["total_operating_cost_vnd"]
        gap = (val_cost - val_oracle) / val_oracle * 100
        sav = (val_base - val_cost) / val_base * 100
        perf["scoring_seconds"] += time.perf_counter() - scoring_started

        curve.append({
            "steps": steps, "val_cost_vnd": val_cost,
            "oracle_gap_pct": gap, "saving_vs_nobess_pct": sav,
        })
        report["runtime"] = _runtime_snapshot(perf)

        io_started = time.perf_counter()
        write_curve(curve_path, curve)
        report["latest"] = curve[-1]
        report["best"] = min(curve, key=lambda point: point["val_cost_vnd"])
        write_report(report_path, report)
        perf["checkpoint_report_io_seconds"] += time.perf_counter() - io_started

        print(
            f"  iter {it+1:>3}/{args.iters} | steps {steps:>5} | "
            f"val {val_cost/1e6:8.1f}M | saving {sav:5.1f}% | "
            f"gap {gap:6.1f}% | pi {losses['pi_loss']:+.3f} | "
            f"oracle_loss {losses['oracle_loss']:.4f} "
            f"(coef={losses['oracle_coef']:.4f})",
            flush=True,
        )

        if val_cost < best_val:
            best_val = val_cost
            agent.meta = {
                **agent.meta,
                "e_cap_kwh": cfg.E_cap,
                "p_rated_kw": cfg.P_rated_nominal,
            }
            if d_run0 is not None:
                agent.meta["d_run_init_kw"] = d_run0
            io_started = time.perf_counter()
            agent.save(RESULTS_DIR / f"policy_{tag}.pt")
            perf["checkpoint_report_io_seconds"] += (
                time.perf_counter() - io_started
            )
            print(
                f"  best {it+1:>4} | steps {steps:>5} | "
                f"val {val_cost/1e6:8.1f}M | saving {sav:5.1f}% | "
                f"gap {gap:6.1f}% | checkpoint updated",
                flush=True,
            )

    # --- final test --------------------------------------------------------
    test_days = csv_days[-args.test_days:]
    test_month = MonthData(days=test_days, source="csv_test")
    best_agent = PROAgent(
        control_probe.observation_dimensions,
        oracle_coef=0.0,  # no oracle needed for test inference
        seed=args.seed,
        gamma=args.gamma,
        device=args.device,
    )
    best_agent.load(RESULTS_DIR / f"policy_{tag}.pt")
    val_result = run_drl_policy(val_month, cfg, best_agent, p_ref_kw=p_ref)
    test_result = run_drl_policy(test_month, cfg, best_agent, p_ref_kw=p_ref)
    test_policy = score_operating_month(
        test_result["p_grid_days"], test_result["p_bess_days"], cfg, days=test_days
    )
    test_no_bess = score_month(
        run_no_bess(test_month, cfg)["p_grid_days"], cfg, days=test_days
    )
    test_oracle = score_cached_oracle(
        args.oracle_cache, [day.day_index for day in test_days], cfg, test_days
    )
    test_saving = (
        (test_no_bess["total_cost_vnd"] - test_policy["total_operating_cost_vnd"])
        / test_no_bess["total_cost_vnd"] * 100
    )
    test_oracle_gap = (
        (test_policy["total_operating_cost_vnd"] - test_oracle["total_operating_cost_vnd"])
        / test_oracle["total_operating_cost_vnd"] * 100
    )
    best_agent.meta = {
        **best_agent.meta,
        "test_saving_pct": round(test_saving, 2),
        "test_oracle_gap_pct": round(test_oracle_gap, 2),
        "test_peak_kw": round(test_policy["pmax_month_kw"], 2),
        "trained": date.today().isoformat(),
    }
    best_agent.save(RESULTS_DIR / f"policy_{tag}.pt")

    report["status"] = "complete"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["runtime"] = _runtime_snapshot(perf)
    report["diagnostics"] = {
        "validation_months": monthly_policy_diagnostics(
            val_month, val_result, cfg, oracle_days=val_oracle_result["days"],
            initial_running_peak_kw=d_run0 or 0.0,
        ),
        "test_months": monthly_policy_diagnostics(
            test_month, test_result, cfg, oracle_days=test_oracle["days"],
            initial_running_peak_kw=d_run0 or 0.0,
        ),
    }
    report["test"] = {
        "policy_cost_vnd": test_policy["total_operating_cost_vnd"],
        "no_bess_vnd": test_no_bess["total_cost_vnd"],
        "oracle_vnd": test_oracle["total_operating_cost_vnd"],
        "saving_pct": test_saving,
        "oracle_gap_pct": test_oracle_gap,
        "energy_cost_vnd": test_policy["energy_cost_vnd"],
        "demand_cost_vnd": test_policy["demand_cost_vnd"],
        "wear_cost_vnd": test_policy["wear_cost_vnd"],
        "peak_kw": test_policy["pmax_month_kw"],
    }
    io_started = time.perf_counter()
    write_curve(curve_path, curve)
    write_report(report_path, report)
    perf["checkpoint_report_io_seconds"] += time.perf_counter() - io_started
    print(
        f"[pro] TEST {test_days[0].date_iso}->{test_days[-1].date_iso}: "
        f"{test_policy['total_operating_cost_vnd']/1e6:.1f}M vs no-BESS "
        f"{test_no_bess['total_cost_vnd']/1e6:.1f}M -> saving {test_saving:.2f}% | "
        f"oracle {test_oracle['total_operating_cost_vnd']/1e6:.1f}M -> gap {test_oracle_gap:.2f}% | "
        f"peak {test_policy['pmax_month_kw']:.1f} kW",
        flush=True,
    )
    print(
        f"[pro] done in {time.time() - t0:.0f}s. Best val "
        f"{best_val/1e6:.1f}M VND -> policy_{tag}.pt",
        flush=True,
    )


if __name__ == "__main__":
    main()
