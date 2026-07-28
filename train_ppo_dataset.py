from __future__ import annotations

import argparse
import csv as _csv
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from baselines import run_drl_policy, run_no_bess
from bess_env import BESSEnv
from benchmark import _rolling_30_minute_average
from common import RESULTS_DIR, load_system_config, make_bess_config, score_month
from ppo_agent import PPOAgent, RolloutBuffer
from scenario_gen import DayData, MonthData
from oracle_cache import load_cached_training_grids
from training_reports import write_curve, write_report
from settings import PPO_GAMMA, PPO_LAMBDA


ROLLOUT_DAYS = 32
LOG_EVERY_UPDATES = 4


def load_csv_days(path: Path) -> list[DayData]:
    by_day: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            day = by_day.setdefault(
                row["date_iso"],
                {
                    "load": [],
                    "pv": [],
                    "day_type": row["day_type"],
                    "day_index": int(row["day_index"]),
                },
            )
            step = int(row["step"])
            while len(day["load"]) <= step:
                day["load"].append(0.0)
                day["pv"].append(0.0)
            day["load"][step] = float(row["P_load_kW"])
            day["pv"][step] = float(row["P_pv_kW"])
    days = []
    for iso in sorted(by_day):
        data = by_day[iso]
        days.append(
            DayData(
                load=np.asarray(data["load"], dtype=np.float64),
                pv=np.asarray(data["pv"], dtype=np.float64),
                day_type=data["day_type"],
                weather="csv",
                day_index=data["day_index"],
                date_iso=iso,
            )
        )
    return days


def month_blocks(days: list[DayData]) -> list[MonthData]:
    blocks: dict[str, MonthData] = {}
    for day in days:
        key = str(day.date_iso)[:7]
        blocks.setdefault(key, MonthData(source=f"csv:{key}")).days.append(day)
    months = [month for _, month in sorted(blocks.items()) if len(month.days) >= 15]
    if months or not days:
        return months
    return [MonthData(days=days, source="csv:train_short")]


def augment_month(
    month: MonthData,
    rng: np.random.Generator,
    sigma_load: float = 0.04,
    sigma_pv: float = 0.08,
    rho: float = 0.9,
) -> MonthData:
    out = MonthData(source=month.source + ":aug")
    for day in month.days:

        def _ar1(n: int, sigma: float) -> np.ndarray:
            err = np.zeros(n)
            white = sigma * np.sqrt(1 - rho ** 2)
            innovations = white * rng.standard_normal(max(0, n - 1))
            for step in range(1, n):
                err[step] = rho * err[step - 1] + innovations[step - 1]
            return err

        n_steps = len(day.load)
        load = np.maximum(0.0, day.load * (1 + _ar1(n_steps, sigma_load)))
        pv = np.maximum(0.0, day.pv * (1 + _ar1(n_steps, sigma_pv)))
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
    args = parser.parse_args()
    if not math.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        raise SystemExit("gamma must be finite and in (0, 1]")
    if not math.isfinite(args.lambda_value) or not 0.0 <= args.lambda_value <= 1.0:
        raise SystemExit("lambda must be finite and in [0, 1]")

    days = load_csv_days(Path(args.csv))
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

    base = load_system_config()
    cfg = make_bess_config(base, args.e_cap, args.p_rated, base.P_target_user)
    cfg.dt = csv_dt
    billing = args.billing
    import json
    from common import TOU_RULES, build_tariff_windows

    tariff = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
    cfg.price_peak = float(tariff.get("price_peak", cfg.price_peak))
    cfg.price_mid = float(tariff.get("price_mid", cfg.price_mid))
    cfg.price_off = float(tariff.get("price_off", cfg.price_off))
    cfg.T_cap = float(tariff.get("t_cap", cfg.T_cap))
    cfg.eta_ch = float(tariff.get("charge_efficiency", cfg.eta_ch))
    cfg.eta_dis = float(tariff.get("discharge_efficiency", cfg.eta_dis))
    cfg.SOC_min = float(tariff.get("minimum_soc", cfg.SOC_min))
    cfg.SOC_max = float(tariff.get("maximum_soc", cfg.SOC_max))
    cfg.SOC_eod = float(tariff.get("required_final_soc", cfg.SOC_eod))
    for key, value in build_tariff_windows(
        tariff["peak_windows"], tariff["off_windows"], cfg.dt
    ).items():
        setattr(cfg, key, value)
    billing = tariff.get("billing_mode", billing)
    TOU_RULES["sunday_no_peak"] = bool(tariff.get("sunday_no_peak", False))
    if billing == "tou":
        cfg.T_cap = 0.0

    tag = args.tag or f"ds_{args.e_cap:.0f}kwh_{args.p_rated:.0f}kw"
    if billing == "tou" and not tag.endswith("_tou"):
        tag += "_tou"

    train_months = month_blocks(train_days)
    val_month = MonthData(days=val_days, source="val")
    gamma = args.gamma
    env = BESSEnv(
        cfg,
        p_ref_kw=p_ref,
        d_run_init_kw=d_run0,
        gamma=gamma,
        control_dt_minutes=args.control_dt_minutes,
    )
    agent = PPOAgent(
        env.obs_dim,
        gamma=gamma,
        lam=args.lambda_value,
        seed=args.seed,
    )
    assert env.gamma == agent.gamma
    agent.meta = {
        "p_ref_kw": p_ref,
        "e_cap_kwh": args.e_cap,
        "p_rated_kw": args.p_rated,
        "obs_variant": "base",
        "obs_dim": env.obs_dim,
        "d_run_init_kw": d_run0,
        "gamma": gamma,
        "lambda": args.lambda_value,
        "native_dt_minutes": csv_dt * 60.0,
        "control_dt_minutes": env.control_dt_minutes,
        "native_steps_per_action": env.native_steps_per_action,
        "billing_mode": billing,
        "train_csv": str(args.csv),
        "test_range": [test_days[0].date_iso, test_days[-1].date_iso],
    }
    decisions_per_day = len(days[0].load) // env.native_steps_per_action
    buffer = RolloutBuffer(decisions_per_day * ROLLOUT_DAYS, env.obs_dim)

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
            "control_dt_minutes": env.control_dt_minutes,
            "native_steps_per_action": env.native_steps_per_action,
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
        f"native dt {csv_dt * 60:g}m | control dt {env.control_dt_minutes:g}m | "
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
    first_month = augment_month(train_months[0], rng)
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
            next_month = augment_month(
                train_months[month_index % len(train_months)], rng
            )
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
    best_agent = PPOAgent(env.obs_dim)
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
