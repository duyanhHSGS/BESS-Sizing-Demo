from __future__ import annotations

import argparse
import csv as _csv
import math
import time
from pathlib import Path

import numpy as np

from baselines import run_drl_policy, run_no_bess, run_oracle
from bess_env import BESSEnv
from benchmark import _rolling_30_minute_average
from common import RESULTS_DIR, load_system_config, make_bess_config, score_month
from settings import DEFAULT_PARAMETERS
from ppo_agent import PPOAgent, RolloutBuffer
from scenario_gen import DayData, MonthData


ROLLOUT_DAYS = 32


def load_csv_days(path: Path) -> list[DayData]:
    by_day: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            day = by_day.setdefault(
                row["date_iso"],
                {"load": [], "pv": [], "day_type": row["day_type"]},
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
                day_index=len(days) + 1,
                date_iso=iso,
            )
        )
    return days


def month_blocks(days: list[DayData]) -> list[MonthData]:
    blocks: dict[str, MonthData] = {}
    for day in days:
        key = str(day.date_iso)[:7]
        blocks.setdefault(key, MonthData(source=f"csv:{key}")).days.append(day)
    return [month for _, month in sorted(blocks.items()) if len(month.days) >= 15]


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
            for step in range(1, n):
                err[step] = rho * err[step - 1] + white * rng.standard_normal()
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--billing", choices=("2tc", "tou"), default="2tc")
    parser.add_argument("--tariff-json", type=str, default="")
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    args = parser.parse_args()

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
    if args.tariff_json:
        import json
        from common import TOU_RULES, build_tariff_windows

        tariff = json.loads(Path(args.tariff_json).read_text(encoding="utf-8"))
        cfg.price_peak = float(tariff.get("price_peak", cfg.price_peak))
        cfg.price_mid = float(tariff.get("price_mid", cfg.price_mid))
        cfg.price_off = float(tariff.get("price_off", cfg.price_off))
        cfg.T_cap = float(tariff.get("t_cap", cfg.T_cap))
        for key, value in build_tariff_windows(tariff["peak_windows"], tariff["off_windows"], cfg.dt).items():
            setattr(cfg, key, value)
        billing = tariff.get("billing_mode", billing)
        TOU_RULES["sunday_no_peak"] = bool(tariff.get("sunday_no_peak", False))
    else:
        from common import build_tariff_windows

        for key, value in build_tariff_windows(
            DEFAULT_PARAMETERS["billing_windows_expensive"],
            DEFAULT_PARAMETERS["billing_windows_cheap"],
            cfg.dt,
        ).items():
            setattr(cfg, key, value)
    if billing == "tou":
        cfg.T_cap = 0.0

    tag = args.tag or f"ds_{args.e_cap:.0f}kwh_{args.p_rated:.0f}kw"
    if billing == "tou" and not tag.endswith("_tou"):
        tag += "_tou"

    train_months = month_blocks(train_days)
    val_month = MonthData(days=val_days, source="val")
    env = BESSEnv(cfg, p_ref_kw=p_ref, d_run_init_kw=d_run0)
    agent = PPOAgent(env.obs_dim, seed=args.seed)
    agent.meta = {
        "p_ref_kw": p_ref,
        "e_cap_kwh": args.e_cap,
        "p_rated_kw": args.p_rated,
        "obs_variant": "base",
        "obs_dim": env.obs_dim,
        "d_run_init_kw": d_run0,
        "billing_mode": billing,
        "train_csv": str(args.csv),
        "test_range": [test_days[0].date_iso, test_days[-1].date_iso],
    }
    buffer = RolloutBuffer(len(days[0].load) * ROLLOUT_DAYS, env.obs_dim)

    val_base = score_month(run_no_bess(val_month, cfg)["p_grid_days"], cfg, days=val_days)["total_cost_vnd"]
    val_oracle = score_month(run_oracle(val_month, cfg)["p_grid_days"], cfg, days=val_days)["total_cost_vnd"]
    print(
        f"[train-ds] {len(days)} days | train {len(train_days)} / "
        f"val {len(val_days)} / test {len(test_days)} | "
        f"p_ref {p_ref:.0f} | val no-BESS {val_base/1e6:.0f}M, oracle {val_oracle/1e6:.0f}M",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    best_val = float("inf")
    curve = []
    month_index = 0
    obs = env.reset(augment_month(train_months[0], rng))
    steps = 0
    started = time.time()
    while steps < args.steps:
        action, logp, value = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, action, logp, reward, value, float(done))
        steps += 1
        if done:
            month_index += 1
            next_obs = env.reset(augment_month(train_months[month_index % len(train_months)], rng))
        obs = next_obs
        if not buffer.full():
            continue
        _, _, last_value = agent.act(obs)
        agent.update(buffer, 0.0 if done else last_value)
        result = run_drl_policy(val_month, cfg, agent, p_ref_kw=p_ref)
        val_cost = score_month(result["p_grid_days"], cfg, days=val_days)["total_cost_vnd"]
        saving = (val_base - val_cost) / val_base * 100
        gap = (val_cost - val_oracle) / val_oracle * 100
        curve.append({"steps": steps, "val_cost_vnd": val_cost, "oracle_gap_pct": gap, "saving_vs_nobess_pct": saving})
        if val_cost < best_val:
            best_val = val_cost
            agent.save(RESULTS_DIR / f"policy_{tag}.pt")
        print(
            f"  step {steps:>7} | val {val_cost/1e6:8.1f}M | saving {saving:5.1f}% | "
            f"gap {gap:6.1f}% | {steps/(time.time()-started):,.0f} sps",
            flush=True,
        )

    with (RESULTS_DIR / f"training_curve_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    from datetime import date

    test_month = MonthData(days=test_days, source="test")
    best_agent = PPOAgent(env.obs_dim)
    best_agent.load(RESULTS_DIR / f"policy_{tag}.pt")
    result = run_drl_policy(test_month, cfg, best_agent, p_ref_kw=p_ref)
    test_cost = score_month(result["p_grid_days"], cfg, days=test_days)["total_cost_vnd"]
    no_bess_cost = score_month(run_no_bess(test_month, cfg)["p_grid_days"], cfg, days=test_days)["total_cost_vnd"]
    test_saving = (no_bess_cost - test_cost) / no_bess_cost * 100
    best_agent.meta = {**agent.meta, "test_saving_pct": round(test_saving, 2), "trained": date.today().isoformat()}
    best_agent.save(RESULTS_DIR / f"policy_{tag}.pt")
    print(
        f"[train-ds] TEST {test_days[0].date_iso}->{test_days[-1].date_iso}: "
        f"{test_cost/1e6:.1f}M vs no-BESS {no_bess_cost/1e6:.1f}M -> saving {test_saving:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
