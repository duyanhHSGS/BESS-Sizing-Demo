"""train_grepo.py — train the GREPO agent (Hu et al. 2026) on simulator data.

Faithful to the paper's episode design: ONE EPISODE = ONE DAY (their
T=1440 at 1-min resolution; ours T=96 at 15-min). Each iteration samples a
random day from the simulator stream, rolls N_g parallel episodes with an
IDENTICAL exogenous trajectory and initial state (SOC and running-peak
floor randomised across iterations for state-space coverage), then runs one
GREPO update. Validation = deterministic rollout on the fixed held-out sim
month; best-val checkpointing as in train.py.

Run: python drl/train_grepo.py [--iters 400] [--group 8]
                               [--e-cap 750 --p-rated 350]
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from common import RESULTS_DIR, load_system_config, make_bess_config, score_month
from scenario_gen import generate_sim_month, MonthData
from bess_env import BESSEnv, OBS_DIM
from grepo_agent import GREPOAgent
from baselines import run_no_bess, run_oracle, run_drl_policy

VAL_SEED = 9999
TRAIN_SEED0 = 20_000        # distinct stream from PPO's 10k range
VAL_EVERY = 10


def _load_csv_days(path):
    """Nạp CSV cache site (date_iso,day_type,step,P_load_kW,P_pv_kW)."""
    import csv as _csv
    from scenario_gen import DayData
    by_day: dict = {}
    with open(path, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            d = by_day.setdefault(r["date_iso"], {
                "load": np.zeros(96), "pv": np.zeros(96),
                "day_type": r["day_type"]})
            t = int(r["step"])
            d["load"][t] = float(r["P_load_kW"])
            d["pv"][t] = float(r["P_pv_kW"])
    days = []
    for iso in sorted(by_day):
        v = by_day[iso]
        days.append(DayData(load=v["load"], pv=v["pv"],
                            day_type=v["day_type"], weather="tb",
                            day_index=len(days) + 1, date_iso=iso))
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--e-cap", type=float, default=None)
    ap.add_argument("--p-rated", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--csv", type=str, default=None,
                    help="train trên dữ liệu site thật (CSV cache) thay "
                         "vì simulator; tự split train/val như train PPO")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="trọng số hybrid baseline Eq.28")
    ap.add_argument("--std", type=float, default=0.30,
                    help="std cố định lambda của policy Gaussian")
    ap.add_argument("--tariff-json", type=str, default="",
                    help="Sizing Demo tariff_config.json path.")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = load_system_config()
    cfg = base
    if args.e_cap and args.p_rated:
        cfg = make_bess_config(base, args.e_cap, args.p_rated,
                               base.P_target_user)
    if args.tariff_json:
        import json as _json
        from pathlib import Path
        from common import TOU_RULES, build_tariff_windows

        tariff = _json.loads(Path(args.tariff_json).read_text(encoding="utf-8"))
        cfg.price_peak = float(tariff.get("price_peak", cfg.price_peak))
        cfg.price_mid = float(tariff.get("price_mid", cfg.price_mid))
        cfg.price_off = float(tariff.get("price_off", cfg.price_off))
        cfg.T_cap = float(tariff.get("t_cap", cfg.T_cap))
        for key, value in build_tariff_windows(tariff.get("peak_windows", ""), tariff.get("off_windows", "")).items():
            setattr(cfg, key, value)
        TOU_RULES["sunday_no_peak"] = bool(tariff.get("sunday_no_peak", False))
        if tariff.get("billing_mode") == "tou":
            cfg.T_cap = 0.0
    # ---- nguồn dữ liệu: simulator (mặc định) hoặc CSV site thật ----
    csv_days = None
    if args.csv:
        import math
        csv_days = _load_csv_days(args.csv)
        if len(csv_days) < 90:
            raise SystemExit(f"CSV chỉ {len(csv_days)} ngày — cần ≥90")
        peak = max(float(d.load.max()) for d in csv_days)
        p_ref = math.ceil(peak / 500.0) * 500.0
        # floor data-driven như PPO (fix bug site peaky)
        peaks = [float(np.convolve(np.maximum(0, d.load - d.pv), [.5, .5],
                                   mode="valid").max(initial=0.0))
                 for d in csv_days[:-60]]
        d_run0 = 0.5 * float(np.mean(peaks))
        train_days = csv_days[:-60]
        val_month = MonthData(source="csv_val")
        val_month.days = csv_days[-60:-30]
    else:
        p_ref = 500.0
        d_run0 = None
        val_month = generate_sim_month(VAL_SEED)
    tag = args.tag or f"grepo_{cfg.E_cap:.0f}kwh_{cfg.P_rated_nominal:.0f}kw"

    agent = GREPOAgent(OBS_DIM, n_group=args.group, seed=args.seed,
                       beta=args.beta, std=args.std)
    # meta trước validation đầu tiên để env val dùng đúng floor/p_ref
    agent.meta = {"p_ref_kw": p_ref, "algo": "grepo"}
    if d_run0 is not None:
        agent.meta["d_run_init_kw"] = d_run0
    make_env = lambda: BESSEnv(cfg, p_ref_kw=p_ref)   # noqa: E731
    val_base = score_month(run_no_bess(val_month, cfg)["p_grid_days"],
                           cfg)["total_cost_vnd"]
    val_oracle = score_month(run_oracle(val_month, cfg)["p_grid_days"],
                             cfg)["total_cost_vnd"]
    print(f"[grepo] config {tag} | group={args.group} | val no-BESS "
          f"{val_base/1e6:.1f}M, oracle {val_oracle/1e6:.1f}M VND", flush=True)

    curve = []
    best_val = float("inf")
    t0 = time.time()
    steps = 0
    rng = np.random.default_rng(args.seed)
    day_pool_month = None
    for it in range(args.iters):
        # one episode = one random day (paper's daily-episode design)
        if csv_days is not None:
            day = train_days[rng.integers(len(train_days))]
        else:
            if it % 10 == 0:
                day_pool_month = generate_sim_month(
                    TRAIN_SEED0 + args.seed * 1000 + it // 10)
            day = day_pool_month.days[rng.integers(len(day_pool_month.days))]
        episode = MonthData(days=[day], source="grepo_day")
        soc_init = float(rng.uniform(cfg.SOC_min + cfg.SOC_safety,
                                     cfg.SOC_max))
        if d_run0 is not None:      # site thật: floor data-driven ± jitter
            d_run_init = float(d_run0 * rng.uniform(0.8, 1.5))
        else:
            d_run_init = float(rng.uniform(0.5, 0.9) * p_ref)
        batch = agent.collect_group(make_env, episode, soc_init=soc_init,
                                    d_run_init=d_run_init)
        steps += batch[3].size
        losses = agent.update(*batch)
        if (it + 1) % VAL_EVERY != 0 and it + 1 != args.iters:
            continue
        res = run_drl_policy(val_month, cfg, agent, p_ref_kw=p_ref)
        val_cost = score_month(res["p_grid_days"], cfg)["total_cost_vnd"]
        gap = (val_cost - val_oracle) / val_oracle * 100
        sav = (val_base - val_cost) / val_base * 100
        curve.append({"steps": steps, "val_cost_vnd": val_cost,
                      "oracle_gap_pct": gap, "saving_vs_nobess_pct": sav})
        if val_cost < best_val:
            best_val = val_cost
            agent.meta = {"p_ref_kw": p_ref, "algo": "grepo",
                          "beta": args.beta, "std": args.std,
                          "e_cap_kwh": cfg.E_cap,
                          "p_rated_kw": cfg.P_rated_nominal}
            if d_run0 is not None:
                agent.meta["d_run_init_kw"] = d_run0
            agent.save(RESULTS_DIR / f"policy_{tag}.pt")
        print(f"  iter {it+1:>3}/{args.iters} | steps {steps:>7} | "
              f"val {val_cost/1e6:8.1f}M | saving {sav:5.1f}% | "
              f"gap {gap:6.1f}% | pi {losses['pi_loss']:+.3f} | "
              f"{steps/(time.time()-t0):,.0f} sps", flush=True)

    with open(RESULTS_DIR / f"training_curve_{tag}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)
    print(f"[grepo] done in {time.time()-t0:.0f}s. Best val "
          f"{best_val/1e6:.1f}M VND -> policy_{tag}.pt", flush=True)


if __name__ == "__main__":
    main()
