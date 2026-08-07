"""train_ppo2_dataset.py — Training loop for PPO2 (decomposed critics, PopArt, squashed Gaussian).

Senior-parity PPO2 experiment:
  - Uses PPO2Env, not the original BESSEnv
  - 17D causal/block-aware observation
  - Physical-only action feasibility, fixed 30-minute demand blocks
  - Decomposed energy/peak rewards with PopArt critics
  - Latent-preserving tanh-squashed Gaussian PPO
  - Separate actor/critic learning rates, low initial action std, KL early stop
  - Forecast-free, one policy action per native data row
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from bess.evaluation.baselines import run_no_bess
from bess.core.ppo2_env import PPO2Env
from bess.evaluation.benchmark import _rolling_30_minute_average
from bess.core.common import RESULTS_DIR, tariff_vector_day
from bess.evaluation.oracle.oracle_cache import load_cached_training_grids
from bess.agents.ppo2_agent import PPO2Agent, RolloutBuffer, resolve_ppo2_device
from bess.core.scenario_gen import MonthData
from bess.core.settings import PPO2_GAMMA, PPO2_LAM_ENERGY, PPO2_LAM_PEAK
from bess.training.training_common import (
    augment_month,
    build_training_bess_config,
    load_training_days,
    month_blocks,
)
from bess.training.training_reports import write_curve, write_report
from bess.forecasting.weather_forecast import fit_attach_forecasts

ROLLOUT_DAYS = 30
LOG_EVERY_UPDATES = 1
PPO2_ACTOR_LR = 3e-5
PPO2_CRITIC_LR = 3e-4
PPO2_INIT_STD = 0.15
PPO2_CLIP_PENALTY_PER_KWH = 100.0


def _fixed_block_pmax_day(grid: np.ndarray, block_slots: int) -> float:
    values = np.maximum(0.0, np.asarray(grid, dtype=np.float64))
    if len(values) % block_slots != 0:
        raise ValueError("grid day must contain complete fixed demand blocks")
    if len(values) == 0:
        return 0.0
    return float(values.reshape(-1, block_slots).mean(axis=1).max(initial=0.0))


def _score_ppo2_month(p_grid_days, cfg, *, days) -> dict:
    block_slots = round(0.5 / cfg.dt)
    energy = 0.0
    peak = 0.0
    for index, grid in enumerate(p_grid_days):
        grid = np.maximum(0.0, np.asarray(grid, dtype=np.float64))
        tariff = tariff_vector_day(cfg, days[index])
        energy += float(np.sum(grid * tariff) * cfg.dt)
        peak = max(peak, _fixed_block_pmax_day(grid, block_slots))
    demand = peak * cfg.T_cap
    return {
        "energy_cost_vnd": energy,
        "demand_cost_vnd": demand,
        "total_cost_vnd": energy + demand,
        "pmax_month_kw": peak,
    }


def _run_ppo2_policy(month: MonthData, cfg, agent, *, p_ref_kw: float) -> dict:
    """Deterministic senior-parity rollout used for PPO2 validation/test."""
    env = PPO2Env(
        cfg,
        p_ref_kw=p_ref_kw,
        degradation_cost_per_kwh_discharged=50.0,
        clip_penalty_per_kwh=PPO2_CLIP_PENALTY_PER_KWH,
    )
    obs = env.reset(month)
    done = False
    while not done:
        action = agent.predict_action(obs)
        obs, _reward, done, _info = env.step(action)
    return {
        "p_grid_days": env.log_grid,
        "soc_days": env.log_soc,
        "p_bess_days": env.log_pbess,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--e-cap", type=float, required=True)
    parser.add_argument("--p-rated", type=float, required=True)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--gamma", type=float, default=PPO2_GAMMA)
    parser.add_argument("--lam-energy", type=float, default=PPO2_LAM_ENERGY)
    parser.add_argument("--lam-peak", type=float, default=PPO2_LAM_PEAK)
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not math.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        raise SystemExit("gamma must be finite and in (0, 1]")
    if not math.isfinite(args.lam_energy) or not 0.0 <= args.lam_energy <= 1.0:
        raise SystemExit("lam-energy must be finite and in [0, 1]")
    if not math.isfinite(args.lam_peak) or not 0.0 <= args.lam_peak <= 1.0:
        raise SystemExit("lam-peak must be finite and in [0, 1]")

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

    tag = args.tag or f"ppo2_{args.e_cap:.0f}kwh_{args.p_rated:.0f}kw"
    if billing == "tou" and not tag.endswith("_tou"):
        tag += "_tou"

    train_months = month_blocks(train_days)
    val_month = MonthData(days=val_days, source="val")
    gamma = args.gamma
    if abs(args.control_dt_minutes - csv_dt * 60.0) > 1e-9:
        raise SystemExit(
            "PPO2 senior-parity mode requires one policy action per native data row; "
            f"control_dt_minutes must equal {csv_dt * 60.0:g}"
        )
    if args.obs_variant != "base":
        raise SystemExit("PPO2 senior-parity mode is forecast-free and requires --obs-variant base")
    env = PPO2Env(
        cfg,
        p_ref_kw=p_ref,
        degradation_cost_per_kwh_discharged=50.0,
        clip_penalty_per_kwh=PPO2_CLIP_PENALTY_PER_KWH,
    )
    learner_device = resolve_ppo2_device(args.device)
    agent = PPO2Agent(
        env.obs_dim,
        gamma=gamma,
        lam_energy=args.lam_energy,
        lam_peak=args.lam_peak,
        actor_lr=PPO2_ACTOR_LR,
        critic_lr=PPO2_CRITIC_LR,
        log_std_init=math.log(PPO2_INIT_STD),
        seed=args.seed,
        device=learner_device,
    )
    agent.meta = {
        "p_ref_kw": p_ref,
        "e_cap_kwh": args.e_cap,
        "p_rated_kw": args.p_rated,
        "obs_variant": args.obs_variant,
        "obs_dim": env.obs_dim,
        "observation_schema": "senior_causal_block_aware",
        "action_distribution": "tanh_squashed_gaussian",
        "action_mapping": "physical_feasible_only",
        "demand_window": "fixed_30m_block_v1",
        "gamma": gamma,
        "lam_energy": args.lam_energy,
        "lam_peak": args.lam_peak,
        "native_dt_minutes": csv_dt * 60.0,
        "control_dt_minutes": csv_dt * 60.0,
        "native_steps_per_action": 1,
        "billing_mode": billing,
        "actor_learning_rate": PPO2_ACTOR_LR,
        "critic_learning_rate": PPO2_CRITIC_LR,
        "initial_action_std": PPO2_INIT_STD,
        "clip_penalty_per_kwh": PPO2_CLIP_PENALTY_PER_KWH,
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
        # PPO2Agent doesn't carry a forecast_bundle attribute; skip embedding
    decisions_per_day = len(days[0].load)
    buffer = RolloutBuffer(decisions_per_day * ROLLOUT_DAYS, env.obs_dim)

    val_base = _score_ppo2_month(
        run_no_bess(val_month, cfg)["p_grid_days"], cfg, days=val_days
    )["total_cost_vnd"]
    oracle_grids = load_cached_training_grids(
        args.oracle_cache, [day.day_index for day in val_days]
    )
    val_oracle = _score_ppo2_month(oracle_grids, cfg, days=val_days)["total_cost_vnd"]
    curve_path = RESULTS_DIR / f"training_curve_{tag}.csv"
    report_path = RESULTS_DIR / f"training_report_{tag}.json"
    report = {
        "version": 1,
        "status": "running",
        "algorithm": "ppo2",
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
            "lam_energy": args.lam_energy,
            "lam_peak": args.lam_peak,
            "rollout_days": ROLLOUT_DAYS,
            "native_dt_minutes": csv_dt * 60.0,
            "control_dt_minutes": csv_dt * 60.0,
            "native_steps_per_action": 1,
            "device_requested": args.device,
            "device": "cpu",
        },
        "billing_mode": billing,
        "p_ref_kw": p_ref,
        "d_run_init_kw": d_run0,
        "validation": {"no_bess_vnd": val_base, "oracle_vnd": val_oracle},
    }
    write_curve(curve_path, [])
    write_report(report_path, report)
    print(
        f"[train-ppo2] {len(days)} days | train {len(train_days)} / "
        f"val {len(val_days)} / test {len(test_days)} | "
        f"gamma {gamma:g} | lam_E {args.lam_energy:g} lam_P {args.lam_peak:g} | "
        f"cpu learner | "
        f"native dt {csv_dt * 60:g}m | control dt {csv_dt * 60:g}m | "
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
        diag = agent.diagnostics
        print(
            f"  perf rollout {perf['rollout']:.2f}s | augment {perf['augment']:.2f}s | "
            f"ppo {perf['update']:.2f}s | validate {perf['validation']:.2f}s | "
            f"score {perf['scoring']:.2f}s | io {perf['io']:.2f}s | "
            f"checkpoint {perf['checkpoint']:.2f}s | "
            f"{perf['native_rows']/rollout_seconds:,.0f} native rows/s | "
            f"{perf['decisions']/rollout_seconds:,.0f} decisions/s | "
            f"adv_std {diag.get('adv_raw_std', 0):.3f} | "
            f"log_std {diag.get('log_std', 0):.3f} | "
            f"V_std_E/P {diag.get('value_std_energy', 0):.2f}/{diag.get('value_std_peak', 0):.2f} | "
            f"KL {diag.get('approx_kl', 0):.4f} @ {diag.get('epochs_run', '?')}eps",
            flush=True,
        )
        for key in perf:
            perf[key] = 0 if key in ("decisions", "native_rows") else 0.0

    month_index = 0
    augment_started = time.perf_counter()
    first_month = train_months[0] if args.obs_variant == "fc" else augment_month(train_months[0], rng)
    perf["augment"] += time.perf_counter() - augment_started
    obs = env.reset(first_month, d_run_shaping_init_kw=d_run0)
    steps = 0
    updates = 0
    started = time.time()
    rollout_started = time.perf_counter()
    while steps < args.steps:
        # PPO2Agent.act() returns (action, logp, latent, v_e, v_p)
        action, logp, latent, v_e, v_p = agent.act(obs)
        next_obs, reward, done, step_info = env.step(action)
        rew_e = -(
            float(step_info["rew_energy_delta"])
            + float(step_info["rew_deg_cost"])
            + float(step_info["rew_terminal_cost"])
            + float(step_info["rew_clip_cost"])
        ) / env.reward_scale_vnd
        rew_p = -float(step_info["rew_peak_delta"]) / env.reward_scale_vnd
        if not step_info["action_held"]:
            buffer.add(obs, action, latent, logp, rew_e, rew_p, v_e, v_p, float(done))
        steps += 1
        perf["decisions"] += 1
        perf["native_rows"] += 1
        if done:
            month_index += 1
            augment_started = time.perf_counter()
            source_month = train_months[month_index % len(train_months)]
            next_month = source_month if args.obs_variant == "fc" else augment_month(source_month, rng)
            perf["augment"] += time.perf_counter() - augment_started
            next_obs = env.reset(next_month, d_run_shaping_init_kw=d_run0)
        obs = next_obs
        # Senior parity: anneal immediately before each PPO update.
        if not buffer.full():
            continue
        perf["rollout"] += time.perf_counter() - rollout_started
        updates += 1
        update_started = time.perf_counter()
        _, _, _, last_v_e, last_v_p = agent.act(obs)
        agent.anneal_lr(steps / max(1, args.steps))
        agent.update(buffer, 0.0 if done else last_v_e, 0.0 if done else last_v_p)
        perf["update"] += time.perf_counter() - update_started
        validation_started = time.perf_counter()
        result = _run_ppo2_policy(val_month, cfg, agent, p_ref_kw=p_ref)
        perf["validation"] += time.perf_counter() - validation_started
        scoring_started = time.perf_counter()
        val_cost = _score_ppo2_month(result["p_grid_days"], cfg, days=val_days)["total_cost_vnd"]
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
            diag = agent.diagnostics
            print(
                f"  update {updates:>4} | step {steps:>7} | val {val_cost/1e6:8.1f}M | "
                f"saving {saving:5.1f}% | gap {gap:6.1f}% | "
                f"KL {diag.get('approx_kl', 0):.4f} | log_σ {diag.get('log_std', 0):.3f} | "
                f"{steps/(time.time()-started):,.0f} sps",
                flush=True,
            )
            print_performance()
        if is_best:
            diag = agent.diagnostics
            print(
                f"  best   {updates:>4} | step {steps:>7} | val {val_cost/1e6:8.1f}M | "
                f"saving {saving:5.1f}% | gap {gap:6.1f}% | "
                f"KL {diag.get('approx_kl', 0):.4f} | checkpoint updated",
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
        result = _run_ppo2_policy(val_month, cfg, agent, p_ref_kw=p_ref)
        val_cost = _score_ppo2_month(result["p_grid_days"], cfg, days=val_days)["total_cost_vnd"]
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
    best_agent = PPO2Agent(env.obs_dim, seed=args.seed, device=learner_device)
    best_agent.load(RESULTS_DIR / f"policy_{tag}.pt")
    result = _run_ppo2_policy(test_month, cfg, best_agent, p_ref_kw=p_ref)
    test_cost = _score_ppo2_month(result["p_grid_days"], cfg, days=test_days)["total_cost_vnd"]
    no_bess_cost = _score_ppo2_month(
        run_no_bess(test_month, cfg)["p_grid_days"], cfg, days=test_days
    )["total_cost_vnd"]
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
        f"[train-ppo2] TEST {test_days[0].date_iso}->{test_days[-1].date_iso}: "
        f"{test_cost/1e6:.1f}M vs no-BESS {no_bess_cost/1e6:.1f}M -> saving {test_saving:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()