"""File-backed runtime/benchmark compatibility for the original Sizing Demo UI.

This module intentionally does not reimplement PPO physics. Offline policy traces
use the existing training/evaluation core; the Flask layer only reshapes those
results into the original browser contract and persists plain JSON files.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date as date_cls
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from flask import Response, jsonify, request
from og_ui_compat import (
    BENCH_DIR,
    DISPATCH_DIR,
    LIVE_DIR,
    ORACLE_CACHE_DIR,
    RESULTS_DIR,
    SHADOW_DIR,
    _dataset_by_id,
    _dataset_hash,
    _effective_config,
    _json_load,
    _json_save,
    _load_training_core,
    _rolling_average,
    _sample_batteries,
    _settings,
    _write_config,
    build_benchmark,
    ensure_og_dirs,
    list_checkpoints,
)

_JOBS: dict[str, dict[str, Any]] = {}
_JOB_CANCEL: dict[str, threading.Event] = {}
_JOB_LOCK = threading.RLock()
_LIVE_LOCK = threading.RLock()
_LIVE_AUTO: dict[str, threading.Event] = {}


def _safe_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value)[:120]


def _cfg_model(cfg_json: dict[str, Any], prefix: str):
    modules = _load_training_core()
    path = _write_config(cfg_json, prefix)
    return modules["common"].load_bess_drl_config(path), path


def _dataset_months(dataset, *, min_coverage: float = 0.0):
    modules = _load_training_core()
    days = modules["runner"].load_csv_days(dataset.path)
    return modules["runner"].complete_month_blocks(days, min_coverage=min_coverage)


def _checkpoint_agent(path: Path):
    modules = _load_training_core()
    raw = modules["runner"].torch.load(path, map_location="cpu", weights_only=True) if hasattr(modules["runner"], "torch") else None
    if raw is None:
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=True)
    meta = dict(raw.get("meta") or {})
    from bess_drl.engine.policy_contract import OBS_DIM, REQUIRED_POLICY_META

    obs_dim = int(meta.get("obs_dim", -1))
    if raw.get("algo") != "ppo" or obs_dim != OBS_DIM:
        raise ValueError(
            f"{path.name} is not compatible with the current PPO contract "
            f"(algo={raw.get('algo')!r}, obs_dim={obs_dim}, expected={OBS_DIM})"
        )
    for key, expected in REQUIRED_POLICY_META.items():
        if meta.get(key) != expected:
            raise ValueError(
                f"{path.name} has {key}={meta.get(key)!r}; expected {expected!r}"
            )
    agent = modules["agent"].PPOAgent(obs_dim)
    agent.load(path)
    return agent, meta


def _fixed_month_peak(grids: list[Any], common: Any) -> tuple[float, int]:
    best_peak = 0.0
    best_day = 0
    for index, grid in enumerate(grids):
        peak = float(common.fixed_pmax_day(grid))
        if peak >= best_peak:
            best_peak = peak
            best_day = index
    return best_peak, best_day


def _series_list(values: Any, digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def _method_days_for_month(
    *,
    month: Any,
    cfg: Any,
    result: dict[str, Any],
    common: Any,
    global_day_start: int,
    include_policy_activity: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grids = list(result["p_grid_days"])
    p_bess = list(result["p_bess_days"])
    socs = list(result["soc_days"])
    peak_kw, peak_owner = _fixed_month_peak(grids, common)
    demand_cost = peak_kw * cfg.t_cap
    score = common.score_month(
        grids,
        cfg,
        days=list(month.days),
        p_bess_days=p_bess,
        soc_days=socs,
    )
    out: list[dict[str, Any]] = []
    for local_index, day in enumerate(month.days):
        grid = _series_list(grids[local_index], 3)
        pb = _series_list(p_bess[local_index], 3)
        soc = [round(float(value) * 100.0, 3) for value in socs[local_index][: len(grid)]]
        rolling = _rolling_average(grid, 2)
        discharge = [max(0.0, value) for value in pb]
        charge = [max(0.0, -value) for value in pb]
        dt = 0.25
        tariff = common.tariff_vector(cfg, day=day)
        energy = float(sum(float(g) * float(price) * dt for g, price in zip(grid, tariff, strict=True)))
        discharged_kwh = sum(discharge) * dt
        wear = discharged_kwh * cfg.degradation_cost_per_kwh_discharged
        owner_demand = demand_cost if local_index == peak_owner else 0.0
        prorated_demand = demand_cost / max(1, len(month.days))
        total_owner = energy + owner_demand + wear
        row: dict[str, Any] = {
            "day_index": global_day_start + local_index,
            "date_iso": str(day.date_iso),
            "day_type": str(day.day_type),
            "grid": grid,
            "rolling_grid": rolling,
            "discharge": discharge,
            "grid_charge": charge,
            "solar_charge": [0.0] * len(grid),
            "soc": soc,
            "month_peak": {
                "value_kW": round(peak_kw, 3),
                "day_index": global_day_start + peak_owner,
            },
            "grid_kWh": sum(grid) * dt,
            "charged_kWh": sum(charge) * dt,
            "discharged_kWh": discharged_kwh,
            "final_soc": soc[-1] if soc else 0.0,
            "peak_grid_kW": max(rolling, default=0.0),
            "energy_bill_vnd": energy,
            "peak_bill_owner_vnd": owner_demand,
            "peak_bill_prorated_vnd": prorated_demand,
            "bill_with_owner_peak_vnd": total_owner,
            "bill_with_prorated_peak_vnd": energy + prorated_demand + wear,
            "wear_cost_vnd": wear,
            "total_operating_owner_vnd": total_owner,
        }
        if include_policy_activity:
            requested = result.get("p_requested_days", [])
            executed = result.get("p_executed_days", [])
            reasons = result.get("clip_reason_days", [])
            req = _series_list(requested[local_index], 3) if local_index < len(requested) else pb
            exe = _series_list(executed[local_index], 3) if local_index < len(executed) else pb
            clips = list(reasons[local_index]) if local_index < len(reasons) else [None] * len(pb)
            row["p_requested_kw"] = req
            row["p_executed_kw"] = exe
            row["clip_reason"] = clips
            row["blocked_action_pct"] = 100.0 * sum(
                1 for wanted, actual in zip(req, exe, strict=True) if abs(wanted - actual) > 1e-6
            ) / max(1, len(exe))
        out.append(row)
    month_summary = {
        **score,
        "peak_kw": peak_kw,
        "demand_cost_vnd": demand_cost,
        "wear_cost_vnd": float(score.get("degradation_cost_vnd", 0.0)),
    }
    return out, month_summary


def _oracle_payload(dataset, cfg_json: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    ensure_og_dirs()
    cfg_hash = str(cfg_json.get("meta", {}).get("configHash", "local"))
    fingerprint = hashlib.sha256(
        f"{_dataset_hash(dataset.path)}|{cfg_hash}|oracle-v2".encode()
    ).hexdigest()
    cache_path = ORACLE_CACHE_DIR / f"oracle_{fingerprint[:24]}.json"
    if cache_path.is_file() and not force:
        payload = _json_load(cache_path, {})
        if payload:
            payload.setdefault("cache", {})["hit"] = True
            return payload

    modules = _load_training_core()
    common = modules["common"]
    baselines = modules["baselines"]
    cfg, _ = _cfg_model(cfg_json, "oracle")
    months = _dataset_months(dataset, min_coverage=0.0)
    days: list[dict[str, Any]] = []
    month_scores: list[dict[str, Any]] = []
    global_day = 1
    statuses: list[str] = []
    for month in months:
        result = baselines.run_oracle(month, cfg)
        month_days, summary = _method_days_for_month(
            month=month,
            cfg=cfg,
            result=result,
            common=common,
            global_day_start=global_day,
            include_policy_activity=False,
        )
        statuses.append(str(result.get("solver_message") or result.get("solver_status") or "solved"))
        days.extend(month_days)
        month_scores.append(summary)
        global_day += len(month.days)

    total_grid = sum(float(item.get("grid_kWh", 0.0)) for item in days)
    total_cost = sum(float(item.get("total_cost_vnd", 0.0)) for item in month_scores)
    peak = max((float(item.get("pmax_month_kw", 0.0)) for item in month_scores), default=0.0)
    payload = {
        "status": statuses[-1] if statuses else "No calendar-month data available.",
        "summary": {
            "solved_day_count": len(days),
            "total_grid_kWh": total_grid,
            "total_bill_vnd": total_cost,
            "peak_grid_kW": peak,
            "seer_factor": float(_settings().get("billing_real_saving_factor", 0.60) or 0.60),
        },
        "days": days,
        "cache": {"hit": False, "fingerprint": fingerprint},
    }
    _json_save(cache_path, payload)
    return payload


def _replay_fingerprint(dataset, checkpoint: Path, cfg_json: dict[str, Any]) -> str:
    source = "|".join(
        (
            _dataset_hash(dataset.path),
            str(checkpoint.resolve()),
            str(checkpoint.stat().st_mtime_ns),
            str(cfg_json.get("meta", {}).get("configHash", "")),
            "policy-replay-v3",
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _policy_replay(policy_name: str, dataset=None, cfg_json: dict[str, Any] | None = None, *, force: bool = False) -> dict[str, Any]:
    ensure_og_dirs()
    dataset = dataset or _dataset_by_id(None)
    if dataset is None:
        raise ValueError("No compatible local CSV dataset found.")
    checkpoint = (RESULTS_DIR / Path(policy_name).name).resolve()
    if checkpoint.parent != RESULTS_DIR.resolve() or not checkpoint.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_name}")
    cfg_json = cfg_json or _effective_config(_settings())
    fingerprint = _replay_fingerprint(dataset, checkpoint, cfg_json)
    cache_path = DISPATCH_DIR / f"replay_{_safe_token(checkpoint.stem)}_{fingerprint[:20]}.json"
    if cache_path.is_file() and not force:
        cached = _json_load(cache_path, {})
        if cached:
            return cached

    modules = _load_training_core()
    common = modules["common"]
    baselines = modules["baselines"]
    cfg, _ = _cfg_model(cfg_json, "dispatch")
    agent, meta = _checkpoint_agent(checkpoint)
    p_ref = float(meta.get("p_ref_kw", 0.0))
    if p_ref <= 0.0:
        raise ValueError(f"{checkpoint.name} has no valid p_ref_kw")
    months = _dataset_months(dataset, min_coverage=0.0)
    days: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    global_day = 1
    for month in months:
        result = baselines.run_drl_policy(month, cfg, agent, p_ref_kw=p_ref, deterministic=True)
        month_days, summary = _method_days_for_month(
            month=month,
            cfg=cfg,
            result=result,
            common=common,
            global_day_start=global_day,
            include_policy_activity=True,
        )
        days.extend(month_days)
        summaries.append(summary)
        global_day += len(month.days)

    total_cost = sum(float(row.get("total_cost_vnd", 0.0)) for row in summaries)
    energy = sum(float(row.get("energy_cost_vnd", 0.0)) for row in summaries)
    demand = sum(float(row.get("demand_cost_vnd", 0.0)) for row in summaries)
    wear = sum(float(row.get("degradation_cost_vnd", 0.0)) for row in summaries)
    throughput = sum(float(row.get("throughput_kwh", 0.0)) for row in summaries)
    peak = max((float(row.get("pmax_month_kw", 0.0)) for row in summaries), default=0.0)
    blocked = sum(float(day.get("blocked_action_pct", 0.0)) * len(day.get("grid", [])) for day in days)
    slot_count = sum(len(day.get("grid", [])) for day in days)
    current_e = float(cfg_json["bess"]["eCapKwh"])
    current_p = float(cfg_json["bess"]["pRatedKw"])
    body_mismatch = (
        meta.get("e_cap_kwh") is not None
        and abs(float(meta["e_cap_kwh"]) - current_e) > 1e-6
    ) or (
        meta.get("p_rated_kw") is not None
        and abs(float(meta["p_rated_kw"]) - current_p) > 1e-6
    )
    warning = (
        f"TRAINED FOR ANOTHER BODY: checkpoint {meta.get('e_cap_kwh', '?')}/{meta.get('p_rated_kw', '?')} vs selected {current_e:g}/{current_p:g}."
        if body_mismatch
        else ""
    )
    payload = {
        "policy": checkpoint.name,
        "created": time.time(),
        "fingerprint": fingerprint,
        "dataset": dataset.source,
        "days": days,
        "summary": {
            "total_operating_cost_vnd": total_cost,
            "energy_cost_vnd": energy,
            "demand_cost_vnd": demand,
            "wear_cost_vnd": wear,
            "throughput_kwh": throughput,
            "peak_kw": peak,
            "blocked_action_pct": blocked / max(1, slot_count),
            "zero_export_violation_days": 0,
            "soc_violation_days": 0,
        },
        "meta": meta,
        "warning": warning,
        "body_mismatch": body_mismatch,
    }
    _json_save(cache_path, payload)
    return payload


def _latest_replay_for_policy(name: str) -> dict[str, Any] | None:
    prefix = f"replay_{_safe_token(Path(name).stem)}_"
    candidates = sorted(DISPATCH_DIR.glob(f"{prefix}*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    payload = _json_load(candidates[0], {})
    return payload or None


def _policy_rows() -> list[dict[str, Any]]:
    cfg_json = _effective_config(_settings())
    current_e = float(cfg_json["bess"]["eCapKwh"])
    current_p = float(cfg_json["bess"]["pRatedKw"])
    rows: list[dict[str, Any]] = []
    for item in list_checkpoints():
        meta = item.get("meta") or {}
        latest = _latest_replay_for_policy(item["name"])
        body_mismatch = False
        if meta.get("e_cap_kwh") is not None:
            body_mismatch |= abs(float(meta["e_cap_kwh"]) - current_e) > 1e-6
        if meta.get("p_rated_kw") is not None:
            body_mismatch |= abs(float(meta["p_rated_kw"]) - current_p) > 1e-6
        error = item.get("error")
        rows.append(
            {
                **item,
                "latest_run": {"created": latest.get("created", 0.0)} if latest else None,
                "warning": "TRAINED FOR ANOTHER BODY" if body_mismatch else "",
                "body_mismatch": body_mismatch,
                "runnable": not bool(error) and str(item.get("algo", "ppo")).lower() == "ppo",
                "reason": error or ("Current debloated core supports PPO checkpoints." if str(item.get("algo", "ppo")).lower() != "ppo" else ""),
            }
        )
    return rows


def _baseline_contestants(dataset, cfg_json: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark = build_benchmark(dataset, cfg_json)
    no_days = []
    for source in benchmark["days"]:
        no_days.append(
            {
                "day_index": source["day_index"],
                "date_iso": source["date_iso"],
                "day_type": source["day_type"],
                "grid": source["grid"],
                "rolling_grid": source["rolling_grid"],
                "soc": [],
                "wear_cost_vnd": 0.0,
                "total_operating_owner_vnd": source["bill_with_owner_peak_vnd"],
                "month_peak": source["month_peak"],
            }
        )
    no_summary = {
        "total_operating_cost_vnd": benchmark["summary"]["total_bill_vnd"],
        "energy_cost_vnd": sum(float(day["energy_bill_vnd"]) for day in benchmark["days"]),
        "demand_cost_vnd": benchmark["summary"]["total_bill_vnd"] - sum(float(day["energy_bill_vnd"]) for day in benchmark["days"]),
        "wear_cost_vnd": 0.0,
        "throughput_kwh": 0.0,
        "peak_kw": benchmark["summary"]["peak_grid_kW"],
        "blocked_action_pct": 0.0,
        "zero_export_violation_days": 0,
        "soc_violation_days": 0,
    }
    oracle = _oracle_payload(dataset, cfg_json)
    oracle_days = oracle.get("days", [])
    oracle_summary = {
        "total_operating_cost_vnd": oracle["summary"].get("total_bill_vnd", 0.0),
        "energy_cost_vnd": sum(float(day.get("energy_bill_vnd", 0.0)) for day in oracle_days),
        "demand_cost_vnd": sum(float(day.get("peak_bill_owner_vnd", 0.0)) for day in oracle_days),
        "wear_cost_vnd": sum(float(day.get("wear_cost_vnd", 0.0)) for day in oracle_days),
        "throughput_kwh": sum(float(day.get("charged_kWh", 0.0)) + float(day.get("discharged_kWh", 0.0)) for day in oracle_days),
        "peak_kw": oracle["summary"].get("peak_grid_kW", 0.0),
        "blocked_action_pct": 0.0,
        "zero_export_violation_days": 0,
        "soc_violation_days": 0,
    }
    return (
        {"id": "no_bess", "label": "No-BESS", "type": "reference", "days": no_days, "summary": no_summary},
        {"id": "oracle", "label": "Oracle", "type": "oracle", "days": oracle_days, "summary": oracle_summary},
    )


def _decorate_contestant(contestant: dict[str, Any], no_bess_total: float, oracle_total: float) -> dict[str, Any]:
    summary = contestant["summary"]
    total = float(summary.get("total_operating_cost_vnd", 0.0))
    if contestant["id"] == "oracle":
        relation = "Oracle reference"
    elif contestant["id"] == "no_bess":
        relation = "No-BESS reference"
    elif total < oracle_total - 1.0:
        relation = "Beats Oracle — investigate accounting"
    elif total >= no_bess_total:
        relation = "Worse than no-BESS"
    else:
        relation = "Between no-BESS and Oracle"
    rows = contestant.get("days", [])
    peak_day = max(rows, key=lambda day: float(day.get("month_peak", {}).get("value_kW", 0.0)), default={}).get("day_index", 1)
    expensive_day = max(rows, key=lambda day: float(day.get("total_operating_owner_vnd", 0.0)), default={}).get("day_index", peak_day)
    wear_day = max(rows, key=lambda day: float(day.get("wear_cost_vnd", 0.0)), default={}).get("day_index", peak_day)
    contestant["detectives"] = {
        "peak_day": peak_day,
        "most_expensive_day": expensive_day,
        "largest_oracle_regret_day": expensive_day,
        "highest_wear_day": wear_day,
        "safety_violation_days": [],
    }
    summary["blocks"] = [
        {
            "start_day": 1,
            "end_day": len(rows),
            **{key: value for key, value in summary.items() if key != "blocks"},
        }
    ]
    contestant["leaderboard"] = {
        "id": contestant["id"],
        "label": contestant["label"],
        "type": contestant["type"],
        **summary,
        "oracle_relation": relation,
        "body_mismatch": bool(contestant.get("body_mismatch")),
    }
    return contestant


def _build_benchmark_result(job_id: str, policies: list[str], cancel: threading.Event) -> None:
    started = time.time()
    try:
        dataset = _dataset_by_id(None)
        if dataset is None:
            raise ValueError("No compatible local CSV dataset found.")
        cfg_json = _effective_config(_settings())
        total = len(policies) + 2
        with _JOB_LOCK:
            _JOBS[job_id].update({"total": total, "completed": 0, "stage": "No-BESS reference"})
        no_bess, oracle = _baseline_contestants(dataset, cfg_json)
        contestants = [no_bess, oracle]
        with _JOB_LOCK:
            _JOBS[job_id]["completed"] = 2
            _JOBS[job_id]["stage"] = "Policy replays"
        for policy in policies:
            if cancel.is_set():
                with _JOB_LOCK:
                    _JOBS[job_id].update({"status": "cancelled", "stage": "cancelled", "elapsed_seconds": round(time.time() - started, 1)})
                return
            with _JOB_LOCK:
                _JOBS[job_id]["fighter"] = policy
            replay = _policy_replay(policy, dataset, cfg_json, force=True)
            contestants.append(
                {
                    "id": _safe_token(Path(policy).stem),
                    "label": policy,
                    "type": "policy",
                    "days": replay["days"],
                    "summary": replay["summary"],
                    "body_mismatch": replay.get("body_mismatch", False),
                }
            )
            with _JOB_LOCK:
                _JOBS[job_id]["completed"] += 1
                _JOBS[job_id]["elapsed_seconds"] = round(time.time() - started, 1)

        no_total = float(no_bess["summary"]["total_operating_cost_vnd"])
        oracle_total = float(oracle["summary"]["total_operating_cost_vnd"])
        contestants = [_decorate_contestant(row, no_total, oracle_total) for row in contestants]
        leaderboard = [row["leaderboard"] for row in contestants]
        policies_only = [row for row in leaderboard if row["type"] == "policy"]
        pool = policies_only or leaderboard
        champions = {
            "cheapest": min(pool, key=lambda row: float(row.get("total_operating_cost_vnd", math.inf)))["id"],
            "lowest_peak": min(pool, key=lambda row: float(row.get("peak_kw", math.inf)))["id"],
            "most_utility_saving": min(pool, key=lambda row: float(row.get("total_operating_cost_vnd", math.inf)))["id"],
            "closest_oracle": min(pool, key=lambda row: abs(float(row.get("total_operating_cost_vnd", 0.0)) - oracle_total))["id"],
            "lowest_wear": min(pool, key=lambda row: float(row.get("wear_cost_vnd", math.inf)))["id"],
            "safest": min(pool, key=lambda row: int(row.get("zero_export_violation_days", 0)) + int(row.get("soc_violation_days", 0)))["id"],
        }
        fingerprint = hashlib.sha256(
            f"{_dataset_hash(dataset.path)}|{cfg_json['meta']['configHash']}|{','.join(policies)}|{time.time_ns()}".encode()
        ).hexdigest()
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{fingerprint[:8]}"
        result = {
            "id": run_id,
            "created": time.time(),
            "fingerprint": fingerprint,
            "config_hash": cfg_json["meta"]["configHash"],
            "policy_names": list(policies),
            "snapshot": {
                "dataset": {
                    "filename": dataset.source,
                    "sha256": _dataset_hash(dataset.path),
                    "dt_minutes": dataset.res_min,
                },
                "shared_bess": {
                    "capacity_kwh": cfg_json["bess"]["eCapKwh"],
                    "power_kw": cfg_json["bess"]["pRatedKw"],
                    "wear_vnd_per_kwh": cfg_json["economics"]["degradationCostPerKwhDischarged"],
                },
            },
            "contestants": contestants,
            "leaderboard": leaderboard,
            "champions": champions,
        }
        _json_save(BENCH_DIR / f"{run_id}.json", result)
        with _JOB_LOCK:
            _JOBS[job_id].update(
                {
                    "status": "complete",
                    "stage": "complete",
                    "completed": total,
                    "run_id": run_id,
                    "fighter": None,
                    "elapsed_seconds": round(time.time() - started, 1),
                }
            )
    except Exception as exc:  # noqa: BLE001 - background job must publish any core failure to the UI.
        with _JOB_LOCK:
            _JOBS[job_id].update(
                {
                    "status": "failed",
                    "stage": "failed",
                    "error": str(exc),
                    "elapsed_seconds": round(time.time() - started, 1),
                }
            )


def _run_history() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(BENCH_DIR.glob("run_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = _json_load(path, {})
        if not payload:
            continue
        rows.append(
            {
                "id": payload.get("id", path.stem),
                "created": payload.get("created", path.stat().st_mtime),
                "dataset": payload.get("snapshot", {}).get("dataset", {}).get("filename", "dataset"),
                "fingerprint": payload.get("fingerprint", ""),
            }
        )
    return rows[:20]


def _live_path(session_id: str) -> Path:
    return LIVE_DIR / f"{_safe_token(session_id)}.json"


def _live_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": payload["id"],
        "source": payload.get("source"),
        "policy": payload.get("policy"),
        "days_done": int(payload.get("days_done", 0)),
        "days_total": len(payload.get("all_days", [])),
        "auto_interval_s": payload.get("auto_interval_s"),
        "error": payload.get("error"),
    }


def _live_visible(payload: dict[str, Any]) -> dict[str, Any]:
    count = int(payload.get("days_done", 0))
    days = payload.get("all_days", [])[:count]
    return {"status": _live_summary(payload), "days": days}


def _live_day_rows(replay: dict[str, Any], dataset) -> list[dict[str, Any]]:
    cfg_json = _effective_config(_settings())
    benchmark = build_benchmark(dataset, cfg_json)
    base_by_index = {int(day["day_index"]): day for day in benchmark["days"]}
    result = []
    cumulative_policy_by_month: dict[str, float] = defaultdict(float)
    cumulative_base_by_month: dict[str, float] = defaultdict(float)
    for day in replay["days"]:
        base = base_by_index.get(int(day["day_index"]), {})
        month = str(day.get("date_iso", ""))[:7]
        cumulative_policy_by_month[month] += float(day.get("total_operating_owner_vnd", 0.0))
        cumulative_base_by_month[month] += float(base.get("bill_with_owner_peak_vnd", 0.0))
        result.append(
            {
                "day": day["day_index"],
                "date": day["date_iso"],
                "day_type": day["day_type"],
                "methods": {
                    "no_bess": {
                        "mtd_total_vnd": cumulative_base_by_month[month],
                        "soc_end_pct": None,
                        "mtd_pmax_kw": base.get("month_peak", {}).get("value_kW", 0.0),
                    },
                    replay["policy"]: {
                        "mtd_total_vnd": cumulative_policy_by_month[month],
                        "soc_end_pct": day.get("final_soc", 0.0),
                        "mtd_pmax_kw": day.get("month_peak", {}).get("value_kW", 0.0),
                    },
                },
                "trace": {
                    "load": base.get("load", []),
                    "pv": base.get("pv", []),
                    "no_bess_grid": base.get("grid", []),
                    "policy_grid": day.get("grid", []),
                    "policy_soc": day.get("soc", []),
                },
            }
        )
    return result


def _stop_live_auto(session_id: str) -> None:
    with _LIVE_LOCK:
        event = _LIVE_AUTO.pop(session_id, None)
        if event:
            event.set()


def _auto_live_worker(session_id: str, interval_s: float, stop_event: threading.Event) -> None:
    path = _live_path(session_id)
    while not stop_event.wait(interval_s):
        with _LIVE_LOCK:
            payload = _json_load(path, {})
            if not payload:
                break
            total = len(payload.get("all_days", []))
            done = int(payload.get("days_done", 0))
            if done >= total:
                payload["auto_interval_s"] = None
                _json_save(path, payload)
                break
            payload["days_done"] = done + 1
            _json_save(path, payload)
    with _LIVE_LOCK:
        _LIVE_AUTO.pop(session_id, None)


def _shadow_config_path() -> Path:
    return SHADOW_DIR / "config.json"


def _shadow_connector_path() -> Path:
    return SHADOW_DIR / "connector.json"


def _shadow_days_path() -> Path:
    return SHADOW_DIR / "days.json"


def _shadow_months(days: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in days:
        grouped[str(row.get("date", ""))[:7]].append(row)
    result = []
    for month, rows in sorted(grouped.items()):
        ok = [row for row in rows if row.get("status") == "OK"]
        nobess = sum(float(row.get("nobess_total_vnd", 0.0)) for row in ok)
        policy = sum(float(row.get("policy_total_vnd", 0.0)) for row in ok)
        result.append(
            {
                "month": month,
                "n_days_ok": len(ok),
                "n_days_skipped": len(rows) - len(ok),
                "nobess_bill_vnd": nobess,
                "policy_bill_vnd": policy,
                "policy_vs_nobess_vnd": nobess - policy,
                "policy_violations": sum(int(row.get("policy_violations", 0)) for row in ok),
            }
        )
    return result


def _tb_login(connector: dict[str, Any]) -> str:
    base = str(connector.get("base_url") or "").rstrip("/")
    if not base.lower().startswith(("https://", "http://localhost", "http://127.0.0.1")):
        raise ValueError("ThingsBoard Base URL must use HTTPS (localhost HTTP is allowed for development).")
    body = json.dumps(
        {
            "username": connector.get("username", ""),
            "password": connector.get("password", ""),
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/auth/login",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    token = str(payload.get("token") or "")
    if not token:
        raise ValueError("ThingsBoard login returned no token.")
    return token


def _tb_keys(connector: dict[str, Any], token: str) -> list[str]:
    base = str(connector["base_url"]).rstrip("/")
    device = urllib.parse.quote(str(connector.get("device_id") or ""), safe="")
    req = urllib.request.Request(
        f"{base}/api/plugins/telemetry/DEVICE/{device}/keys/timeseries",
        headers={"X-Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [str(item) for item in payload] if isinstance(payload, list) else []


def _tb_fetch_dataset(connector: dict[str, Any], start_date: str, end_date: str) -> Any:
    token = _tb_login(connector)
    base = str(connector["base_url"]).rstrip("/")
    device = urllib.parse.quote(str(connector.get("device_id") or ""), safe="")
    timezone_name = str(connector.get("timezone") or "Asia/Bangkok")
    tz = ZoneInfo(timezone_name)
    start = datetime.combine(date_cls.fromisoformat(start_date), datetime.min.time(), tzinfo=tz)
    end = datetime.combine(date_cls.fromisoformat(end_date) + timedelta(days=1), datetime.min.time(), tzinfo=tz)
    load_keys = [item.strip() for item in str(connector.get("key_load") or "").split(",") if item.strip()]
    pv_keys = [item.strip() for item in str(connector.get("key_pv") or "").split(",") if item.strip()]
    keys = load_keys + pv_keys
    if not load_keys or not pv_keys:
        raise ValueError("ThingsBoard load and PV keys are required.")
    interval_minutes = int(connector.get("interval_minutes") or 15)
    if interval_minutes != 15:
        raise ValueError("The current PPO core requires a 15-minute ThingsBoard interval.")
    query = urllib.parse.urlencode(
        {
            "keys": ",".join(keys),
            "startTs": int(start.timestamp() * 1000),
            "endTs": int(end.timestamp() * 1000),
            "interval": 15 * 60 * 1000,
            "agg": "AVG",
            "limit": 100000,
        }
    )
    req = urllib.request.Request(
        f"{base}/api/plugins/telemetry/DEVICE/{device}/values/timeseries?{query}",
        headers={"X-Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    by_ts: dict[int, dict[str, float]] = defaultdict(dict)
    for key, points in payload.items():
        for point in points or []:
            try:
                by_ts[int(point["ts"])][key] = float(point["value"])
            except (KeyError, TypeError, ValueError):
                continue
    scale = float(connector.get("unit_scale") or 1.0)
    rows = []
    for ts_ms, values in sorted(by_ts.items()):
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=tz)
        step = dt.hour * 4 + dt.minute // 15
        if step < 0 or step >= 96:
            continue
        load = sum(values.get(key, 0.0) for key in load_keys) * scale
        pv = sum(values.get(key, 0.0) for key in pv_keys) * scale
        rows.append(
            {
                "date_iso": dt.date().isoformat(),
                "step": step,
                "day_type": "weekday" if dt.weekday() < 5 else "weekend",
                "P_load_kW": load,
                "P_pv_kW": max(0.0, pv),
            }
        )
    if not rows:
        raise ValueError("ThingsBoard returned no usable telemetry rows for the requested dates.")
    path = SHADOW_DIR / f"thingsboard_{start_date}_{end_date}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date_iso", "step", "day_type", "P_load_kW", "P_pv_kW"])
        writer.writeheader()
        writer.writerows(rows)
    from og_ui_compat import _inspect_dataset

    dataset = _inspect_dataset(path, "shadow")
    if dataset is None:
        raise ValueError("Downloaded ThingsBoard telemetry could not form a compatible dataset.")
    return dataset


def _shadow_worker(job_id: str, start_date: str | None, end_date: str | None, cancel: threading.Event) -> None:
    started = time.time()
    try:
        config = _json_load(_shadow_config_path(), {})
        policy = str(config.get("policy") or "")
        if not policy:
            raise ValueError("Save a Shadow policy first.")
        source_kind = str(config.get("source_kind") or "csv")
        if source_kind == "thingsboard":
            connector = _json_load(_shadow_connector_path(), {})
            old_days = _json_load(_shadow_days_path(), [])
            today = datetime.now(ZoneInfo(str(connector.get("timezone") or "Asia/Bangkok"))).date()
            if not start_date:
                if old_days:
                    start_date = (date_cls.fromisoformat(str(old_days[-1]["date"])) + timedelta(days=1)).isoformat()
                else:
                    start_date = (today - timedelta(days=29)).isoformat()
            end_date = end_date or today.isoformat()
            dataset = _tb_fetch_dataset(connector, start_date, end_date)
        else:
            dataset = _dataset_by_id(str(config.get("source") or ""))
            if dataset is None:
                raise ValueError("Configured Shadow CSV no longer exists.")
        cfg_json = _effective_config(
            _settings(),
            e_cap_kwh=float(config.get("e_cap_kwh") or 1250),
            p_rated_kw=float(config.get("p_rated_kw") or 450),
        )
        replay = _policy_replay(policy, dataset, cfg_json)
        benchmark = build_benchmark(dataset, cfg_json)
        base_by_date = {day["date_iso"]: day for day in benchmark["days"]}
        selected = []
        for day in replay["days"]:
            date_iso = str(day["date_iso"])
            if start_date and date_iso < start_date:
                continue
            if end_date and date_iso > end_date:
                continue
            selected.append(day)
        with _JOB_LOCK:
            _JOBS[job_id].update({"total": len(selected), "stage": "replaying policy"})
        existing = _json_load(_shadow_days_path(), [])
        by_date = {str(row.get("date")): row for row in existing}
        processed = 0
        for index, day in enumerate(selected, start=1):
            if cancel.is_set():
                with _JOB_LOCK:
                    _JOBS[job_id].update({"status": "cancelled", "stage": "cancelled"})
                return
            base = base_by_date.get(str(day["date_iso"]), {})
            trace = {
                "load": base.get("load", []),
                "pv": base.get("pv", []),
                "no_bess_grid": base.get("grid", []),
                "policy_grid": day.get("grid", []),
                "policy_soc": day.get("soc", []),
            }
            by_date[str(day["date_iso"])] = {
                "date": day["date_iso"],
                "status": "OK",
                "day_type": day["day_type"],
                "load_kwh": base.get("load_kWh", 0.0),
                "nobess_energy_vnd": base.get("energy_bill_vnd", 0.0),
                "policy_energy_vnd": day.get("energy_bill_vnd", 0.0),
                "nobess_total_vnd": base.get("bill_with_owner_peak_vnd", 0.0),
                "policy_total_vnd": day.get("total_operating_owner_vnd", 0.0),
                "policy_mtd_peak_kw": day.get("month_peak", {}).get("value_kW", 0.0),
                "policy_soc_end_pct": day.get("final_soc"),
                "policy_violations": 0,
                "trace": trace,
            }
            processed += 1
            with _JOB_LOCK:
                _JOBS[job_id].update(
                    {
                        "completed": index,
                        "current_date": day["date_iso"],
                        "elapsed_seconds": round(time.time() - started, 1),
                    }
                )
        rows = [by_date[key] for key in sorted(by_date)]
        _json_save(_shadow_days_path(), rows)
        with _JOB_LOCK:
            _JOBS[job_id].update(
                {
                    "status": "complete",
                    "stage": "complete",
                    "completed": len(selected),
                    "elapsed_seconds": round(time.time() - started, 1),
                    "result": {
                        "processed_ok": processed,
                        "skipped": 0,
                        "already_done": max(0, len(existing) + processed - len(rows)),
                    },
                }
            )
    except Exception as exc:  # noqa: BLE001 - publish any replay/core/network failure to the UI.
        with _JOB_LOCK:
            _JOBS[job_id].update({"status": "failed", "stage": "failed", "error": str(exc), "elapsed_seconds": round(time.time() - started, 1)})


def register_og_runtime_routes(app: Any) -> None:
    ensure_og_dirs()

    @app.get("/candidate-oracle/<int:index>")
    def og_candidate_oracle(index: int):
        candidates = _sample_batteries()
        if index < 0 or index >= len(candidates):
            return jsonify({"error": "candidate index out of range"}), 404
        dataset = _dataset_by_id(None)
        if dataset is None:
            return jsonify({"error": "No compatible local CSV dataset found."}), 400
        candidate = dict(candidates[index])
        try:
            cfg_json = _effective_config(
                _settings(),
                e_cap_kwh=float(candidate["battery_capacity_kWh"]),
                p_rated_kw=float(candidate["battery_power_limit_kW"]),
            )
            candidate["oracle"] = _oracle_payload(dataset, cfg_json, force=request.args.get("force") == "1")
            return jsonify({"index": index, "candidate": candidate})
        except Exception as exc:  # noqa: BLE001 - solver errors belong in the original UI progress box.
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/dispatch/policies")
    def og_dispatch_policies():
        return jsonify({"policies": _policy_rows()})

    @app.get("/api/dispatch/policy-traces")
    def og_dispatch_traces():
        names = [item for item in str(request.args.get("policies") or "").split(",") if item]
        traces: dict[str, Any] = {}
        warnings: list[str] = []
        for name in names:
            replay = _latest_replay_for_policy(Path(name).name)
            if replay is None:
                warnings.append(f"{name}: no saved trace yet; click Create saved run.")
                continue
            traces[name] = replay
            if replay.get("warning"):
                warnings.append(str(replay["warning"]))
        return jsonify({"policies": traces, "warnings": warnings})

    @app.post("/api/dispatch/run")
    def og_dispatch_run():
        payload = request.get_json(silent=True) or {}
        names = [Path(str(name)).name for name in payload.get("policies", [])]
        dataset = _dataset_by_id(None)
        if dataset is None:
            return jsonify({"error": "No compatible local CSV dataset found."}), 400
        cfg_json = _effective_config(_settings())
        warnings = []
        try:
            for name in names:
                replay = _policy_replay(name, dataset, cfg_json, force=True)
                if replay.get("warning"):
                    warnings.append(replay["warning"])
            return jsonify({"ok": True, "warnings": warnings})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc), "warnings": warnings}), 400

    @app.get("/api/benchmarking/context")
    def og_benchmark_context():
        dataset = _dataset_by_id(None)
        cfg_json = _effective_config(_settings())
        return jsonify(
            {
                "dataset": {
                    "filename": dataset.source if dataset else None,
                    "sha256": _dataset_hash(dataset.path) if dataset else "",
                    "dt_minutes": dataset.res_min if dataset else 0.0,
                },
                "shared_bess": {
                    "capacity_kwh": cfg_json["bess"]["eCapKwh"],
                    "power_kw": cfg_json["bess"]["pRatedKw"],
                    "wear_vnd_per_kwh": cfg_json["economics"]["degradationCostPerKwhDischarged"],
                },
                "oracle_ready": dataset is not None,
                "oracle_status": "Oracle is computed from the current fixed-block monthly LP when the tournament starts." if dataset else "No compatible dataset.",
                "policies": _policy_rows(),
                "history": _run_history(),
            }
        )

    @app.post("/api/benchmarking/jobs")
    def og_benchmark_job_start():
        payload = request.get_json(silent=True) or {}
        policies = [Path(str(name)).name for name in payload.get("policies", [])]
        if not policies:
            return jsonify({"error": "Select at least one policy."}), 400
        job_id = f"bench_{uuid4().hex[:12]}"
        cancel = threading.Event()
        with _JOB_LOCK:
            _JOBS[job_id] = {"id": job_id, "status": "running", "stage": "starting", "completed": 0, "total": len(policies) + 2, "elapsed_seconds": 0.0, "fighter": None}
            _JOB_CANCEL[job_id] = cancel
        threading.Thread(target=_build_benchmark_result, args=(job_id, policies, cancel), daemon=True, name=f"bess-benchmark-{job_id}").start()
        return jsonify(_JOBS[job_id])

    @app.get("/api/benchmarking/jobs/<job_id>")
    def og_benchmark_job_status(job_id: str):
        with _JOB_LOCK:
            job = _JOBS.get(job_id)
            if not job:
                return jsonify({"error": "benchmark job not found"}), 404
            return jsonify(dict(job))

    @app.post("/api/benchmarking/jobs/<job_id>/cancel")
    def og_benchmark_job_cancel(job_id: str):
        with _JOB_LOCK:
            event = _JOB_CANCEL.get(job_id)
            if event is None:
                return jsonify({"error": "benchmark job not found"}), 404
            event.set()
            return jsonify({"ok": True})

    @app.get("/api/benchmarking/runs/<run_id>")
    def og_benchmark_run(run_id: str):
        payload = _json_load(BENCH_DIR / f"{_safe_token(run_id)}.json", {})
        return jsonify(payload) if payload else (jsonify({"error": "benchmark run not found"}), 404)

    @app.get("/api/benchmarking/runs/<run_id>/export.json")
    def og_benchmark_export_json(run_id: str):
        payload = _json_load(BENCH_DIR / f"{_safe_token(run_id)}.json", {})
        if not payload:
            return jsonify({"error": "benchmark run not found"}), 404
        return Response(json.dumps(payload, indent=2), mimetype="application/json", headers={"Content-Disposition": f"attachment; filename={_safe_token(run_id)}.json"})

    @app.get("/api/benchmarking/runs/<run_id>/export.csv")
    def og_benchmark_export_csv(run_id: str):
        payload = _json_load(BENCH_DIR / f"{_safe_token(run_id)}.json", {})
        if not payload:
            return jsonify({"error": "benchmark run not found"}), 404
        buffer = io.StringIO()
        fields = ["id", "label", "type", "total_operating_cost_vnd", "energy_cost_vnd", "demand_cost_vnd", "wear_cost_vnd", "throughput_kwh", "peak_kw", "blocked_action_pct", "oracle_relation"]
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for row in payload.get("leaderboard", []):
            writer.writerow({key: row.get(key) for key in fields})
        return Response(buffer.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={_safe_token(run_id)}.csv"})

    @app.get("/api/live-runs")
    def og_live_list():
        sessions = []
        for path in sorted(LIVE_DIR.glob("live_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            payload = _json_load(path, {})
            if payload:
                sessions.append(_live_summary(payload))
        return jsonify({"sessions": sessions})

    @app.post("/api/live-runs")
    def og_live_create():
        payload = request.get_json(silent=True) or {}
        policy = Path(str(payload.get("policy") or "")).name
        dataset = _dataset_by_id(None)
        if dataset is None:
            return jsonify({"error": "No compatible local CSV dataset found."}), 400
        try:
            replay = _policy_replay(policy, dataset, _effective_config(_settings()))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400
        session_id = f"live_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        session = {
            "id": session_id,
            "source": dataset.source,
            "policy": policy,
            "days_done": 0,
            "auto_interval_s": None,
            "error": None,
            "tariff": _settings(),
            "all_days": _live_day_rows(replay, dataset),
        }
        _json_save(_live_path(session_id), session)
        return jsonify(_live_summary(session))

    @app.get("/api/live-runs/<session_id>")
    def og_live_get(session_id: str):
        payload = _json_load(_live_path(session_id), {})
        return jsonify(_live_visible(payload)) if payload else (jsonify({"error": "live session not found"}), 404)

    @app.post("/api/live-runs/<session_id>/step")
    def og_live_step(session_id: str):
        with _LIVE_LOCK:
            path = _live_path(session_id)
            payload = _json_load(path, {})
            if not payload:
                return jsonify({"error": "live session not found"}), 404
            total = len(payload.get("all_days", []))
            payload["days_done"] = min(total, int(payload.get("days_done", 0)) + 1)
            _json_save(path, payload)
            return jsonify({"done": payload["days_done"] >= total})

    @app.post("/api/live-runs/<session_id>/auto")
    def og_live_auto(session_id: str):
        payload = request.get_json(silent=True) or {}
        interval_s = max(0.1, float(payload.get("interval_s") or 3.0))
        with _LIVE_LOCK:
            path = _live_path(session_id)
            session = _json_load(path, {})
            if not session:
                return jsonify({"error": "live session not found"}), 404
            _stop_live_auto(session_id)
            stop = threading.Event()
            _LIVE_AUTO[session_id] = stop
            session["auto_interval_s"] = interval_s
            _json_save(path, session)
            threading.Thread(target=_auto_live_worker, args=(session_id, interval_s, stop), daemon=True, name=f"bess-live-{session_id}").start()
        return jsonify({"ok": True})

    @app.post("/api/live-runs/<session_id>/stop")
    def og_live_stop(session_id: str):
        _stop_live_auto(session_id)
        path = _live_path(session_id)
        payload = _json_load(path, {})
        if payload:
            payload["auto_interval_s"] = None
            _json_save(path, payload)
        return jsonify({"ok": True})

    @app.delete("/api/live-runs/<session_id>")
    def og_live_delete(session_id: str):
        _stop_live_auto(session_id)
        _live_path(session_id).unlink(missing_ok=True)
        return jsonify({"ok": True})

    @app.get("/api/shadow/connector")
    def og_shadow_connector_get():
        saved = _json_load(_shadow_connector_path(), {})
        public = {key: value for key, value in saved.items() if key != "password"}
        public["password_configured"] = bool(saved.get("password"))
        return jsonify(public)

    @app.post("/api/shadow/connector")
    def og_shadow_connector_save():
        incoming = request.get_json(silent=True) or {}
        saved = _json_load(_shadow_connector_path(), {})
        if not incoming.get("password") and saved.get("password"):
            incoming["password"] = saved["password"]
        _json_save(_shadow_connector_path(), incoming)
        public = {key: value for key, value in incoming.items() if key != "password"}
        public["password_configured"] = bool(incoming.get("password"))
        return jsonify(public)

    @app.post("/api/shadow/connector/test")
    def og_shadow_connector_test():
        connector = _json_load(_shadow_connector_path(), {})
        try:
            token = _tb_login(connector)
            keys = _tb_keys(connector, token)
            wanted = [item.strip() for field in ("key_load", "key_pv") for item in str(connector.get(field) or "").split(",") if item.strip()]
            missing = [item for item in wanted if item not in keys]
            return jsonify({"message": "ThingsBoard login and telemetry-key inspection succeeded.", "available_keys": keys, "missing_keys": missing})
        except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/shadow/config")
    def og_shadow_config_get():
        config = _json_load(_shadow_config_path(), {})
        config.setdefault("source_kind", "csv")
        config.setdefault("parameters", _settings())
        return jsonify(config)

    @app.post("/api/shadow/config")
    def og_shadow_config_save():
        config = request.get_json(silent=True) or {}
        config["parameters"] = _settings()
        _json_save(_shadow_config_path(), config)
        return jsonify(config)

    @app.post("/api/shadow/catchup")
    def og_shadow_catchup():
        payload = request.get_json(silent=True) or {}
        job_id = f"shadow_{uuid4().hex[:12]}"
        cancel = threading.Event()
        with _JOB_LOCK:
            _JOBS[job_id] = {"id": job_id, "status": "running", "stage": "starting", "completed": 0, "total": 0, "elapsed_seconds": 0.0, "current_date": None}
            _JOB_CANCEL[job_id] = cancel
        threading.Thread(target=_shadow_worker, args=(job_id, payload.get("start_date"), payload.get("end_date"), cancel), daemon=True, name=f"bess-shadow-{job_id}").start()
        return jsonify(_JOBS[job_id])

    @app.get("/api/shadow/jobs/<job_id>")
    def og_shadow_job(job_id: str):
        with _JOB_LOCK:
            job = _JOBS.get(job_id)
            return jsonify(dict(job)) if job else (jsonify({"error": "shadow job not found"}), 404)

    @app.post("/api/shadow/jobs/<job_id>/cancel")
    def og_shadow_cancel(job_id: str):
        with _JOB_LOCK:
            event = _JOB_CANCEL.get(job_id)
            if event is None:
                return jsonify({"error": "shadow job not found"}), 404
            event.set()
        return jsonify({"ok": True})

    @app.post("/api/shadow/reset")
    def og_shadow_reset():
        _shadow_days_path().unlink(missing_ok=True)
        return jsonify({"ok": True})

    @app.get("/api/shadow/days")
    def og_shadow_days():
        return jsonify({"days": _json_load(_shadow_days_path(), [])})

    @app.get("/api/shadow/monthly")
    def og_shadow_monthly():
        days = _json_load(_shadow_days_path(), [])
        return jsonify({"months": _shadow_months(days)})
