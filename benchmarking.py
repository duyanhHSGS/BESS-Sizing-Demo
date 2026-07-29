from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

import benchmark_store
import dispatch_runner
import oracle_cache
from baselines import run_no_bess, run_sadrbc
from benchmark import _demand_charge, _month_start_day, selected_data_path
from benchmark_jobs import BenchmarkCancelled
from common import check_hard_constraints
from training_checkpoints import CHECKPOINT_DIR, list_checkpoints


SCHEMA_VERSION = 1
REFERENCE_IDS = ("no_bess", "sadrbc_v13", "oracle")
SAMPLING_FIELDS = {"native_dt_minutes", "control_dt_minutes", "native_steps_per_action"}


def context(parameters: dict[str, Any]) -> dict[str, Any]:
    csv_path = selected_data_path(parameters)
    actual_dt_minutes = float(parameters["dt"]) * 60.0
    policies = []
    for checkpoint in list_checkpoints():
        meta = checkpoint.get("meta", {})
        expected_dt = meta.get("native_dt_minutes")
        incompatible_dt = expected_dt is not None and abs(float(expected_dt) - actual_dt_minutes) > 1e-9
        reason = checkpoint.get("error")
        if checkpoint.get("algo", "").lower() not in {"ppo", "grepo"}:
            reason = reason or f"Unsupported algorithm: {checkpoint.get('algo') or 'unknown'}"
        elif not SAMPLING_FIELDS.issubset(meta):
            reason = reason or "Legacy checkpoint lacks required sampling metadata."
        elif incompatible_dt:
            reason = reason or f"Needs {float(expected_dt):g}-minute data; selected CSV is {actual_dt_minutes:g}-minute."
        policies.append(
            {
                **checkpoint,
                "path": None,
                "runnable": not reason,
                "reason": reason,
                "body_mismatch": _body_mismatch(meta, parameters),
            }
        )
    oracle = oracle_cache.cached_oracle_lp(parameters)
    return {
        "dataset": {
            "filename": csv_path.name,
            "sha256": _file_hash(csv_path),
            "dt_minutes": actual_dt_minutes,
        },
        "shared_bess": _shared_bess(parameters),
        "policies": policies,
        "oracle_ready": _oracle_ready(oracle),
        "oracle_status": (oracle or {}).get("status") or "Exact Oracle is not cached.",
        "history": benchmark_store.list_runs(),
    }


def fingerprint(parameters: dict[str, Any], policy_names: list[str]) -> str:
    checkpoints = _selected_checkpoints(policy_names)
    payload = {
        "schema": SCHEMA_VERSION,
        "dataset_sha256": _file_hash(selected_data_path(parameters)),
        "parameters": {key: parameters[key] for key in sorted(parameters)},
        "policies": [
            {"name": row["name"], "sha256": _file_hash(CHECKPOINT_DIR / row["name"])}
            for row in sorted(checkpoints, key=lambda item: item["name"])
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cached_result(parameters: dict[str, Any], policy_names: list[str]) -> dict[str, Any] | None:
    return benchmark_store.find_exact(fingerprint(parameters, policy_names))


def run_and_save(
    parameters: dict[str, Any],
    policy_names: list[str],
    progress: Callable[[str, int, int, str | None], None],
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    checkpoints = _selected_checkpoints(policy_names)
    oracle = oracle_cache.cached_oracle_lp(parameters)
    if not _oracle_ready(oracle):
        raise ValueError("Exact Oracle is missing. Calculate this battery and dataset in Sizing Demo first.")

    month = dispatch_runner.dataset_to_month(selected_data_path(parameters))
    cfg = dispatch_runner.build_dispatch_config(
        parameters,
        float(parameters.get("battery_capacity_kWh") or 0),
        float(parameters.get("battery_power_limit_kW") or 0),
    )
    total_stages = 3 + len(checkpoints)
    contestants: list[dict[str, Any]] = []

    _check_cancelled(cancelled)
    progress("Running no-BESS baseline", 0, total_stages, "No-BESS")
    no_bess = run_no_bess(month, cfg)
    contestants.append(_rollout_contestant("no_bess", "No-BESS", "reference", no_bess, month, cfg, parameters))

    _check_cancelled(cancelled)
    progress("Running rule-based BESS", 1, total_stages, "SADRBC v13")
    sadrbc = run_sadrbc(month, cfg)
    contestants.append(_rollout_contestant("sadrbc_v13", "SADRBC v13", "rule", sadrbc, month, cfg, parameters))

    for index, checkpoint in enumerate(checkpoints, start=2):
        _check_cancelled(cancelled)
        name = checkpoint["name"]
        progress("Running policy brain", index, total_stages, name)
        agent, algo, meta = dispatch_runner.load_policy(name)
        p_ref = float(meta.get("p_ref_kw") or dispatch_runner._policy_reference_kw(month))
        rollout = dispatch_runner.run_drl_policy(month, cfg, agent, p_ref_kw=p_ref)
        contestant = _rollout_contestant(name, name, "policy", rollout, month, cfg, parameters)
        contestant["algo"] = algo
        contestant["body_mismatch"] = _body_mismatch(meta, parameters)
        contestant["trained_bess"] = {
            "capacity_kwh": meta.get("e_cap_kwh"),
            "power_kw": meta.get("p_rated_kw"),
        }
        contestants.append(contestant)

    _check_cancelled(cancelled)
    progress("Loading Oracle Final Boss", total_stages - 1, total_stages, "Oracle")
    contestants.append(_oracle_contestant(oracle, cfg, parameters))

    no_bess_total = contestants[0]["summary"]["total_operating_cost_vnd"]
    oracle_total = contestants[-1]["summary"]["total_operating_cost_vnd"]
    oracle_days = {day["day_index"]: day for day in contestants[-1]["days"]}
    for contestant in contestants:
        summary = contestant["summary"]
        summary["saving_vs_no_bess_vnd"] = round(no_bess_total - summary["total_operating_cost_vnd"])
        summary["gap_to_oracle_vnd"] = round(summary["total_operating_cost_vnd"] - oracle_total)
        summary["oracle_relation"] = _oracle_relation(summary["total_operating_cost_vnd"], no_bess_total, oracle_total)
        contestant["detectives"] = _detectives(contestant["days"], oracle_days)

    leaderboard = sorted(
        (_leaderboard_row(row) for row in contestants),
        key=lambda row: (row["total_operating_cost_vnd"], row["id"]),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint(parameters, policy_names),
        "snapshot": {
            "dataset": context(parameters)["dataset"],
            "shared_bess": _shared_bess(parameters),
            "parameters": dict(parameters),
            "policies": [
                {
                    "name": row["name"],
                    "sha256": _file_hash(CHECKPOINT_DIR / row["name"]),
                    "body_mismatch": _body_mismatch(row.get("meta", {}), parameters),
                }
                for row in checkpoints
            ],
        },
        "contestants": contestants,
        "leaderboard": leaderboard,
        "champions": _champions(leaderboard),
    }
    _check_cancelled(cancelled)
    progress("Saving exact tournament", total_stages, total_stages, None)
    return benchmark_store.save_result(result)


def _selected_checkpoints(policy_names: list[str]) -> list[dict[str, Any]]:
    if not isinstance(policy_names, list) or not policy_names:
        raise ValueError("Select at least one runnable .pt policy.")
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("A policy was selected more than once.")
    known = {row["name"]: row for row in list_checkpoints()}
    selected = []
    for name in policy_names:
        if Path(str(name)).name != name or name not in known:
            raise ValueError(f"Unknown local policy: {name}")
        row = known[name]
        if row.get("error"):
            raise ValueError(f"{name}: checkpoint is not loadable ({row['error']})")
        if row.get("algo", "").lower() not in {"ppo", "grepo"}:
            raise ValueError(f"{name}: unsupported algorithm {row.get('algo')}")
        if not SAMPLING_FIELDS.issubset(row.get("meta", {})):
            raise ValueError(f"{name}: legacy checkpoint lacks required sampling metadata")
        selected.append(row)
    return selected


def _rollout_contestant(identifier, label, kind, rollout, month, cfg, parameters):
    days = dispatch_runner.policy_result_to_days(month, rollout, cfg, parameters)
    safety_days = _safety_day_indices(rollout, month, cfg)
    _decorate_days(days, rollout["p_bess_days"], cfg.dt, parameters, safety_days)
    constraints = check_hard_constraints(rollout["p_grid_days"], rollout["soc_days"], cfg)
    return {
        "id": identifier,
        "label": label,
        "type": kind,
        "days": days,
        "summary": _summary(days, parameters, constraints),
    }


def _oracle_contestant(oracle, cfg, parameters):
    days = [dict(day) for day in oracle.get("days", [])]
    for day in days:
        throughput = cfg.dt * sum(
            sum(float(value) for value in day.get(key, []))
            for key in ("discharge", "grid_charge", "solar_charge")
        )
        day.setdefault("safety_violation", False)
        day["throughput_kwh"] = round(throughput, 2)
        day.setdefault("total_operating_owner_vnd", round(day.get("bill_with_owner_peak_vnd", 0) + day.get("wear_cost_vnd", 0)))
        day.setdefault("total_operating_prorated_vnd", round(day.get("bill_with_prorated_peak_vnd", 0) + day.get("wear_cost_vnd", 0)))
    return {
        "id": "oracle",
        "label": "Oracle Final Boss",
        "type": "oracle",
        "days": days,
        "summary": _summary(days, parameters, {"zero_export_violation_days": 0, "soc_violation_days": 0}),
    }


def _decorate_days(days, p_bess_days, dt, parameters, safety_days):
    wear_rate = float(parameters.get("battery_wear_cost") or 0.0)
    for index, day in enumerate(days):
        p_bess = np.asarray(p_bess_days[index], dtype=np.float64)
        throughput = float(np.sum(np.abs(p_bess)) * dt)
        wear = throughput * wear_rate
        day["throughput_kwh"] = round(throughput, 2)
        day["wear_cost_vnd"] = round(wear)
        day["total_operating_owner_vnd"] = round(day.get("bill_with_owner_peak_vnd", 0) + wear)
        day["total_operating_prorated_vnd"] = round(day.get("bill_with_prorated_peak_vnd", 0) + wear)
        day["safety_violation"] = day.get("day_index") in safety_days


def _summary(days, parameters, constraints):
    blocks = []
    for month_start in sorted({_month_start_day(day["day_index"]) for day in days}):
        block_days = [day for day in days if _month_start_day(day["day_index"]) == month_start]
        energy = sum(float(day.get("energy_bill_vnd", day.get("energy_cost_vnd", 0))) for day in block_days)
        wear = sum(float(day.get("wear_cost_vnd", 0)) for day in block_days)
        peak = max((max(day.get("rolling_grid", []) or [0]) for day in block_days), default=0.0)
        demand = _demand_charge(parameters, peak)
        blocks.append(
            {
                "start_day": month_start,
                "end_day": block_days[-1]["day_index"] if block_days else month_start,
                "energy_cost_vnd": round(energy),
                "demand_cost_vnd": round(demand),
                "wear_cost_vnd": round(wear),
                "total_operating_cost_vnd": round(energy + demand + wear),
                "peak_kw": round(peak, 2),
            }
        )
    energy = sum(row["energy_cost_vnd"] for row in blocks)
    demand = sum(row["demand_cost_vnd"] for row in blocks)
    wear = sum(row["wear_cost_vnd"] for row in blocks)
    return {
        "energy_cost_vnd": energy,
        "demand_cost_vnd": demand,
        "utility_bill_vnd": energy + demand,
        "wear_cost_vnd": wear,
        "total_operating_cost_vnd": energy + demand + wear,
        "grid_import_kwh": round(sum(float(day.get("grid_kWh", 0)) for day in days), 2),
        "peak_kw": max((row["peak_kw"] for row in blocks), default=0.0),
        "throughput_kwh": round(sum(float(day.get("throughput_kwh", 0)) for day in days), 2),
        "zero_export_violation_days": int(constraints.get("zero_export_violation_days", 0)),
        "soc_violation_days": int(constraints.get("soc_violation_days", 0)),
        "blocks": blocks,
    }


def _leaderboard_row(contestant):
    summary = contestant["summary"]
    return {
        "id": contestant["id"],
        "label": contestant["label"],
        "type": contestant["type"],
        "algo": contestant.get("algo"),
        "body_mismatch": bool(contestant.get("body_mismatch")),
        **{key: value for key, value in summary.items() if key != "blocks"},
    }


def _champions(leaderboard):
    policies = [row for row in leaderboard if row["type"] == "policy"]
    if not policies:
        return {}

    def pick(key, reverse=False):
        return sorted(policies, key=lambda row: ((-row[key]) if reverse else row[key], row["total_operating_cost_vnd"], row["id"]))[0]["id"]

    safest = sorted(
        policies,
        key=lambda row: (
            row["zero_export_violation_days"] + row["soc_violation_days"],
            row["total_operating_cost_vnd"],
            row["id"],
        ),
    )[0]["id"]
    return {
        "cheapest": pick("total_operating_cost_vnd"),
        "lowest_peak": pick("peak_kw"),
        "most_utility_saving": pick("saving_vs_no_bess_vnd", reverse=True),
        "closest_oracle": pick("gap_to_oracle_vnd"),
        "lowest_wear": pick("wear_cost_vnd"),
        "safest": safest,
    }


def _detectives(days, oracle_days):
    if not days:
        return {}
    by_peak = max(days, key=lambda day: max(day.get("rolling_grid", []) or [0]))
    expensive = max(days, key=lambda day: day.get("total_operating_owner_vnd", 0))
    wear = max(days, key=lambda day: day.get("wear_cost_vnd", 0))
    regret = max(
        days,
        key=lambda day: day.get("total_operating_owner_vnd", 0)
        - oracle_days.get(day["day_index"], {}).get("total_operating_owner_vnd", 0),
    )
    safety = [day["day_index"] for day in days if day.get("safety_violation")]
    return {
        "peak_day": by_peak["day_index"],
        "most_expensive_day": expensive["day_index"],
        "largest_oracle_regret_day": regret["day_index"],
        "highest_wear_day": wear["day_index"],
        "safety_violation_days": safety,
    }


def _safety_day_indices(rollout, month, cfg):
    violating = set()
    for index, day in enumerate(month.days):
        grid = np.asarray(rollout["p_grid_days"][index])
        soc = np.asarray(rollout["soc_days"][index])
        if np.any(grid < -1e-6) or np.any(soc < cfg.SOC_min - 1e-4) or np.any(soc > cfg.SOC_max + 1e-4):
            violating.add(day.day_index)
    return violating


def _oracle_relation(total, no_bess_total, oracle_total):
    tolerance = max(1.0, abs(oracle_total) * 1e-9)
    if total < oracle_total - tolerance:
        return "Beats Oracle — investigate accounting"
    if total >= no_bess_total:
        return "Worse than no-BESS"
    return "Between no-BESS and Oracle"


def _body_mismatch(meta, parameters):
    expected_e = meta.get("e_cap_kwh")
    expected_p = meta.get("p_rated_kw")
    actual_e = float(parameters.get("battery_capacity_kWh") or 0)
    actual_p = float(parameters.get("battery_power_limit_kW") or 0)
    return (
        expected_e is not None and abs(float(expected_e) - actual_e) > 1e-9
    ) or (
        expected_p is not None and abs(float(expected_p) - actual_p) > 1e-9
    )


def _shared_bess(parameters):
    return {
        "capacity_kwh": float(parameters.get("battery_capacity_kWh") or 0),
        "power_kw": float(parameters.get("battery_power_limit_kW") or 0),
        "soc_min": float(parameters.get("minimum_soc") or 0),
        "soc_max": float(parameters.get("maximum_soc") or 0),
        "wear_vnd_per_kwh": float(parameters.get("battery_wear_cost") or 0),
    }


def _oracle_ready(oracle):
    return bool(
        oracle
        and oracle.get("available")
        and oracle.get("days")
        and oracle.get("summary", {}).get("solved_day_count") == len(oracle.get("days", []))
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_cancelled(cancelled):
    if cancelled():
        raise BenchmarkCancelled()
