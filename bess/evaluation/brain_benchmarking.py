"""Brain tournament built on the exact same runtime used by Dispatch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from bess.brain.runtime import run_controllers
from bess.core.config import BrainConfig
from bess.evaluation import benchmark_store
from bess.evaluation.benchmark import selected_data_path
from bess.evaluation.oracle import oracle_cache
from bess.evaluation.benchmark_jobs import BenchmarkCancelled
from bess.training.brain3_checkpoints import CHECKPOINT_DIR, list_compatible_checkpoints


def roster(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": "brain1", "display_name": "Brain 1", "algo": "rule"},
        {"id": "brain2", "display_name": "Brain 2", "algo": "schedule"},
        *list_compatible_checkpoints(BrainConfig.from_parameters(parameters).fingerprint()),
    ]


def context(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "controllers": roster(parameters),
        "dataset": selected_data_path(parameters).name,
        "oracle_ready": oracle_cache.selected_csv_has_cache(parameters),
        "oracle_optional": True,
    }


def fingerprint(parameters: dict[str, Any], controller_ids: list[str]) -> str:
    if not isinstance(controller_ids, list) or not controller_ids or not all(
        isinstance(controller_id, str) for controller_id in controller_ids
    ):
        raise ValueError("controllers must be a non-empty list of controller IDs")
    known = {row["id"] for row in roster(parameters) if not row.get("error")}
    unknown = sorted(set(controller_ids) - known)
    if unknown:
        raise ValueError(f"unknown benchmark controller: {', '.join(unknown)}")
    path = selected_data_path(parameters)
    checkpoints = []
    for controller_id in controller_ids:
        if controller_id.startswith("brain3:"):
            checkpoint = CHECKPOINT_DIR / controller_id.split(":", 1)[1]
            checkpoints.append(
                {"name": checkpoint.name, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            )
    payload = {
        "parameters": parameters,
        "controllers": controller_ids,
        "dataset": {"name": path.name, "size": path.stat().st_size, "mtime": path.stat().st_mtime_ns},
        "checkpoints": checkpoints,
        "schema": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def cached_result(parameters: dict[str, Any], controller_ids: list[str]):
    row = benchmark_store.find_exact(fingerprint(parameters, controller_ids))
    return benchmark_store.get_result(row["id"]) if row else None


def run_and_save(
    parameters: dict[str, Any],
    controller_ids: list[str],
    progress: Callable[[str, int, int, str | None], None],
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    if not controller_ids:
        raise ValueError("select at least one brain for the tournament")
    results: dict[str, Any] = {}
    warnings: list[str] = []
    for index, controller_id in enumerate(controller_ids):
        if cancelled():
            raise BenchmarkCancelled()
        progress("Running identical BrainEnv billing periods", index, len(controller_ids), controller_id)
        rows, row_warnings = run_controllers(
            [controller_id], selected_data_path(parameters), parameters, CHECKPOINT_DIR
        )
        results.update(rows)
        warnings.extend(row_warnings)
    leaderboard = sorted(
        (
            {
                "id": controller_id,
                "display_name": result["meta"].get("display_name", controller_id),
                **result["kpi"],
            }
            for controller_id, result in results.items()
        ),
        key=lambda row: row["bess_cost_vnd"],
    )
    progress("Saving frozen tournament", len(controller_ids), len(controller_ids), None)
    result = {
        "fingerprint": fingerprint(parameters, controller_ids),
        "snapshot": {
            "dataset": {"filename": selected_data_path(parameters).name},
            "controllers": controller_ids,
            "parameters": parameters,
        },
        "leaderboard": leaderboard,
        "controllers": results,
        "warnings": warnings,
        "oracle": {
            "ready": oracle_cache.selected_csv_has_cache(parameters),
            "role": "optional theoretical ceiling; never training input or launch gate",
        },
    }
    return benchmark_store.save_result(result)
