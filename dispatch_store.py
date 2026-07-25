from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / "runs"
DB_PATH = RUN_DIR / "runs.sqlite"
TRACE_DIR = RUN_DIR / "traces"


class DispatchStoreError(ValueError):
    pass


def _inside_base(path: Path, base: Path | None = None) -> Path:
    base = base or BASE_DIR
    resolved = path.resolve()
    base_resolved = base.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise DispatchStoreError(f"path escapes Sizing_Demo: {path}")
    return resolved


def _paths(run_dir: Path | None = None) -> tuple[Path, Path]:
    run_dir = run_dir or RUN_DIR
    root = _inside_base(run_dir)
    return root / "runs.sqlite", root / "traces"


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created REAL NOT NULL,
            config TEXT NOT NULL,
            kpi TEXT NOT NULL
        )
        """
    )
    return conn


def save_run(
    name: str,
    config: dict[str, Any],
    results: dict[str, Any],
    run_dir: Path | None = None,
) -> str:
    db_path, trace_dir = _paths(run_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    traces = {}
    kpi = {}
    for policy_name, result in results.items():
        traces[policy_name] = result.get("days", [])
        kpi[policy_name] = result.get("kpi", {})

    trace_path = _inside_base(trace_dir / f"{run_id}.json")
    trace_path.write_text(json.dumps(traces), encoding="utf-8")
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO runs(id, name, created, config, kpi) VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                name,
                time.time(),
                json.dumps(config),
                json.dumps(kpi),
            ),
        )
    return run_id


def list_runs(run_dir: Path | None = None) -> list[dict[str, Any]]:
    db_path, _ = _paths(run_dir)
    if not db_path.exists():
        return []
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, created, config, kpi FROM runs ORDER BY created DESC"
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_run(run_id: str, run_dir: Path | None = None) -> dict[str, Any] | None:
    db_path, _ = _paths(run_dir)
    if not db_path.exists():
        return None
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT id, name, created, config, kpi FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def latest_run_for_policy(policy_name: str, run_dir: Path | None = None) -> dict[str, Any] | None:
    for run in list_runs(run_dir):
        policies = run.get("config", {}).get("policies", [])
        if policy_name in policies or run.get("config", {}).get("policy") == policy_name:
            return run
    return None


def latest_runs_by_policy(run_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    latest = {}
    for run in list_runs(run_dir):
        for policy_name in run.get("config", {}).get("policies", []):
            latest.setdefault(policy_name, run)
        if run.get("config", {}).get("policy"):
            latest.setdefault(run["config"]["policy"], run)
    return latest


def get_traces(run_id: str, run_dir: Path | None = None) -> dict[str, Any] | None:
    _, trace_dir = _paths(run_dir)
    trace_path = _inside_base(trace_dir / f"{run_id}.json")
    try:
        raw = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "created": row["created"],
        "config": _json_object(row["config"]),
        "kpi": _json_object(row["kpi"]),
    }


def _json_object(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
