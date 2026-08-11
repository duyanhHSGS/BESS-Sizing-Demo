from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from bess.paths import PROJECT_ROOT
from typing import Any


BASE_DIR = PROJECT_ROOT
STORE_DIR = BASE_DIR / "runs" / "benchmarking"
DB_PATH = STORE_DIR / "runs.sqlite"
RESULT_DIR = STORE_DIR / "results"


def _conn() -> sqlite3.Connection:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id TEXT PRIMARY KEY,
            created REAL NOT NULL,
            fingerprint TEXT NOT NULL,
            dataset TEXT NOT NULL,
            roster TEXT NOT NULL,
            summary TEXT NOT NULL,
            result_file TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_benchmark_fingerprint "
        "ON benchmark_runs(fingerprint, created DESC)"
    )
    return conn


def save_result(result: dict[str, Any]) -> dict[str, Any]:
    run_id = uuid.uuid4().hex
    created = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = {**result, "id": run_id, "created": created}
    final_path = RESULT_DIR / f"{run_id}.json"
    temporary_path = RESULT_DIR / f".{run_id}.tmp"
    temporary_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    temporary_path.replace(final_path)

    snapshot = result.get("snapshot", {})
    roster = snapshot.get("controllers", [])
    leaderboard = result.get("leaderboard", [])
    compact = {
        "controller_count": len(leaderboard),
        "best_controller": leaderboard[0].get("id") if leaderboard else None,
    }
    with _conn() as conn:
        conn.execute(
            "INSERT INTO benchmark_runs(id, created, fingerprint, dataset, roster, summary, result_file) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                created,
                result["fingerprint"],
                str(snapshot.get("dataset", {}).get("filename", "")),
                json.dumps(roster),
                json.dumps(compact),
                final_path.name,
            ),
        )
    return result


def find_exact(fingerprint: str) -> dict[str, Any] | None:
    if not DB_PATH.exists():
        return None
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM benchmark_runs WHERE fingerprint = ? ORDER BY created DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()
    return _row(row) if row else None


def list_runs(limit: int = 30) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM benchmark_runs ORDER BY created DESC LIMIT ?",
            (max(1, min(int(limit), 100)),),
        ).fetchall()
    return [_row(row) for row in rows]


def get_result(run_id: str) -> dict[str, Any] | None:
    if not run_id or any(character not in "0123456789abcdef" for character in run_id.lower()):
        return None
    path = RESULT_DIR / f"{run_id}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "created": row["created"],
        "fingerprint": row["fingerprint"],
        "dataset": row["dataset"],
        "roster": _json(row["roster"], []),
        "summary": _json(row["summary"], {}),
    }


def _json(raw: str, fallback):
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return fallback
