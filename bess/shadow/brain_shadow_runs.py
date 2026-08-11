"""Persistent, no-command Brain shadow evaluation."""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Callable

from bess.paths import PROJECT_ROOT
from bess.brain.runtime import (
    BrainDay,
    load_csv_days,
    run_controller,
    split_billing_periods,
)
from bess.core.config import BrainConfig
from bess.core.settings import DEFAULT_PARAMETERS
from bess.evaluation.benchmark import selected_data_path
from bess.integrations import thingsboard_connector
from bess.training.brain3_checkpoints import CHECKPOINT_DIR, list_compatible_checkpoints

STORE_DIR = PROJECT_ROOT / "runs" / "shadow"
DB_PATH = STORE_DIR / "shadow.sqlite"
_LOCK = threading.RLock()


class ShadowRunError(ValueError):
    pass


def _conn() -> sqlite3.Connection:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS shadow_config(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS shadow_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            source TEXT NOT NULL,
            result_json TEXT NOT NULL
        );
        """
    )
    return conn


def _default_config(parameters: dict[str, Any]) -> dict[str, Any]:
    merged = {**DEFAULT_PARAMETERS, **(parameters or {})}
    return {
        "source_kind": "csv",
        "source": merged.get("selected_data_csv"),
        "controllers": ["brain1", "brain2"],
        "parameters": merged,
    }


def get_config(parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = _default_config(parameters or {})
    with _conn() as conn:
        row = conn.execute("SELECT payload FROM shadow_config WHERE id=1").fetchone()
    if not row:
        return fallback
    try:
        saved = json.loads(row["payload"])
    except json.JSONDecodeError:
        return fallback
    return {**fallback, **saved, "parameters": {**fallback["parameters"], **saved.get("parameters", {})}}


def set_config(payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    config = _default_config(parameters)
    updates = dict(payload)
    incoming_parameters = updates.pop("parameters", {})
    if not isinstance(incoming_parameters, dict):
        raise ShadowRunError("shadow parameters must be an object")
    config.update(updates)
    config["parameters"] = {**config["parameters"], **incoming_parameters}
    controllers = config.get("controllers") or []
    if not isinstance(controllers, list) or not controllers or not all(
        isinstance(controller_id, str) for controller_id in controllers
    ):
        raise ShadowRunError("select at least one shadow Brain controller")
    environment_fingerprint = BrainConfig.from_parameters(config["parameters"]).fingerprint()
    known = {
        "brain1",
        "brain2",
        *{row["id"] for row in list_compatible_checkpoints(environment_fingerprint)},
    }
    unknown = sorted(set(controllers) - known)
    if unknown:
        raise ShadowRunError(f"unknown shadow controller: {', '.join(unknown)}")
    if config.get("source_kind") not in {"csv", "thingsboard"}:
        raise ShadowRunError("shadow source_kind must be csv or thingsboard")
    with _LOCK, _conn() as conn:
        history = conn.execute("SELECT COUNT(*) FROM shadow_runs").fetchone()[0]
        existing = conn.execute("SELECT payload FROM shadow_config WHERE id=1").fetchone()
        encoded = json.dumps(config, sort_keys=True)
        if history and existing and existing["payload"] != encoded:
            raise ShadowRunError("shadow history is frozen; reset it before changing configuration")
        conn.execute(
            "INSERT INTO shadow_config(id,payload) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
            (encoded,),
        )
    return config


def _source_days(config: dict[str, Any], start_iso: str | None, end_iso: str | None) -> list[BrainDay]:
    if config["source_kind"] == "thingsboard":
        if not start_iso or not end_iso:
            raise ShadowRunError("ThingsBoard shadow catch-up requires start_date and end_date")
        days, missing, _ = thingsboard_connector.fetch_days(start_iso, end_iso)
        if missing:
            raise ShadowRunError("ThingsBoard returned incomplete days: " + "; ".join(f"{key}: {value}" for key, value in missing.items()))
        return [
            BrainDay(
                day_index=index + 1,
                date_iso=day.date_iso,
                day_type=day.day_type,
                load_kw=tuple(float(value) for value in day.load),
                pv_kw=tuple(float(value) for value in day.pv),
            )
            for index, day in enumerate(days)
        ]
    params = dict(config["parameters"])
    params["selected_data_csv"] = config.get("source") or params.get("selected_data_csv")
    days = load_csv_days(selected_data_path(params))
    if start_iso:
        days = [day for day in days if day.date_iso and day.date_iso >= start_iso]
    if end_iso:
        days = [day for day in days if day.date_iso and day.date_iso <= end_iso]
    return days


def catchup(
    payload: dict[str, Any],
    parameters: dict[str, Any],
    progress: Callable[[str, int, int, str | None], None],
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    config = get_config(parameters)
    source_days = _source_days(config, payload.get("start_date"), payload.get("end_date"))
    if not source_days:
        raise ShadowRunError("shadow source contains no completed days in the requested range")
    applied = {**parameters, **config.get("parameters", {})}
    applied["dt"] = str(24.0 / len(source_days[0].load_kw))
    brain_config = BrainConfig.from_parameters(applied)
    periods = split_billing_periods(source_days, reject_leftover=True)
    results = {}
    warnings = []
    controllers = config["controllers"]
    for index, controller_id in enumerate(controllers):
        if cancelled():
            return {"cancelled": True}
        progress("Replaying measured data without emitting commands", index, len(controllers), source_days[0].date_iso)
        try:
            results[controller_id] = run_controller(controller_id, periods, brain_config, CHECKPOINT_DIR)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{controller_id}: {exc}")
    if not results:
        raise ShadowRunError("no shadow controller completed")
    frozen = {
        "source": config["source_kind"],
        "controllers": results,
        "warnings": warnings,
        "start_date": source_days[0].date_iso,
        "end_date": source_days[-1].date_iso,
    }
    with _LOCK, _conn() as conn:
        cursor = conn.execute(
            "INSERT INTO shadow_runs(source,result_json) VALUES(?,?)",
            (config["source_kind"], json.dumps(frozen)),
        )
        run_id = cursor.lastrowid
    progress("Shadow history saved", len(controllers), len(controllers), source_days[-1].date_iso)
    return {"run_id": run_id, **frozen}


def _latest() -> dict[str, Any] | None:
    with _conn() as conn:
        row = conn.execute("SELECT result_json FROM shadow_runs ORDER BY id DESC LIMIT 1").fetchone()
    return json.loads(row["result_json"]) if row else None


def list_days(month: str | None = None) -> list[dict[str, Any]]:
    latest = _latest()
    if not latest:
        return []
    grouped: dict[tuple[int, str | None], dict[str, Any]] = {}
    for controller_id, result in latest["controllers"].items():
        for row in result["trace"]:
            date_iso = row.get("date_iso")
            if month and (not date_iso or not date_iso.startswith(month)):
                continue
            key = (int(row["day_index"]), date_iso)
            day = grouped.setdefault(
                key,
                {"day_index": key[0], "date": date_iso, "controllers": {}},
            )
            day["controllers"].setdefault(controller_id, []).append(row)
    return list(grouped.values())


def monthly_report() -> list[dict[str, Any]]:
    days = list_days()
    months: dict[str, dict[str, Any]] = {}
    for day in days:
        key = (day.get("date") or "undated")[:7]
        bucket = months.setdefault(key, {"month": key, "controllers": {}})
        for controller_id, trace in day["controllers"].items():
            row = bucket["controllers"].setdefault(
                controller_id,
                {"energy_cost_vnd": 0.0, "demand_cost_vnd": 0.0, "wear_cost_vnd": 0.0, "savings_vnd": 0.0},
            )
            row["energy_cost_vnd"] += sum(item["energy_cost_vnd"] for item in trace)
            row["demand_cost_vnd"] += sum(item["demand_cost_vnd"] for item in trace)
            row["wear_cost_vnd"] += sum(item["wear_cost_vnd"] for item in trace)
            row["savings_vnd"] += sum(item["reward_vnd"] for item in trace)
    return list(months.values())


def reset_history() -> None:
    with _LOCK, _conn() as conn:
        conn.execute("DELETE FROM shadow_runs")
        conn.execute("DELETE FROM shadow_config")
