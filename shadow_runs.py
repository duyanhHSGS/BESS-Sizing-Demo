"""Restart-safe shadow evaluation for Sizing Demo.

Shadow runs never emit battery commands.  They replay measured CSV rows through
No-BESS, SADRBC, and a selected policy, then persist only audit/KPI results in
SQLite.  Controller state is reconstructed deterministically from the source
prefix on every catch-up, so a Flask restart cannot silently reset the science.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np

from baselines import run_drl_policy, run_sadrbc, validate_dispatch_sampling
from benchmark import list_data_csvs, selected_data_path
from common import check_hard_constraints, rolling_pmax_day, tariff_vector_day
from dispatch_runner import (
    build_dispatch_config,
    dataset_to_month,
    load_policy,
    prepare_policy_forecast,
)
from settings import PPO_GAMMA
from training_checkpoints import list_checkpoints


BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "shadow"
DB_PATH = STORE_DIR / "shadow.sqlite"
_RUN_LOCK = threading.Lock()


class ShadowRunError(ValueError):
    pass


def _conn() -> sqlite3.Connection:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS shadow_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shadow_days (
            date TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            day_type TEXT NOT NULL,
            load_kwh REAL NOT NULL,
            pv_kwh REAL NOT NULL,
            nobess_energy_vnd REAL NOT NULL,
            nobess_day_peak_kw REAL NOT NULL,
            nobess_mtd_peak_kw REAL NOT NULL,
            sadrbc_energy_vnd REAL NOT NULL,
            sadrbc_day_peak_kw REAL NOT NULL,
            sadrbc_mtd_peak_kw REAL NOT NULL,
            sadrbc_soc_end_pct REAL,
            policy_energy_vnd REAL NOT NULL,
            policy_day_peak_kw REAL NOT NULL,
            policy_mtd_peak_kw REAL NOT NULL,
            policy_soc_end_pct REAL,
            policy_violations INTEGER NOT NULL,
            checkpoint_id TEXT NOT NULL
        );
        """
    )
    return conn


def _default_config(parameters: dict[str, Any]) -> dict[str, Any]:
    checkpoints = [row for row in list_checkpoints() if not row.get("error")]
    return {
        "source": str(parameters.get("selected_data_csv") or ""),
        "policy": checkpoints[0]["name"] if checkpoints else "",
        "e_cap_kwh": _float(parameters.get("battery_capacity_kWh"), 0.0),
        "p_rated_kw": _float(parameters.get("battery_power_limit_kW"), 0.0),
        "parameters": dict(parameters),
    }


def _float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def get_config(parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    with _conn() as conn:
        row = conn.execute("SELECT payload FROM shadow_config WHERE id = 1").fetchone()
    if row:
        try:
            payload = json.loads(row["payload"])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    return _default_config(parameters or {})


def set_config(payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    if not _RUN_LOCK.acquire(blocking=False):
        raise ShadowRunError("Stop the running shadow catch-up before changing configuration.")
    try:
        return _set_config_unlocked(payload, parameters)
    finally:
        _RUN_LOCK.release()


def _set_config_unlocked(payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(payload.get("source") or "")).name
    if source != payload.get("source") or source not in set(list_data_csvs()):
        raise ShadowRunError("Choose a CSV inside Sizing_Demo/data.")
    policy = str(payload.get("policy") or "")
    known = {row["name"] for row in list_checkpoints() if not row.get("error")}
    if policy not in known:
        raise ShadowRunError("Choose a loadable local policy checkpoint.")
    e_cap = _float(payload.get("e_cap_kwh"), 0.0)
    p_rated = _float(payload.get("p_rated_kw"), 0.0)
    if e_cap <= 0 or p_rated <= 0:
        raise ShadowRunError("Shadow battery capacity and power must be positive.")

    snapshot = dict(parameters)
    snapshot["selected_data_csv"] = source
    snapshot["battery_capacity_kWh"] = str(e_cap)
    snapshot["battery_power_limit_kW"] = str(p_rated)
    config = {
        "source": source,
        "policy": policy,
        "e_cap_kwh": e_cap,
        "p_rated_kw": p_rated,
        "parameters": snapshot,
    }
    with _conn() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM shadow_days").fetchone()[0]
        current = conn.execute("SELECT payload FROM shadow_config WHERE id = 1").fetchone()
        if existing and current and json.loads(current["payload"]) != config:
            raise ShadowRunError(
                "Shadow history already exists. Reset it before changing source, policy, battery, or tariff."
            )
        conn.execute(
            "INSERT OR REPLACE INTO shadow_config(id, payload) VALUES (1, ?)",
            (json.dumps(config, separators=(",", ":")),),
        )
    return config


def reset_history() -> None:
    if not _RUN_LOCK.acquire(blocking=False):
        raise ShadowRunError("Stop the running shadow catch-up before resetting history.")
    try:
        with _conn() as conn:
            conn.execute("DELETE FROM shadow_days")
    finally:
        _RUN_LOCK.release()


def _policy_reference_kw(month) -> float:
    peak = max(
        (float(np.max(np.maximum(0.0, day.load - day.pv))) for day in month.days),
        default=500.0,
    )
    return max(500.0, float(np.ceil(peak / 500.0) * 500.0))


def _build_rollouts(config: dict[str, Any], progress: Callable) -> tuple:
    parameters = dict(config["parameters"])
    month = dataset_to_month(selected_data_path(parameters))
    if not month.days:
        raise ShadowRunError("The configured shadow CSV has no days.")
    cfg = build_dispatch_config(parameters, config["e_cap_kwh"], config["p_rated_kw"])

    progress("Loading policy brain", 0, len(month.days), config["policy"])
    agent, algo, meta = load_policy(config["policy"])
    control_minutes = validate_dispatch_sampling(meta, cfg.dt * 60.0)
    p_ref = float(meta.get("p_ref_kw") or _policy_reference_kw(month))
    prepare_policy_forecast(config["policy"], agent, meta, month, p_ref)
    policy = run_drl_policy(month, cfg, agent, p_ref_kw=p_ref)
    progress("Running shadow SADRBC", 0, len(month.days), "SADRBC v13")
    sadrbc = run_sadrbc(month, cfg)
    return month, cfg, policy, sadrbc, algo, control_minutes


def catchup(
    start_iso: str | None,
    end_iso: str | None,
    progress: Callable,
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    if not _RUN_LOCK.acquire(blocking=False):
        raise ShadowRunError("A shadow catch-up is already running.")
    try:
        config = get_config()
        if not config.get("policy") or not config.get("source"):
            raise ShadowRunError("Save a valid Shadow configuration first.")
        month, cfg, policy, sadrbc, algo, control_minutes = _build_rollouts(config, progress)
        by_date = {day.date_iso: (index, day) for index, day in enumerate(month.days)}
        source_dates = sorted(date.fromisoformat(value) for value in by_date)
        with _conn() as conn:
            saved_dates = [
                date.fromisoformat(row[0])
                for row in conn.execute("SELECT date FROM shadow_days ORDER BY date")
            ]
        start = date.fromisoformat(start_iso) if start_iso else (
            max(saved_dates) + timedelta(days=1) if saved_dates else source_dates[0]
        )
        end = date.fromisoformat(end_iso) if end_iso else source_dates[-1]
        if start > end:
            if not start_iso and not end_iso:
                return {
                    "id": "shadow",
                    "processed_ok": 0,
                    "skipped": 0,
                    "already_done": 0,
                    "range": [start.isoformat(), end.isoformat()],
                    "algorithm": algo,
                    "control_dt_minutes": control_minutes,
                }
            raise ShadowRunError("Shadow start date must not be after the end date.")
        total = (end - start).days + 1
        processed = skipped = already_done = 0
        cursor = start
        while cursor <= end:
            if cancelled():
                break
            iso = cursor.isoformat()
            with _conn() as conn:
                exists = conn.execute("SELECT 1 FROM shadow_days WHERE date = ?", (iso,)).fetchone()
            if exists:
                already_done += 1
            elif iso not in by_date:
                _save_skipped(iso, config["policy"])
                skipped += 1
            else:
                index, day = by_date[iso]
                _save_day(index, day, month, cfg, policy, sadrbc, config["policy"])
                processed += 1
            complete = (cursor - start).days + 1
            progress("Shadow catch-up", complete, total, iso)
            cursor += timedelta(days=1)
        _rebuild_all_mtd()
        return {
            "id": "shadow",
            "processed_ok": processed,
            "skipped": skipped,
            "already_done": already_done,
            "range": [start.isoformat(), end.isoformat()],
            "algorithm": algo,
            "control_dt_minutes": control_minutes,
        }
    finally:
        _RUN_LOCK.release()


def _save_day(index, day, month, cfg, policy, sadrbc, checkpoint_id: str) -> None:
    no_bess_grid = np.maximum(0.0, day.load - day.pv)
    sadrbc_grid = np.maximum(0.0, np.asarray(sadrbc["p_grid_days"][index], dtype=float))
    policy_grid = np.maximum(0.0, np.asarray(policy["p_grid_days"][index], dtype=float))
    tariff = tariff_vector_day(cfg, day)
    policy_soc = np.asarray(policy["soc_days"][index], dtype=float)
    sadrbc_soc = np.asarray(sadrbc["soc_days"][index], dtype=float)
    violations = check_hard_constraints([policy_grid], [policy_soc], cfg)
    row = (
        day.date_iso,
        "OK",
        day.day_type,
        float(np.sum(day.load) * cfg.dt),
        float(np.sum(day.pv) * cfg.dt),
        float(np.sum(no_bess_grid * tariff) * cfg.dt),
        rolling_pmax_day(no_bess_grid, cfg.dt),
        0.0,
        float(np.sum(sadrbc_grid * tariff) * cfg.dt),
        rolling_pmax_day(sadrbc_grid, cfg.dt),
        0.0,
        float(sadrbc_soc[-1] * 100.0),
        float(np.sum(policy_grid * tariff) * cfg.dt),
        rolling_pmax_day(policy_grid, cfg.dt),
        0.0,
        float(policy_soc[-1] * 100.0),
        int(sum(violations.values())),
        checkpoint_id,
    )
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO shadow_days VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )


def _save_skipped(iso: str, checkpoint_id: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO shadow_days VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (iso, "SKIPPED_MISSING_DATA", "?", 0, 0, 0, 0, 0, 0, 0, 0, None, 0, 0, 0, None, 0, checkpoint_id),
        )


def _rebuild_all_mtd() -> None:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT date,status,nobess_day_peak_kw,sadrbc_day_peak_kw,policy_day_peak_kw "
            "FROM shadow_days ORDER BY date"
        ).fetchall()
        month = None
        peaks = [0.0, 0.0, 0.0]
        for row in rows:
            row_month = row["date"][:7]
            if row_month != month:
                month = row_month
                peaks = [0.0, 0.0, 0.0]
            if row["status"] == "OK":
                peaks = [max(peaks[i], float(row[i + 2] or 0.0)) for i in range(3)]
            conn.execute(
                "UPDATE shadow_days SET nobess_mtd_peak_kw=?,sadrbc_mtd_peak_kw=?,policy_mtd_peak_kw=? WHERE date=?",
                (*peaks, row["date"]),
            )


def list_days(month: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM shadow_days"
    args: tuple = ()
    if month:
        query += " WHERE date LIKE ?"
        args = (month + "%",)
    query += " ORDER BY date"
    with _conn() as conn:
        return [dict(row) for row in conn.execute(query, args).fetchall()]


def monthly_report() -> list[dict[str, Any]]:
    config = get_config()
    parameters = config.get("parameters", {})
    cfg = build_dispatch_config(
        parameters,
        _float(config.get("e_cap_kwh"), 1.0),
        _float(config.get("p_rated_kw"), 1.0),
    )
    output = []
    with _conn() as conn:
        months = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT substr(date,1,7) FROM shadow_days ORDER BY 1"
            ).fetchall()
        ]
        for month in months:
            rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM shadow_days WHERE date LIKE ? ORDER BY date",
                    (month + "%",),
                ).fetchall()
            ]
            ok = [row for row in rows if row["status"] == "OK"]
            if not ok:
                continue
            record = {
                "month": month,
                "n_days_ok": len(ok),
                "n_days_skipped": len(rows) - len(ok),
            }
            for name, prefix in (("nobess", "nobess"), ("sadrbc", "sadrbc"), ("policy", "policy")):
                energy = sum(row[f"{prefix}_energy_vnd"] for row in ok)
                peak = max(row[f"{prefix}_day_peak_kw"] for row in ok)
                record[f"{name}_energy_vnd"] = round(energy)
                record[f"{name}_peak_kw"] = round(peak, 2)
                record[f"{name}_bill_vnd"] = round(energy + cfg.T_cap * peak)
            record["policy_vs_nobess_vnd"] = record["nobess_bill_vnd"] - record["policy_bill_vnd"]
            record["policy_vs_sadrbc_vnd"] = record["sadrbc_bill_vnd"] - record["policy_bill_vnd"]
            record["policy_violations"] = sum(row["policy_violations"] for row in ok)
            output.append(record)
    return output
