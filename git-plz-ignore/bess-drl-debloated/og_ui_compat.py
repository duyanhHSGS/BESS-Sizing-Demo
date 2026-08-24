"""Compatibility layer for the original Sizing Demo single-file web UI.

The HTML in ``templates/index.html`` is the original UI source.  This module
supplies its Jinja context and maps the old browser API contract onto the
Mongo-free/file-backed debloated runner.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from _core import core_src
from flask import Response, jsonify, render_template, request

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
UPLOADS_DIR = HERE / "uploads"
DATA_DIR = HERE / "data"
STATE_DIR = HERE / "state"
UI_SETTINGS_PATH = STATE_DIR / "ui_settings.json"
CONFIG_EXAMPLE = HERE / "config.example.json"
TRAINING_CONFIG_DIR = STATE_DIR / "training_configs"
ORACLE_CACHE_DIR = STATE_DIR / "oracle_cache"
DISPATCH_DIR = STATE_DIR / "dispatch_runs"
LIVE_DIR = STATE_DIR / "live_runs"
SHADOW_DIR = STATE_DIR / "shadow"
WEATHER_DIR = STATE_DIR / "weather"
BENCH_DIR = STATE_DIR / "benchmarking"

_REQUIRED_DATA_COLUMNS = {"date_iso", "step", "day_type", "P_load_kW", "P_pv_kW"}
_BENCH_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_TRAINING_CORE: dict[str, Any] | None = None
_AUTO_THREADS: dict[str, threading.Event] = {}
_AUTO_LOCK = threading.RLock()


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    path: Path
    source: str
    n_days: int
    status: str
    res_min: float
    start_date: str | None
    end_date: str | None

    def api_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "n_days": self.n_days,
            "status": self.status,
            "res_min": self.res_min,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


def ensure_og_dirs() -> None:
    for path in (
        RESULTS_DIR,
        UPLOADS_DIR,
        DATA_DIR,
        STATE_DIR,
        TRAINING_CONFIG_DIR,
        ORACLE_CACHE_DIR,
        DISPATCH_DIR,
        LIVE_DIR,
        SHADOW_DIR,
        WEATHER_DIR,
        BENCH_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _json_load(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _json_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def _settings() -> dict[str, Any]:
    return _json_load(UI_SETTINGS_PATH, {})


def _base_config() -> dict[str, Any]:
    return _json_load(CONFIG_EXAMPLE, {})


def _dataset_roots() -> list[tuple[str, Path]]:
    legacy = HERE.parent / "bess-drl" / "var" / "lib" / "bess-drl" / "datasets"
    return [("data", DATA_DIR), ("uploads", UPLOADS_DIR), ("legacy", legacy)]


def _inspect_dataset(path: Path, root_tag: str) -> DatasetInfo | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not _REQUIRED_DATA_COLUMNS.issubset(fields):
                return None
            dates: set[str] = set()
            max_step = -1
            rows = 0
            for row in reader:
                date_iso = str(row.get("date_iso", "")).strip()
                if date_iso:
                    dates.add(date_iso)
                try:
                    max_step = max(max_step, int(row.get("step", -1)))
                except (TypeError, ValueError):
                    pass
                rows += 1
    except OSError:
        return None
    steps = max_step + 1 if max_step >= 0 else 0
    res_min = 1440.0 / steps if steps else 0.0
    ordered = sorted(dates)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return DatasetInfo(
        id=f"{root_tag}:{digest}",
        path=path.resolve(),
        source=path.name,
        n_days=len(dates),
        status="ready" if rows and dates and steps else "invalid",
        res_min=res_min,
        start_date=ordered[0] if ordered else None,
        end_date=ordered[-1] if ordered else None,
    )


def discover_datasets() -> list[DatasetInfo]:
    ensure_og_dirs()
    found: list[DatasetInfo] = []
    seen: set[Path] = set()
    seen_names: set[str] = set()
    for root_tag, root in _dataset_roots():
        if not root.is_dir():
            continue
        pattern = "*.csv" if root_tag != "legacy" else "**/*.csv"
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            lowered = {part.lower() for part in path.parts}
            if "results" in lowered or path.name.startswith("training_curve_"):
                continue
            resolved = path.resolve()
            if resolved in seen or path.name in seen_names:
                continue
            info = _inspect_dataset(resolved, root_tag)
            if info is not None:
                found.append(info)
                seen.add(resolved)
                seen_names.add(path.name)
    found.sort(key=lambda item: (item.source.lower(), str(item.path)))
    return found


def _dataset_by_id(dataset_id: str | None) -> DatasetInfo | None:
    datasets = discover_datasets()
    if not datasets:
        return None
    if dataset_id:
        for item in datasets:
            if item.id == dataset_id or item.source == dataset_id:
                return item
    selected = str(_settings().get("selected_data_csv", ""))
    for item in datasets:
        if item.source == selected:
            return item
    return datasets[0]


def _float_setting(settings: dict[str, Any], name: str, default: float) -> float:
    try:
        return float(settings.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _bool_setting(settings: dict[str, Any], name: str, default: bool) -> bool:
    value = settings.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "checked"}


def _effective_config(
    settings: dict[str, Any],
    *,
    e_cap_kwh: float | None = None,
    p_rated_kw: float | None = None,
    wear: float | None = None,
) -> dict[str, Any]:
    cfg = _base_config()
    bess = cfg.setdefault("bess", {})
    tariff = cfg.setdefault("tariff", {})
    economics = cfg.setdefault("economics", {})
    bess["eCapKwh"] = e_cap_kwh if e_cap_kwh is not None else _float_setting(
        settings, "battery_capacity_kWh", bess.get("eCapKwh", 1250.0)
    )
    bess["pRatedKw"] = p_rated_kw if p_rated_kw is not None else _float_setting(
        settings, "battery_power_limit_kW", bess.get("pRatedKw", 450.0)
    )
    bess["etaCh"] = _float_setting(settings, "charge_efficiency", bess.get("etaCh", 0.9))
    bess["etaDis"] = _float_setting(settings, "discharge_efficiency", bess.get("etaDis", 0.9))
    bess["socMin"] = _float_setting(settings, "minimum_soc", bess.get("socMin", 0.2))
    bess["socMax"] = _float_setting(settings, "maximum_soc", bess.get("socMax", 0.9))
    economics["degradationCostPerKwhDischarged"] = wear if wear is not None else _float_setting(
        settings,
        "battery_wear_cost",
        economics.get("degradationCostPerKwhDischarged", 500.0),
    )
    tariff["billingMode"] = str(settings.get("billing_mode", tariff.get("billingMode", "2tc")))
    tariff["pricePeak"] = _float_setting(settings, "billing_expensive", tariff.get("pricePeak", 2251.0))
    tariff["priceMid"] = _float_setting(settings, "billing_normal", tariff.get("priceMid", 1332.0))
    tariff["priceOff"] = _float_setting(settings, "billing_cheap", tariff.get("priceOff", 904.0))
    tariff["tCap"] = _float_setting(settings, "billing_peak_penalty", tariff.get("tCap", 285000.0))
    tariff["peakWindows"] = str(settings.get("billing_windows_expensive", tariff.get("peakWindows", "17:30-22:30")))
    tariff["offWindows"] = str(settings.get("billing_windows_cheap", tariff.get("offWindows", "00:00-06:00")))
    tariff["sundayNoPeak"] = _bool_setting(settings, "billing_sunday", tariff.get("sundayNoPeak", True))
    cfg.setdefault("meta", {})["resolvedAt"] = "debloated-ui"
    cfg["meta"]["effectiveDate"] = datetime.now(timezone.utc).date().isoformat()
    digest_source = json.dumps({"bess": bess, "tariff": tariff, "economics": economics}, sort_keys=True)
    cfg["meta"]["configHash"] = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
    return cfg


def _write_config(cfg: dict[str, Any], prefix: str) -> Path:
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    path = TRAINING_CONFIG_DIR / f"{prefix}_{digest}.json"
    if not path.exists():
        _json_save(path, cfg)
    return path


def _parse_clock(value: str) -> int | None:
    try:
        hour_text, minute_text = value.strip().split(":", 1)
        hour, minute = int(hour_text), int(minute_text)
    except (ValueError, AttributeError):
        return None
    if hour == 24 and minute == 0:
        return 1440
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _window_contains(minute: float, raw: str) -> bool:
    for part in str(raw or "").split(","):
        if "-" not in part:
            continue
        start_text, end_text = part.split("-", 1)
        start, end = _parse_clock(start_text), _parse_clock(end_text)
        if start is None or end is None or start == end:
            continue
        if start < end and start <= minute < end:
            return True
        if start > end and (minute >= start or minute < end):
            return True
    return False


def _price_for_step(step: int, count: int, date_iso: str, cfg: dict[str, Any]) -> float:
    tariff = cfg["tariff"]
    minute = (step + 0.5) * 1440.0 / max(count, 1)
    if _window_contains(minute, tariff.get("offWindows", "")):
        return float(tariff.get("priceOff", 0.0))
    sunday = False
    try:
        sunday = date_cls.fromisoformat(date_iso).weekday() == 6
    except ValueError:
        pass
    if not (sunday and tariff.get("sundayNoPeak", False)) and _window_contains(
        minute, tariff.get("peakWindows", "")
    ):
        return float(tariff.get("pricePeak", 0.0))
    return float(tariff.get("priceMid", 0.0))


def _rolling_average(values: list[float], slots: int) -> list[float]:
    slots = max(1, slots)
    out: list[float] = []
    running = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > slots:
            running -= queue.pop(0)
        out.append(running / len(queue))
    return out


def build_benchmark(dataset: DatasetInfo | None, cfg: dict[str, Any]) -> dict[str, Any]:
    empty_summary = {
        "battery_cost_vnd": 0.0,
        "day_count": 0,
        "total_load_kWh": 0.0,
        "total_pv_kWh": 0.0,
        "total_grid_kWh": 0.0,
        "total_bill_vnd": 0.0,
        "peak_grid_kW": 0.0,
        "peak_day_index": 0,
    }
    if dataset is None:
        return {"summary": empty_summary, "days": []}
    key = (
        str(dataset.path),
        dataset.path.stat().st_mtime_ns,
        cfg["meta"]["configHash"],
    )
    cached = _BENCH_CACHE.get(key)
    if cached is not None:
        return cached

    grouped: dict[str, dict[str, Any]] = {}
    try:
        with dataset.path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                date_iso = row["date_iso"]
                entry = grouped.setdefault(
                    date_iso,
                    {"day_type": row.get("day_type", "unknown"), "rows": {}},
                )
                entry["rows"][int(row["step"])] = (
                    float(row["P_load_kW"]),
                    float(row["P_pv_kW"]),
                )
    except (OSError, KeyError, ValueError):
        return {"summary": empty_summary, "days": []}

    raw_days: list[dict[str, Any]] = []
    month_groups: dict[str, list[dict[str, Any]]] = {}
    for day_index, (date_iso, entry) in enumerate(sorted(grouped.items()), start=1):
        rows = entry["rows"]
        count = max(rows, default=-1) + 1
        if count <= 0:
            continue
        load = [rows.get(index, (0.0, 0.0))[0] for index in range(count)]
        pv = [rows.get(index, (0.0, 0.0))[1] for index in range(count)]
        grid = [max(0.0, l - p) for l, p in zip(load, pv, strict=True)]
        dt_hours = 24.0 / count
        demand_slots = max(1, round(0.5 / dt_hours))
        rolling = _rolling_average(grid, demand_slots)
        fixed_blocks = [
            sum(grid[index:index + demand_slots]) / len(grid[index:index + demand_slots])
            for index in range(0, len(grid), demand_slots)
            if len(grid[index:index + demand_slots]) == demand_slots
        ]
        energy_bill = sum(
            power * _price_for_step(index, count, date_iso, cfg) * dt_hours
            for index, power in enumerate(grid)
        )
        day = {
            "day_index": day_index,
            "date_iso": date_iso,
            "day_type": entry["day_type"],
            "load": load,
            "pv": pv,
            "grid": grid,
            "rolling_grid": rolling,
            "load_kWh": sum(load) * dt_hours,
            "pv_kWh": sum(pv) * dt_hours,
            "grid_kWh": sum(grid) * dt_hours,
            "surplus_kWh": sum(max(0.0, p - l) for l, p in zip(load, pv, strict=True)) * dt_hours,
            # Billing truth follows the current DRL contract: aligned,
            # non-overlapping 30-minute blocks. ``rolling_grid`` is retained
            # only because the original UI charts that diagnostic series.
            "peak_grid_kW": max(fixed_blocks, default=0.0),
            "energy_bill_vnd": energy_bill,
        }
        raw_days.append(day)
        month_groups.setdefault(date_iso[:7], []).append(day)

    t_cap = 0.0 if cfg["tariff"].get("billingMode") == "tou" else float(cfg["tariff"].get("tCap", 0.0))
    total_bill = 0.0
    peak_global = 0.0
    peak_day_index = 0
    for month_days in month_groups.values():
        owner = max(month_days, key=lambda item: item["peak_grid_kW"])
        month_peak = owner["peak_grid_kW"]
        demand = month_peak * t_cap
        peak_global = max(peak_global, month_peak)
        if month_peak >= peak_global:
            peak_day_index = owner["day_index"]
        total_bill += sum(item["energy_bill_vnd"] for item in month_days) + demand
        for item in month_days:
            item["month_peak"] = {"value_kW": month_peak, "day_index": owner["day_index"]}
            owner_charge = demand if item is owner else 0.0
            prorated = demand / max(1, len(month_days))
            item["peak_bill_owner_vnd"] = owner_charge
            item["peak_bill_prorated_vnd"] = prorated
            item["bill_with_owner_peak_vnd"] = item["energy_bill_vnd"] + owner_charge
            item["bill_with_prorated_peak_vnd"] = item["energy_bill_vnd"] + prorated

    summary = {
        "battery_cost_vnd": 0.0,
        "day_count": len(raw_days),
        "total_load_kWh": sum(day["load_kWh"] for day in raw_days),
        "total_pv_kWh": sum(day["pv_kWh"] for day in raw_days),
        "total_grid_kWh": sum(day["grid_kWh"] for day in raw_days),
        "total_bill_vnd": total_bill,
        "peak_grid_kW": peak_global,
        "peak_day_index": peak_day_index,
    }
    result = {"summary": summary, "days": raw_days}
    if len(_BENCH_CACHE) > 4:
        _BENCH_CACHE.clear()
    _BENCH_CACHE[key] = result
    return result


def _sample_batteries() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for capacity in (250, 500, 750, 1000, 1250):
        for ratio in (0.35, 0.50, 0.70):
            power = round(capacity * ratio, 1)
            result.append(
                {
                    "label": f"{capacity} kWh / {power:g} kW ({ratio:.2f}C)",
                    "battery_capacity_kWh": float(capacity),
                    "battery_power_limit_kW": power,
                }
            )
    return result


def _blank_oracle(seer_factor: float, status: str = "Oracle not calculated yet.") -> dict[str, Any]:
    return {
        "status": status,
        "summary": {
            "solved_day_count": 0,
            "total_grid_kWh": 0.0,
            "total_bill_vnd": 0.0,
            "peak_grid_kW": 0.0,
            "seer_factor": seer_factor,
        },
        "days": [],
        "cache": {"hit": False},
    }


def build_index_context(
    *,
    should_calculate: bool = False,
    force_calculate: bool = False,
    saved: bool = False,
) -> dict[str, Any]:
    settings = _settings()
    cfg = _effective_config(settings)
    datasets = discover_datasets()
    selected = _dataset_by_id(str(settings.get("selected_data_csv", "")))
    if selected is not None:
        settings["selected_data_csv"] = selected.source
    benchmark = build_benchmark(selected, cfg)
    bess = cfg["bess"]
    tariff = cfg["tariff"]
    economics = cfg["economics"]
    real_factor = _float_setting(settings, "billing_real_saving_factor", 0.60)
    sample = _sample_batteries()
    blank = _blank_oracle(real_factor)
    candidates = [{**item, "oracle": _blank_oracle(real_factor)} for item in sample]
    selected_idx = min(
        range(len(sample)),
        key=lambda index: abs(sample[index]["battery_capacity_kWh"] - float(bess["eCapKwh"]))
        + abs(sample[index]["battery_power_limit_kW"] - float(bess["pRatedKw"])),
    ) if sample else 0
    battery_cost = (
        float(bess["eCapKwh"]) * _float_setting(settings, "billing_battery_per_kWh", 5_000_000.0)
        + float(bess["pRatedKw"]) * _float_setting(settings, "billing_battery_per_kW", 4_000_000.0)
    )
    benchmark["summary"]["battery_cost_vnd"] = battery_cost
    return {
        "data_csv_files": [item.source for item in datasets],
        "selected_data_csv": selected.source if selected else "No compatible CSV found",
        "dt": (selected.res_min / 60.0) if selected and selected.res_min else 0.25,
        "battery_capacity_kWh": bess["eCapKwh"],
        "battery_power_limit_kW": bess["pRatedKw"],
        "charge_efficiency": bess["etaCh"],
        "discharge_efficiency": bess["etaDis"],
        "battery_wear_cost": economics["degradationCostPerKwhDischarged"],
        "minimum_soc": bess["socMin"],
        "maximum_soc": bess["socMax"],
        "required_final_soc": _float_setting(settings, "required_final_soc", min(float(bess["socMax"]), float(bess["socMin"]) + 0.05)),
        "billing_mode": tariff["billingMode"],
        "checked_2tc": "checked" if tariff["billingMode"] == "2tc" else "",
        "checked_tou": "checked" if tariff["billingMode"] == "tou" else "",
        "billing_sunday": bool(tariff["sundayNoPeak"]),
        "checked_sunday": "checked" if tariff["sundayNoPeak"] else "",
        "billing_expensive": tariff["pricePeak"],
        "billing_normal": tariff["priceMid"],
        "billing_cheap": tariff["priceOff"],
        "billing_peak_penalty": tariff["tCap"],
        "billing_windows_expensive": tariff["peakWindows"],
        "billing_windows_cheap": tariff["offWindows"],
        "billing_battery_per_kWh": _float_setting(settings, "billing_battery_per_kWh", 5_000_000.0),
        "billing_battery_per_kW": _float_setting(settings, "billing_battery_per_kW", 4_000_000.0),
        "billing_yearly_maintain_percentage": _float_setting(settings, "billing_yearly_maintain_percentage", 0.02),
        "billing_discount_rate": _float_setting(settings, "billing_discount_rate", 0.08),
        "billing_years": _float_setting(settings, "billing_years", 10),
        "billing_real_saving_factor": real_factor,
        "use_sample_battery_options": str(settings.get("use_sample_battery_options", "no")),
        "saved": saved,
        "benchmark": benchmark,
        "oracle": blank,
        "candidate_oracles": candidates,
        "sample_battery_candidates": sample,
        "selected_candidate_index": selected_idx,
        "should_calculate": should_calculate,
        "should_force_calculate": force_calculate,
        "exact_oracle_cache_exists": False,
        "csv_has_oracle_cache": any(ORACLE_CACHE_DIR.glob("*.json")),
        "ppo_gamma": 0.999,
        "ppo_lambda": 0.95,
        "ppo2_gamma": 1.0,
        "ppo2_lam_energy": 0.97,
        "ppo2_lam_peak": 0.97,
    }


def _load_training_core() -> dict[str, Any]:
    global _TRAINING_CORE
    if _TRAINING_CORE is not None:
        return _TRAINING_CORE
    engine_dir = core_src() / "bess_drl" / "training" / "drl_engine"
    engine_text = str(engine_dir)
    src_text = str(core_src())
    for item in (src_text, engine_text):
        if item not in sys.path:
            sys.path.insert(0, item)
    modules = {
        "runner": importlib.import_module("run_train_dataset"),
        "common": importlib.import_module("common"),
        "baselines": importlib.import_module("baselines"),
        "agent": importlib.import_module("ppo_agent_train"),
    }
    _TRAINING_CORE = modules
    return modules


def _checkpoint_raw(path: Path) -> dict[str, Any]:
    import torch

    value = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} is not a checkpoint dictionary")
    return value


def checkpoint_info(path: Path) -> dict[str, Any]:
    try:
        raw = _checkpoint_raw(path)
        meta = dict(raw.get("meta") or {})
        # The original UI displays native/control cadence badges. Current PPO is
        # contract-fixed at 15 minutes, so publish that fact even for checkpoints
        # created before these friendly UI metadata names existed.
        meta.setdefault("native_dt_minutes", 15)
        meta.setdefault("control_dt_minutes", meta.get("action_interval_minutes", 15))
        trained = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
        return {
            "name": path.name,
            "display_name": path.stem,
            "algo": str(raw.get("algo") or "ppo"),
            "e_cap_kwh": meta.get("e_cap_kwh"),
            "p_rated_kw": meta.get("p_rated_kw"),
            "billing_mode": meta.get("billing_mode", ""),
            "test_saving_pct": meta.get("test_saving_pct"),
            "trained": trained,
            "meta": meta,
        }
    except Exception as exc:  # noqa: BLE001 - checkpoint errors are presented in the UI.
        return {"name": path.name, "error": str(exc), "algo": "unknown", "trained": ""}


def list_checkpoints() -> list[dict[str, Any]]:
    ensure_og_dirs()
    rows = [checkpoint_info(path) for path in RESULTS_DIR.glob("policy_*.pt") if path.is_file()]
    rows.sort(key=lambda row: (RESULTS_DIR / row["name"]).stat().st_mtime, reverse=True)
    return rows


def _curve_for_checkpoint(name: str) -> list[dict[str, Any]]:
    stem = Path(name).stem
    tag = stem.removeprefix("policy_")
    path = RESULTS_DIR / f"training_curve_{tag}.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _curve_summary(curve: list[dict[str, Any]]) -> dict[str, Any]:
    def number(row: dict[str, Any], key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None

    points = [row for row in curve if number(row, "steps") is not None]
    valid_bill = [row for row in points if number(row, "val_cost_vnd") is not None]
    valid_saving = [row for row in points if number(row, "saving_vs_nobess_pct") is not None]
    valid_gap = [row for row in points if number(row, "oracle_gap_pct") is not None]
    return {
        "best_bill": min(valid_bill, key=lambda row: float(row["val_cost_vnd"])) if valid_bill else {},
        "best_saving": max(valid_saving, key=lambda row: float(row["saving_vs_nobess_pct"])) if valid_saving else {},
        "best_oracle_gap": min(valid_gap, key=lambda row: float(row["oracle_gap_pct"])) if valid_gap else {},
        "max_steps": max((float(row["steps"]) for row in points), default=0),
    }


def checkpoint_report(name: str) -> dict[str, Any]:
    path = (RESULTS_DIR / Path(name).name).resolve()
    if path.parent != RESULTS_DIR.resolve() or not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {name}")
    info = checkpoint_info(path)
    raw = _checkpoint_raw(path)
    meta = dict(raw.get("meta") or {})
    curve = _curve_for_checkpoint(path.name)
    tag = path.stem.removeprefix("policy_")
    evaluation_path = RESULTS_DIR / f"evaluation_{tag}.json"
    evaluation = _json_load(evaluation_path, {})
    test_metrics = dict(meta.get("test_metrics") or {})
    test = {
        "saving_pct": meta.get("test_saving_pct"),
        "oracle_gap_pct": meta.get("test_oracle_gap_pct"),
        "peak_kw": test_metrics.get("pmax_month_kw"),
        "policy_cost_vnd": test_metrics.get("total_cost_vnd"),
        "energy_cost_vnd": test_metrics.get("energy_cost_vnd"),
        "demand_cost_vnd": test_metrics.get("demand_cost_vnd"),
        "activity": {
            "throughput_kwh": test_metrics.get("throughput_kwh"),
        },
    }
    training = {
        "status": "saved",
        "algorithm": info.get("algo", "ppo"),
        "training": {
            "requested_steps": meta.get("total_steps"),
            "seed": meta.get("seed"),
            "actor_lr": meta.get("actor_learning_rate"),
            "critic_lr": meta.get("critic_learning_rate"),
            "lambda_energy": meta.get("lambda_energy"),
            "lambda_peak_selected": meta.get("lambda_peak_selected"),
        },
        "dataset": {
            "source": meta.get("train_csv"),
            "validation_range": meta.get("validation_range"),
            "test_range": meta.get("test_range"),
        },
        "validation": {
            "no_bess_vnd": None,
            "oracle_vnd": None,
        },
        "test": test,
        "battery": {
            "e_cap_kwh": meta.get("e_cap_kwh"),
            "p_rated_kw": meta.get("p_rated_kw"),
        },
        "billing_mode": meta.get("billing_mode"),
        "p_ref_kw": meta.get("p_ref_kw"),
    }
    warnings: list[str] = []
    if not curve:
        warnings.append("No persisted training_curve CSV was found for this checkpoint.")
    if not evaluation:
        warnings.append("No holdout evaluation JSON was found for this checkpoint.")
    return {
        "checkpoint": {**info, "meta": meta},
        "training": training,
        "curve": curve,
        "summary": _curve_summary(curve),
        "artifacts": {
            "curve": str((RESULTS_DIR / f"training_curve_{tag}.csv").name) if curve else None,
            "evaluation": evaluation_path.name if evaluation else None,
            "report": None,
        },
        "warnings": warnings,
        "evaluation": evaluation,
        "run_settings": {},
    }


def _dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_config_for_request(payload: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    settings = _settings()
    cfg = _effective_config(
        settings,
        e_cap_kwh=float(payload.get("e_cap_kwh") or 1250),
        p_rated_kw=float(payload.get("p_rated_kw") or 450),
        wear=float(payload.get("battery_wear_cost") or _float_setting(settings, "battery_wear_cost", 500)),
    )
    return _write_config(cfg, "train"), cfg


def _safe_tag(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return f"ui_{time.strftime('%Y%m%d_%H%M%S')}"
    clean = "".join(character if character.isalnum() or character in "_.-" else "_" for character in text)
    return clean[:80] or "ui_policy"


def training_command(payload: dict[str, Any], dataset: DatasetInfo) -> tuple[list[str], str, str]:
    config_path, _ = _training_config_for_request(payload)
    algo = str(payload.get("algo") or "ppo").lower()
    tag = _safe_tag(payload.get("tag"))
    steps = int(payload.get("ppo2_steps") if algo == "ppo2" else payload.get("steps") or 1_500_000)
    seed = int(payload.get("ppo2_seed") if algo == "ppo2" else 0)
    command = [
        sys.executable,
        str(HERE / "main.py"),
        "train",
        "--csv",
        str(dataset.path),
        "--config-json",
        str(config_path),
        "--steps",
        str(max(1, steps)),
        "--tag",
        tag,
        "--seed",
        str(seed),
    ]
    if algo == "ppo2":
        optional = [
            ("--seeds", payload.get("ppo2_seeds")),
            ("--rollout", payload.get("ppo2_rollout")),
            ("--eval-every", payload.get("ppo2_eval_every")),
            ("--min-month-coverage", payload.get("ppo2_min_month_coverage")),
            ("--val-months", payload.get("ppo2_val_months")),
            ("--test-months", payload.get("ppo2_test_months")),
            ("--torch-threads", payload.get("ppo2_torch_threads")),
            ("--gamma", payload.get("ppo2_gamma")),
            ("--actor-lr", payload.get("ppo2_actor_lr")),
            ("--critic-lr", payload.get("ppo2_critic_lr")),
            ("--init-std", payload.get("ppo2_init_std")),
            ("--clip-penalty", payload.get("ppo2_clip_penalty")),
            ("--bc-epochs", payload.get("ppo2_bc_epochs")),
            ("--ppo-clip", payload.get("ppo2_clip")),
            ("--ppo-epochs", payload.get("ppo2_epochs")),
            ("--ppo-minibatch", payload.get("ppo2_minibatch")),
            ("--ent-coef", payload.get("ppo2_ent_coef")),
            ("--vf-coef", payload.get("ppo2_vf_coef")),
            ("--target-kl", payload.get("ppo2_target_kl")),
            ("--shaping-margin", payload.get("ppo2_shaping_margin")),
            ("--aug-load-sigma", payload.get("ppo2_aug_load_sigma")),
            ("--aug-pv-sigma", payload.get("ppo2_aug_pv_sigma")),
            ("--aug-rho-load", payload.get("ppo2_aug_rho_load")),
            ("--aug-rho-pv", payload.get("ppo2_aug_rho_pv")),
            ("--bc-lr", payload.get("ppo2_bc_lr")),
            ("--bc-minibatch", payload.get("ppo2_bc_minibatch")),
            ("--bc-action-clip", payload.get("ppo2_bc_action_clip")),
            ("--lambda-energy", payload.get("ppo2_lam_energy")),
            ("--lambda-peak", payload.get("ppo2_lam_peak")),
        ]
        for flag, value in optional:
            if value not in (None, ""):
                command.extend([flag, str(value)])
    else:
        gamma = payload.get("gamma")
        lam = payload.get("lambda")
        val_days = max(1, int(payload.get("val_days") or 30))
        test_days = max(1, int(payload.get("test_days") or 30))
        command.extend([
            "--val-months", str(max(1, round(val_days / 30))),
            "--test-months", str(max(1, round(test_days / 30))),
        ])
        if gamma not in (None, ""):
            command.extend(["--gamma", str(gamma)])
        if lam not in (None, ""):
            command.extend(["--lambda-energy", str(lam)])
    return command, f"policy_{tag}.pt", algo


def sse_for_training_slot(slot: Any) -> Response:
    def stream():
        after = 0
        while True:
            lines = slot.log_since(after)
            for row in lines:
                after = max(after, int(row["seq"]))
                clean = str(row["line"]).replace("\r", " ").replace("\n", " ")
                yield f"data: {clean}\n\n"
            snapshot = slot.snapshot()
            if not snapshot["running"] and snapshot["status"] in {"done", "error", "stopped"}:
                yield "event: end\ndata: end\n\n"
                break
            yield ": keepalive\n\n"
            time.sleep(0.35)

    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


def _save_form_settings() -> dict[str, Any]:
    settings = _settings()
    for key, value in request.form.items():
        if key == "form_action":
            continue
        settings[key] = value
    settings["billing_sunday"] = "billing_sunday" in request.form
    selected = request.form.get("selected_data_csv")
    if selected:
        settings["selected_data_csv"] = selected
    _json_save(UI_SETTINGS_PATH, settings)
    _BENCH_CACHE.clear()
    return settings


def register_og_routes(app: Any, manager: Any) -> None:
    """Register the original UI's route contract on the Flask app."""

    @app.post("/set-parameters")
    def og_set_parameters():
        _save_form_settings()
        action = request.form.get("form_action", "save")
        return render_template(
            "index.html",
            **build_index_context(
                should_calculate=action in {"calculate", "recalculate"},
                force_calculate=action == "recalculate",
                saved=action == "save",
            ),
        )

    @app.get("/api/training/datasets")
    def og_training_datasets():
        return jsonify([item.api_dict() for item in discover_datasets()])

    @app.get("/api/training/checkpoints")
    def og_training_checkpoints():
        return jsonify(list_checkpoints())

    @app.get("/api/training/checkpoints/<path:name>/report")
    def og_checkpoint_report(name: str):
        try:
            return jsonify(checkpoint_report(Path(name).name))
        except (OSError, ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 404

    @app.post("/api/training/oracle-status")
    def og_oracle_status():
        payload = request.get_json(silent=True) or {}
        dataset = _dataset_by_id(str(payload.get("dataset_id") or ""))
        if dataset is None:
            return jsonify({"ready": False, "error": "No compatible local CSV dataset found."}), 400
        if abs(dataset.res_min - 15.0) > 1e-6:
            return jsonify({"ready": False, "error": f"Current debloated PPO core requires native 15-minute CSV data; this dataset is {dataset.res_min:g}m."}), 400
        return jsonify({"ready": True, "solved_days": dataset.n_days, "message": "The current trainer computes its month Oracle internally; no separate cache gate is required."})

    @app.post("/api/training/start")
    def og_training_start():
        payload = request.get_json(silent=True) or {}
        dataset = _dataset_by_id(str(payload.get("dataset_id") or ""))
        if dataset is None:
            return jsonify({"error": "No compatible training dataset selected."}), 400
        control_dt = float(payload.get("control_dt_minutes") or 15.0)
        if abs(dataset.res_min - 15.0) > 1e-6 or abs(control_dt - 15.0) > 1e-6:
            return jsonify({
                "error": (
                    "The current causal PPO contract is fixed at 15-minute control. "
                    f"Dataset={dataset.res_min:g}m, requested control={control_dt:g}m."
                )
            }), 400
        if str(payload.get("device") or "auto").lower() == "cuda":
            return jsonify({
                "error": "This PPO implementation currently runs on CPU; explicit CUDA mode is not implemented."
            }), 400
        try:
            command, checkpoint_name, algo = training_command(payload, dataset)
            snapshot = manager.start("training", command)
        except (OSError, ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(
            {
                "job_id": f"training-{snapshot['started_at'] or uuid4().hex[:8]}",
                "checkpoint": str(RESULTS_DIR / checkpoint_name),
                "algo": algo,
                "device": "cpu",
            }
        )

    @app.get("/api/training/jobs/<path:_job_id>/events")
    def og_training_events(_job_id: str):
        return sse_for_training_slot(manager.slots["training"])

    @app.post("/api/training/stop/<path:_job_id>")
    def og_training_stop(_job_id: str):
        return jsonify(manager.stop("training"))

    @app.get("/api/weather/context")
    def og_weather_context():
        rows = []
        for dataset in discover_datasets():
            status = _json_load(WEATHER_DIR / f"{dataset.id.replace(':', '_')}.json", {})
            weather = status.get("meta", {}) if isinstance(status, dict) else {}
            rows.append({**dataset.api_dict(), "weather": weather or {"ready": False, "message": "Real weather has not been downloaded."}})
        return jsonify({"datasets": rows})

    @app.post("/api/weather/fetch")
    def og_weather_fetch():
        payload = request.get_json(silent=True) or {}
        dataset = _dataset_by_id(str(payload.get("dataset_id") or ""))
        if dataset is None or not dataset.start_date or not dataset.end_date:
            return jsonify({"error": "A dated local dataset is required."}), 400
        try:
            latitude = float(payload["latitude"])
            longitude = float(payload["longitude"])
            timezone_name = str(payload.get("timezone") or "Asia/Ho_Chi_Minh")
            provider = str(payload.get("provider") or "open-meteo")
            if provider == "custom":
                template = str(payload.get("custom_url") or "")
                if not template.lower().startswith("https://"):
                    raise ValueError("Custom weather URL must use HTTPS.")
                url = template.format(start_date=dataset.start_date, end_date=dataset.end_date, latitude=latitude, longitude=longitude, timezone=urllib.parse.quote(timezone_name))
                headers = {}
                header_name = str(payload.get("api_key_header") or "").strip()
                api_key = str(payload.get("api_key") or "")
                if header_name and api_key:
                    headers[header_name] = api_key
            else:
                query = urllib.parse.urlencode(
                    {
                        "latitude": latitude,
                        "longitude": longitude,
                        "start_date": dataset.start_date,
                        "end_date": dataset.end_date,
                        "timezone": timezone_name,
                        "hourly": "temperature_2m,precipitation,cloud_cover,shortwave_radiation",
                    }
                )
                url = f"https://archive-api.open-meteo.com/v1/archive?{query}"
                headers = {}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                weather_payload = json.loads(response.read().decode("utf-8"))
            hourly = weather_payload.get("hourly", {}) if isinstance(weather_payload, dict) else {}
            hours = len(hourly.get("time", [])) if isinstance(hourly, dict) else 0
            saved = {
                "meta": {
                    "ready": hours > 0,
                    "provider": provider,
                    "start_date": dataset.start_date,
                    "end_date": dataset.end_date,
                    "hours": hours,
                },
                "data": weather_payload,
            }
            _json_save(WEATHER_DIR / f"{dataset.id.replace(':', '_')}.json", saved)
            return jsonify(saved["meta"])
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/benchmarking/cache")
    def og_benchmark_cache():
        payload = request.get_json(silent=True) or {}
        policies = sorted(Path(str(name)).name for name in payload.get("policies", []))
        dataset = _dataset_by_id(None)
        if dataset is None or not policies:
            return jsonify({"cached": None})
        dataset_sha = _dataset_hash(dataset.path)
        config_hash = _effective_config(_settings())["meta"]["configHash"]
        for path in sorted(
            BENCH_DIR.glob("run_*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            run = _json_load(path, {})
            if not run:
                continue
            if run.get("snapshot", {}).get("dataset", {}).get("sha256") != dataset_sha:
                continue
            if run.get("config_hash") != config_hash:
                continue
            if sorted(run.get("policy_names", [])) != policies:
                continue
            return jsonify({
                "cached": {
                    "id": run.get("id", path.stem),
                    "created": run.get("created", path.stat().st_mtime),
                    "fingerprint": run.get("fingerprint", ""),
                }
            })
        return jsonify({"cached": None})


def index_context() -> dict[str, Any]:
    return build_index_context()
