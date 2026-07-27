from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from benchmark import selected_data_path


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "user_data" / "oracle_lp_cache"
CACHE_VERSION = 1

ORACLE_PARAMETER_KEYS = (
    "selected_data_csv",
    "dt",
    "battery_capacity_kWh",
    "battery_power_limit_kW",
    "charge_efficiency",
    "discharge_efficiency",
    "battery_wear_cost",
    "minimum_soc",
    "maximum_soc",
    "required_final_soc",
    "billing_mode",
    "billing_sunday",
    "billing_expensive",
    "billing_normal",
    "billing_cheap",
    "billing_peak_penalty",
    "billing_windows_expensive",
    "billing_windows_cheap",
    "billing_battery_per_kWh",
    "billing_battery_per_kW",
    "billing_yearly_maintain_percentage",
    "billing_discount_rate",
    "billing_years",
    "billing_real_saving_factor",
)


def cached_oracle_lp(parameters: dict[str, Any]) -> dict[str, Any] | None:
    payload = _read_cache(_cache_path(parameters))
    if not payload:
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    return _with_cache_meta(result, payload, hit=True)


def get_or_build_oracle_lp(
    parameters: dict[str, Any],
    builder: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    force: bool = False,
) -> dict[str, Any]:
    cache_path = _cache_path(parameters)
    if not force:
        payload = _read_cache(cache_path)
        if payload and isinstance(payload.get("result"), dict):
            return _with_cache_meta(payload["result"], payload, hit=True)

    result = builder(parameters)
    payload = _payload(parameters, result)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
    return _with_cache_meta(result, payload, hit=False)


def selected_csv_has_cache(parameters: dict[str, Any]) -> bool:
    prefix = _csv_cache_prefix(parameters)
    return CACHE_DIR.exists() and any(CACHE_DIR.glob(f"{prefix}-*.json"))


def exact_cache_exists(parameters: dict[str, Any]) -> bool:
    return _cache_path(parameters).exists()


def _payload(parameters: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    csv_path = selected_data_path(parameters)
    return {
        "version": CACHE_VERSION,
        "created_at": time.time(),
        "csv": {
            "filename": csv_path.name,
            "fingerprint": _file_fingerprint(csv_path),
        },
        "parameters": _normalized_parameters(parameters),
        "result": result,
    }


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        return None
    csv_info = payload.get("csv")
    if not isinstance(csv_info, dict):
        return None
    csv_path = selected_data_path(payload.get("parameters", {}))
    if csv_info.get("fingerprint") != _file_fingerprint(csv_path):
        return None
    return payload


def _cache_path(parameters: dict[str, Any]) -> Path:
    key = {
        "version": CACHE_VERSION,
        "csv": {
            "filename": selected_data_path(parameters).name,
            "fingerprint": _file_fingerprint(selected_data_path(parameters)),
        },
        "parameters": _normalized_parameters(parameters),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{_csv_cache_prefix(parameters)}-{digest}.json"


def _csv_cache_prefix(parameters: dict[str, Any]) -> str:
    csv_path = selected_data_path(parameters)
    raw = f"{csv_path.name}:{_file_fingerprint(csv_path)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_parameters(parameters: dict[str, Any]) -> dict[str, str]:
    normalized = {}
    for key in ORACLE_PARAMETER_KEYS:
        value = parameters.get(key)
        if isinstance(value, bool):
            normalized[key] = "true" if value else "false"
        else:
            normalized[key] = str(value)
    return normalized


def _with_cache_meta(result: dict[str, Any], payload: dict[str, Any], *, hit: bool) -> dict[str, Any]:
    cloned = json.loads(json.dumps(result, default=_json_default))
    cloned["cache"] = {
        "hit": hit,
        "created_at": payload.get("created_at"),
        "csv_filename": payload.get("csv", {}).get("filename"),
    }
    if hit:
        cloned["status"] = f"Cached Oracle LP: {cloned.get('status', 'ready')}"
    return cloned


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
