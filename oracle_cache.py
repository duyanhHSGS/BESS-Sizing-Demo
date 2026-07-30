from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from benchmark import selected_data_path


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "user_data" / "oracle_lp_cache"
CACHE_VERSION = 2

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
)
NUMERIC_PARAMETER_KEYS = frozenset(
    {
        "dt",
        "battery_capacity_kWh",
        "battery_power_limit_kW",
        "charge_efficiency",
        "discharge_efficiency",
        "battery_wear_cost",
        "minimum_soc",
        "maximum_soc",
        "required_final_soc",
        "billing_expensive",
        "billing_normal",
        "billing_cheap",
        "billing_peak_penalty",
    }
)
BOOLEAN_PARAMETER_KEYS = frozenset({"billing_sunday"})


class OracleCacheRequired(ValueError):
    """Raised when training has no exact, complete month-wide Oracle result."""


def cached_oracle_lp(parameters: dict[str, Any]) -> dict[str, Any] | None:
    payload = _read_cache(_cache_path(parameters))
    if not payload:
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    return _with_cache_meta(result, payload, hit=True, parameters=parameters)


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
            return _with_cache_meta(payload["result"], payload, hit=True, parameters=parameters)

    result = builder(parameters)
    payload = _payload(parameters, result)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
    return _with_cache_meta(result, payload, hit=False, parameters=parameters)


def selected_csv_has_cache(parameters: dict[str, Any]) -> bool:
    prefix = _csv_cache_prefix(parameters)
    return CACHE_DIR.exists() and any(CACHE_DIR.glob(f"{prefix}-*.json"))


def exact_cache_exists(parameters: dict[str, Any]) -> bool:
    return _cache_path(parameters).exists()


def require_cached_oracle(parameters: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _cache_path(parameters)
    payload = _read_cache(path)
    result = payload.get("result") if payload else None
    if not isinstance(result, dict):
        raise OracleCacheRequired(
            "Exact Oracle LP cache required. Select this CSV and battery configuration "
            "in Sizing, calculate Oracle LP, then start training again."
        )
    days = result.get("days")
    if not result.get("available") or not isinstance(days, list) or not days:
        raise OracleCacheRequired("Oracle LP cache exists but has no usable day traces.")
    failed = [day.get("day_index") for day in days if not day.get("solved")]
    if failed:
        raise OracleCacheRequired(
            f"Oracle LP cache is incomplete; unsolved day indexes: {failed[:10]}"
        )
    from benchmark import _load_rows

    expected = {row["day_index"] for row in _load_rows(selected_data_path(parameters))}
    cached = {int(day["day_index"]) for day in days}
    missing = sorted(expected - cached)
    if missing:
        raise OracleCacheRequired(
            f"Oracle LP cache is incomplete; missing day indexes: {missing[:10]}"
        )
    return path, _with_cache_meta(result, payload, hit=True, parameters=parameters)


def load_cached_training_grids(
    cache_path: str | Path,
    day_indexes: list[int],
) -> list[list[float]]:
    path = Path(cache_path).resolve()
    if path.parent != CACHE_DIR.resolve():
        raise OracleCacheRequired(f"Oracle cache path is outside the cache directory: {path}")
    payload = _read_cache(path)
    result = payload.get("result") if payload else None
    days = result.get("days") if isinstance(result, dict) else None
    if not isinstance(days, list):
        raise OracleCacheRequired("Oracle LP cache is missing or invalid.")
    by_index = {
        int(day["day_index"]): day
        for day in days
        if day.get("solved") and isinstance(day.get("grid"), list)
    }
    missing = [index for index in day_indexes if index not in by_index]
    if missing:
        raise OracleCacheRequired(
            f"Oracle LP cache does not contain validation day indexes: {missing[:10]}"
        )
    return [[float(value) for value in by_index[index]["grid"]] for index in day_indexes]


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
    csv_path = selected_data_path(parameters)
    fingerprint = _file_fingerprint(csv_path)
    key = {
        "version": CACHE_VERSION,
        "csv": {
            "filename": csv_path.name,
            "fingerprint": fingerprint,
        },
        "parameters": _normalized_parameters(parameters),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    raw_prefix = f"{csv_path.name}:{fingerprint}"
    prefix = hashlib.sha256(raw_prefix.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{prefix}-{digest}.json"


def _csv_cache_prefix(parameters: dict[str, Any]) -> str:
    csv_path = selected_data_path(parameters)
    raw = f"{csv_path.name}:{_file_fingerprint(csv_path)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return _cached_file_fingerprint(
        str(path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
    )


@lru_cache(maxsize=32)
def _cached_file_fingerprint(path_text: str, file_size: int, file_mtime_ns: int) -> str:
    del file_size, file_mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_parameters(parameters: dict[str, Any]) -> dict[str, str]:
    normalized = {}
    for key in ORACLE_PARAMETER_KEYS:
        value = parameters.get(key)
        if key in NUMERIC_PARAMETER_KEYS:
            try:
                normalized[key] = format(float(value), ".12g")
            except (TypeError, ValueError):
                normalized[key] = "invalid"
        elif key in BOOLEAN_PARAMETER_KEYS:
            enabled = value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "on"}
            normalized[key] = "true" if enabled else "false"
        else:
            normalized[key] = str(value)
    return normalized


def _with_cache_meta(
    result: dict[str, Any],
    payload: dict[str, Any],
    *,
    hit: bool,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cloned = dict(result)
    if isinstance(result.get("summary"), dict):
        cloned["summary"] = dict(result["summary"])
    if parameters is not None:
        _refresh_economics(cloned, parameters)
    cloned["cache"] = {
        "hit": hit,
        "created_at": payload.get("created_at"),
        "csv_filename": payload.get("csv", {}).get("filename"),
    }
    if hit:
        cloned["status"] = f"Cached Oracle LP: {cloned.get('status', 'ready')}"
    return cloned


def _refresh_economics(result: dict[str, Any], parameters: dict[str, Any]) -> None:
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return
    from oracle_lp import _sizing_economics

    try:
        seer_factor = min(max(float(parameters.get("billing_real_saving_factor", 1.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        seer_factor = 1.0
    oracle_saving = float(summary.get("oracle_saving_vnd", 0.0))
    oracle_annual = float(summary.get("oracle_annual_saving_vnd", 0.0))
    seer_annual = max(0.0, oracle_annual) * seer_factor
    summary["seer_factor"] = seer_factor
    summary["seer_saving_vnd"] = round(max(0.0, oracle_saving) * seer_factor)
    summary["seer_annual_saving_vnd"] = round(seer_annual)
    summary["sizing_economics"] = _sizing_economics(
        parameters,
        oracle_annual,
        seer_annual,
        float(summary.get("peak_grid_kW", 0.0)),
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
