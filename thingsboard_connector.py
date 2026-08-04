"""Custom ThingsBoard telemetry connector for restart-safe Shadow Running."""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from scenario_gen import DayData


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "shadow" / "thingsboard.json"
TOKEN_TTL_SECONDS = 2 * 60 * 60
MAX_INTERVALS_PER_REQUEST = 650
DEFAULT_CONFIG = {
    "base_url": "",
    "username": "",
    "password": "",
    "device_id": "",
    "key_load": "",
    "key_pv": "",
    "unit_scale": 1.0,
    "timezone": "Asia/Bangkok",
    "interval_minutes": 15,
    "max_gap_minutes": 15,
    "pv_zero_threshold_kw": 1.0,
}
_TOKEN_LOCK = threading.Lock()
_TOKEN: str | None = None
_TOKEN_EXPIRES_AT = 0.0
_TOKEN_SIGNATURE: tuple[str, str, str] | None = None


class ThingsBoardError(RuntimeError):
    pass


def _raw_config() -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            config.update({key: saved[key] for key in DEFAULT_CONFIG if key in saved})
    except (OSError, json.JSONDecodeError):
        pass
    return config


def public_config() -> dict[str, Any]:
    config = _raw_config()
    config.pop("password", None)
    config["has_password"] = bool(_raw_config().get("password"))
    return config


def save_config(payload: dict[str, Any]) -> dict[str, Any]:
    current = _raw_config()
    password = str(payload.get("password") or "")
    config = {
        "base_url": str(payload.get("base_url") or "").strip().rstrip("/"),
        "username": str(payload.get("username") or "").strip(),
        "password": password if password else str(current.get("password") or ""),
        "device_id": str(payload.get("device_id") or "").strip(),
        "key_load": str(payload.get("key_load") or "").strip(),
        "key_pv": str(payload.get("key_pv") or "").strip(),
        "unit_scale": _positive_float(payload.get("unit_scale"), "unit scale"),
        "timezone": str(payload.get("timezone") or "").strip(),
        "interval_minutes": _positive_int(payload.get("interval_minutes"), "interval minutes"),
        "max_gap_minutes": _nonnegative_int(payload.get("max_gap_minutes"), "maximum gap minutes"),
        "pv_zero_threshold_kw": _nonnegative_float(
            payload.get("pv_zero_threshold_kw"), "PV zero threshold"
        ),
    }
    if not config["base_url"].startswith(("https://", "http://")):
        raise ThingsBoardError("ThingsBoard URL must begin with https:// or http://.")
    for field in ("username", "password", "device_id", "key_load", "key_pv", "timezone"):
        if not config[field]:
            raise ThingsBoardError(f"ThingsBoard {field.replace('_', ' ')} is required.")
    try:
        ZoneInfo(config["timezone"])
    except ZoneInfoNotFoundError as exc:
        raise ThingsBoardError(f"Unknown timezone: {config['timezone']}") from exc
    if 1440 % config["interval_minutes"]:
        raise ThingsBoardError("Sampling interval must divide exactly into 24 hours.")
    if config["max_gap_minutes"] % config["interval_minutes"]:
        raise ThingsBoardError("Maximum gap must be an exact multiple of the sampling interval.")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CONFIG_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
    temporary.replace(CONFIG_PATH)
    _clear_token()
    return public_config()


def _positive_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ThingsBoardError(f"{label} must be a number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ThingsBoardError(f"{label} must be positive.")
    return number


def _nonnegative_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ThingsBoardError(f"{label} must be a number.") from exc
    if not math.isfinite(number) or number < 0:
        raise ThingsBoardError(f"{label} cannot be negative.")
    return number


def _positive_int(value: Any, label: str) -> int:
    number = int(_positive_float(value, label))
    if float(value) != number:
        raise ThingsBoardError(f"{label} must be a whole number.")
    return number


def _nonnegative_int(value: Any, label: str) -> int:
    number = int(_nonnegative_float(value, label))
    if float(value) != number:
        raise ThingsBoardError(f"{label} must be a whole number.")
    return number


def _clear_token() -> None:
    global _TOKEN, _TOKEN_EXPIRES_AT, _TOKEN_SIGNATURE
    with _TOKEN_LOCK:
        _TOKEN = None
        _TOKEN_EXPIRES_AT = 0.0
        _TOKEN_SIGNATURE = None


def _http_error(error: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(error.read().decode("utf-8"))
        return str(body.get("message") or body)[:300]
    except Exception:  # noqa: BLE001
        return f"HTTP {error.code}"


def _json_request(request: urllib.request.Request, timeout: int = 30) -> Any:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ThingsBoardError(f"ThingsBoard HTTP {exc.code}: {_http_error(exc)}") from exc
    except urllib.error.URLError as exc:
        raise ThingsBoardError(f"ThingsBoard connection failed: {exc.reason}") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise ThingsBoardError(f"Invalid or timed-out ThingsBoard response: {exc}") from exc


def _login(config: dict[str, Any], force: bool = False) -> str:
    global _TOKEN, _TOKEN_EXPIRES_AT, _TOKEN_SIGNATURE
    signature = (config["base_url"], config["username"], config["device_id"])
    with _TOKEN_LOCK:
        if not force and _TOKEN and _TOKEN_SIGNATURE == signature and time.time() < _TOKEN_EXPIRES_AT:
            return _TOKEN
        request = urllib.request.Request(
            f"{config['base_url']}/api/auth/login",
            data=json.dumps({"username": config["username"], "password": config["password"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = _json_request(request, timeout=20)
        token = response.get("token") if isinstance(response, dict) else None
        if not token:
            raise ThingsBoardError("ThingsBoard login response did not contain a token.")
        _TOKEN = str(token)
        _TOKEN_EXPIRES_AT = time.time() + TOKEN_TTL_SECONDS
        _TOKEN_SIGNATURE = signature
        return _TOKEN


def _authorized_get(config: dict[str, Any], url: str, force_login: bool = False) -> Any:
    token = _login(config, force=force_login)
    request = urllib.request.Request(url, headers={"X-Authorization": f"Bearer {token}"})
    try:
        return _json_request(request)
    except ThingsBoardError as exc:
        if "HTTP 401" in str(exc) and not force_login:
            return _authorized_get(config, url, force_login=True)
        raise


def test_connection() -> dict[str, Any]:
    config = _validated_runtime_config()
    url = (
        f"{config['base_url']}/api/plugins/telemetry/DEVICE/"
        f"{urllib.parse.quote(config['device_id'], safe='')}/keys/timeseries"
    )
    keys = _authorized_get(config, url)
    if not isinstance(keys, list):
        raise ThingsBoardError("ThingsBoard telemetry-key response was not a list.")
    requested = _keys(config["key_load"]) + _keys(config["key_pv"])
    missing = [key for key in requested if key not in keys]
    return {
        "ok": not missing,
        "available_keys": sorted(map(str, keys)),
        "requested_keys": requested,
        "missing_keys": missing,
        "message": "Connection and telemetry keys are ready." if not missing else "Connected, but configured telemetry keys are missing.",
    }


def _validated_runtime_config() -> dict[str, Any]:
    config = _raw_config()
    if not all(config.get(field) for field in ("base_url", "username", "password", "device_id", "key_load", "key_pv")):
        raise ThingsBoardError("Save the complete ThingsBoard connector configuration first.")
    return config


def _keys(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def fetch_days(start_iso: str, end_iso: str) -> tuple[list[DayData], dict[str, str], dict[str, Any]]:
    config = _validated_runtime_config()
    start_day = date.fromisoformat(start_iso)
    end_day = date.fromisoformat(end_iso)
    yesterday = date.today() - timedelta(days=1)
    if end_day > yesterday:
        end_day = yesterday
    if start_day > end_day:
        raise ThingsBoardError("ThingsBoard range is empty after excluding today's incomplete data.")
    telemetry = _fetch_range(config, start_day, end_day)
    days, failures = _build_days(config, telemetry, start_day, end_day)
    return days, failures, {
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "valid_days": len(days),
        "invalid_days": len(failures),
        "interval_minutes": config["interval_minutes"],
    }


def _fetch_range(config: dict[str, Any], start_day: date, end_day: date) -> dict[str, list]:
    timezone = ZoneInfo(config["timezone"])
    start_ms = int(datetime.combine(start_day, datetime.min.time(), timezone).timestamp() * 1000)
    end_ms = int(datetime.combine(end_day + timedelta(days=1), datetime.min.time(), timezone).timestamp() * 1000)
    interval_ms = int(config["interval_minutes"]) * 60_000
    chunk_ms = MAX_INTERVALS_PER_REQUEST * interval_ms
    requested_keys = list(dict.fromkeys(_keys(config["key_load"]) + _keys(config["key_pv"])))
    merged: dict[str, list] = defaultdict(list)
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)
        query = urllib.parse.urlencode({
            "keys": ",".join(requested_keys),
            "startTs": cursor,
            "endTs": chunk_end,
            "interval": interval_ms,
            "agg": "AVG",
            "limit": 60000,
            "orderBy": "ASC",
        })
        url = (
            f"{config['base_url']}/api/plugins/telemetry/DEVICE/"
            f"{urllib.parse.quote(config['device_id'], safe='')}/values/timeseries?{query}"
        )
        part = _authorized_get(config, url)
        if isinstance(part, dict):
            for key, points in part.items():
                if isinstance(points, list):
                    merged[str(key)].extend(points)
        cursor = chunk_end
    return dict(merged)


def _channel(config: dict[str, Any], telemetry: dict[str, list], key_spec: str) -> dict[int, float]:
    scale = float(config["unit_scale"])
    output: dict[int, float] = defaultdict(float)
    for key in _keys(key_spec):
        buckets: dict[int, list[float]] = defaultdict(list)
        for point in telemetry.get(key, []):
            try:
                timestamp = int(point["ts"])
                buckets[timestamp].append(float(point["value"]) * scale)
            except (KeyError, TypeError, ValueError):
                continue
        for bucket, values in buckets.items():
            output[bucket] += math.fsum(values) / len(values)
    return dict(output)


def _build_days(config, telemetry, start_day: date, end_day: date) -> tuple[list[DayData], dict[str, str]]:
    timezone = ZoneInfo(config["timezone"])
    interval = int(config["interval_minutes"])
    steps = 1440 // interval
    max_gap_steps = int(config["max_gap_minutes"]) // interval
    load_channel = _channel(config, telemetry, config["key_load"])
    pv_channel = _channel(config, telemetry, config["key_pv"])
    samples: dict[date, dict[int, tuple[float | None, float | None]]] = defaultdict(dict)
    for timestamp in load_channel.keys() | pv_channel.keys():
        local = datetime.fromtimestamp(timestamp / 1000, timezone)
        step = (local.hour * 60 + local.minute) // interval
        samples[local.date()][step] = (load_channel.get(timestamp), pv_channel.get(timestamp))
    valid = []
    failures: dict[str, str] = {}
    cursor = start_day
    while cursor <= end_day:
        points = samples.get(cursor, {})
        load = _interpolate([points.get(step, (None, None))[0] for step in range(steps)], max_gap_steps)
        pv = _interpolate(
            [points.get(step, (None, None))[1] for step in range(steps)],
            max_gap_steps,
            zero_threshold=float(config["pv_zero_threshold_kw"]),
        )
        if load is None or pv is None:
            failures[cursor.isoformat()] = "load/PV telemetry is missing or exceeds the configured gap limit"
        else:
            valid.append(DayData(
                load=_array(load),
                pv=_array(pv),
                day_type="weekend" if cursor.weekday() >= 5 else "working",
                weather="thingsboard",
                date_iso=cursor.isoformat(),
            ))
        cursor += timedelta(days=1)
    return valid, failures


def _array(values: list[float]):
    import numpy as np
    return np.asarray(values, dtype=np.float64)


def _interpolate(values, max_gap_steps: int, zero_threshold: float | None = None) -> list[float] | None:
    known = [index for index, value in enumerate(values) if value is not None]
    if not known:
        return None
    result = list(values)
    first, last = known[0], known[-1]
    first_value, last_value = float(values[first]), float(values[last])
    first_zero = zero_threshold is not None and abs(first_value) <= zero_threshold
    last_zero = zero_threshold is not None and abs(last_value) <= zero_threshold
    if first > max_gap_steps and not first_zero:
        return None
    if len(values) - last - 1 > max_gap_steps and not last_zero:
        return None
    for index in range(first):
        result[index] = 0.0 if first_zero else first_value
    for left, right in zip(known, known[1:]):
        gap = right - left - 1
        left_value, right_value = float(values[left]), float(values[right])
        zero_gap = zero_threshold is not None and abs(left_value) <= zero_threshold and abs(right_value) <= zero_threshold
        if gap > max_gap_steps and not zero_gap:
            return None
        for index in range(left + 1, right):
            fraction = (index - left) / (right - left)
            result[index] = left_value + (right_value - left_value) * fraction
    for index in range(last + 1, len(values)):
        result[index] = 0.0 if last_zero else last_value
    return [float(value) for value in result]
