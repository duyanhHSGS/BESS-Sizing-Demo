"""Custom real-weather provider and causal forecast inference for Shadow."""
from __future__ import annotations

import ipaddress
import json
import socket
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from bess.paths import PROJECT_ROOT
from typing import Any
from urllib.parse import urlencode, urlparse

import numpy as np
import requests

from bess.forecasting.weather_forecast import ARTIFACT_FIELDS, FORECAST_DIR, HOURLY_FIELDS, WeatherError


BASE_DIR = PROJECT_ROOT
STORE_DIR = BASE_DIR / "shadow"
CONFIG_PATH = STORE_DIR / "weather.json"
DB_PATH = STORE_DIR / "weather.sqlite"
DEFAULT_CONFIG = {
    "provider": "open-meteo",
    "latitude": None,
    "longitude": None,
    "timezone": "Asia/Bangkok",
    "custom_url": "",
    "api_key_header": "",
    "api_key": "",
}


class ShadowWeatherError(RuntimeError):
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
    raw = _raw_config()
    raw.pop("api_key", None)
    raw["has_api_key"] = bool(_raw_config().get("api_key"))
    return raw


def scientific_snapshot() -> dict[str, Any]:
    config = public_config()
    config.pop("has_api_key", None)
    return config


def save_config(payload: dict[str, Any]) -> dict[str, Any]:
    current = _raw_config()
    provider = str(payload.get("provider") or "open-meteo").strip().lower()
    if provider not in {"open-meteo", "custom"}:
        raise ShadowWeatherError("Weather provider must be Open-Meteo or custom.")
    try:
        latitude = float(payload.get("latitude"))
        longitude = float(payload.get("longitude"))
    except (TypeError, ValueError) as exc:
        raise ShadowWeatherError("Weather latitude and longitude are required numbers.") from exc
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ShadowWeatherError("Weather coordinates are outside their valid range.")
    config = {
        "provider": provider,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": str(payload.get("timezone") or "Asia/Bangkok").strip(),
        "custom_url": str(payload.get("custom_url") or "").strip(),
        "api_key_header": str(payload.get("api_key_header") or "").strip(),
        "api_key": str(payload.get("api_key") or "") or str(current.get("api_key") or ""),
    }
    if not config["timezone"]:
        raise ShadowWeatherError("Weather timezone is required.")
    if provider == "custom":
        _validate_custom_url(config["custom_url"])
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CONFIG_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
    temporary.replace(CONFIG_PATH)
    return public_config()


def _validate_custom_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ShadowWeatherError("Custom weather URL must be HTTPS.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except OSError as exc:
        raise ShadowWeatherError(f"Custom weather host cannot be resolved: {exc}") from exc
    for address in addresses:
        if not ipaddress.ip_address(address[4][0]).is_global:
            raise ShadowWeatherError("Custom weather URL cannot target a private or local address.")


def _signature(config: dict[str, Any]) -> str:
    safe = {key: config.get(key) for key in scientific_snapshot()}
    return json.dumps(safe, sort_keys=True, separators=(",", ":"))


def _conn() -> sqlite3.Connection:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS hourly_weather ("
        "source TEXT NOT NULL,time_iso TEXT NOT NULL,values_json TEXT NOT NULL,"
        "PRIMARY KEY(source,time_iso))"
    )
    return conn


def test_connection() -> dict[str, Any]:
    end = date.today() - timedelta(days=7)
    rows = _fetch_range(end, end)
    return {
        "ok": True,
        "provider": _raw_config()["provider"],
        "date": end.isoformat(),
        "hours": len(rows),
        "message": "Real hourly weather provider is ready.",
    }


def ensure_weather(start: date, end: date) -> dict[str, Any]:
    config = _validated_config()
    source = _signature(config)
    expected_dates = []
    cursor = start
    while cursor <= end:
        expected_dates.append(cursor)
        cursor += timedelta(days=1)
    with _conn() as conn:
        existing = {
            row[0][:10]
            for row in conn.execute(
                "SELECT time_iso FROM hourly_weather WHERE source=?", (source,)
            ).fetchall()
        }
    missing_dates = [day for day in expected_dates if day.isoformat() not in existing]
    for range_start, range_end in _date_ranges(missing_dates):
        rows = _fetch_range(range_start, range_end)
        with _conn() as conn:
            for row in rows:
                values = [float(row[field]) for field in HOURLY_FIELDS]
                conn.execute(
                    "INSERT OR REPLACE INTO hourly_weather VALUES (?,?,?)",
                    (source, row["time_iso"], json.dumps(values, separators=(",", ":"))),
                )
    weather = load_weather()
    expected_start = datetime.combine(start, datetime.min.time())
    expected_end = datetime.combine(end, datetime.min.time()) + timedelta(hours=23)
    stamp = expected_start
    missing_hours = []
    while stamp <= expected_end:
        if stamp not in weather:
            missing_hours.append(stamp.isoformat(timespec="minutes"))
        stamp += timedelta(hours=1)
    if missing_hours:
        raise ShadowWeatherError(
            f"Weather provider has incomplete coverage: {len(missing_hours)} hours missing; first {missing_hours[0]}"
        )
    return {"start_date": start.isoformat(), "end_date": end.isoformat(), "hours": len(weather)}


def _date_ranges(days: list[date]) -> list[tuple[date, date]]:
    if not days:
        return []
    ranges = []
    start = previous = days[0]
    for current in days[1:]:
        if current != previous + timedelta(days=1):
            ranges.append((start, previous))
            start = current
        previous = current
    ranges.append((start, previous))
    return ranges


def _validated_config() -> dict[str, Any]:
    config = _raw_config()
    if config.get("latitude") is None or config.get("longitude") is None:
        raise ShadowWeatherError("Save Shadow weather coordinates first.")
    if config["provider"] == "custom":
        _validate_custom_url(config["custom_url"])
    return config


def _fetch_range(start: date, end: date) -> list[dict[str, Any]]:
    config = _validated_config()
    if config["provider"] == "custom":
        return _request_provider(config, config["custom_url"], start, end)
    cutoff = date.today() - timedelta(days=5)
    rows = []
    if start <= cutoff:
        archive_end = min(end, cutoff)
        rows.extend(_request_provider(config, "https://archive-api.open-meteo.com/v1/archive", start, archive_end))
    if end > cutoff:
        recent_start = max(start, cutoff + timedelta(days=1))
        rows.extend(_request_provider(config, "https://api.open-meteo.com/v1/forecast", recent_start, end))
    return rows


def _request_provider(config, endpoint: str, start: date, end: date) -> list[dict[str, Any]]:
    if start > end:
        return []
    headers = {"Accept": "application/json"}
    if config["provider"] == "custom":
        try:
            url = endpoint.format(
                latitude=config["latitude"], longitude=config["longitude"],
                start_date=start.isoformat(), end_date=end.isoformat(),
                timezone=config["timezone"], hourly=",".join(HOURLY_FIELDS),
            )
        except (KeyError, ValueError) as exc:
            raise ShadowWeatherError(f"Invalid custom weather URL template: {exc}") from exc
        _validate_custom_url(url)
        if config["api_key_header"] and config["api_key"]:
            headers[config["api_key_header"]] = config["api_key"]
    else:
        query = urlencode({
            "latitude": config["latitude"], "longitude": config["longitude"],
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "hourly": ",".join(HOURLY_FIELDS), "timezone": config["timezone"],
        })
        url = f"{endpoint}?{query}"
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=(10, 90),
            allow_redirects=config["provider"] == "open-meteo",
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ShadowWeatherError(f"Weather provider failed: {exc}") from exc
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ShadowWeatherError("Weather response must use the Open-Meteo hourly JSON shape.")
    rows = []
    for index, time_iso in enumerate(hourly["time"]):
        row = {"time_iso": str(time_iso)}
        for field in HOURLY_FIELDS:
            values = hourly.get(field)
            if not isinstance(values, list) or index >= len(values) or values[index] is None:
                raise ShadowWeatherError(f"Weather response is missing {field} at {time_iso}.")
            row[field] = float(values[index])
        rows.append(row)
    return rows


def load_weather() -> dict[datetime, np.ndarray]:
    config = _validated_config()
    source = _signature(config)
    with _conn() as conn:
        rows = conn.execute(
            "SELECT time_iso,values_json FROM hourly_weather WHERE source=?", (source,)
        ).fetchall()
    return {
        datetime.fromisoformat(row["time_iso"]).replace(
            minute=0, second=0, microsecond=0, tzinfo=None
        ):
        np.asarray(json.loads(row["values_json"]), dtype=np.float64)
        for row in rows
    }


def attach_live_forecasts(days, checkpoint_meta: dict[str, Any], p_ref_kw: float) -> dict[str, Any]:
    if not days:
        raise ShadowWeatherError("No telemetry days are available for forecast inference.")
    model_path = _model_path(checkpoint_meta)
    first = date.fromisoformat(days[0].date_iso) - timedelta(days=1)
    last = date.fromisoformat(days[-1].date_iso)
    weather_status = ensure_weather(first, last)
    weather = load_weather()
    model = np.load(model_path, allow_pickle=False)
    required = {
        "x_mean", "x_scale", "weights", "y_mean", "weather_pv_mean",
        "weather_pv_scale", "weather_pv_weights", "weather_pv_intercept",
        "exogenous_columns", "p_ref_kw",
    }
    if not required.issubset(model.files):
        raise ShadowWeatherError("Checkpoint forecast model artifact is incomplete.")
    model_ref = float(np.asarray(model["p_ref_kw"]).reshape(-1)[0])
    if abs(model_ref - float(p_ref_kw)) > 1e-6:
        raise ShadowWeatherError(
            f"Forecast model p_ref {model_ref:g} kW does not match policy p_ref {p_ref_kw:g} kW."
        )
    for day in days:
        base = _day_features(day, weather)
        exogenous = np.asarray(model["exogenous_columns"], dtype=np.int64)
        z = (base[:, exogenous] - model["weather_pv_mean"]) / model["weather_pv_scale"]
        current_error = base[:, 3] - (
            z @ model["weather_pv_weights"]
            + float(np.asarray(model["weather_pv_intercept"]).reshape(-1)[0])
        )
        enriched = np.column_stack((base, current_error))
        prediction = np.maximum(
            0.0,
            ((enriched - model["x_mean"]) / model["x_scale"]) @ model["weights"] + model["y_mean"],
        )
        if prediction.shape != (len(day.load), len(ARTIFACT_FIELDS)):
            raise ShadowWeatherError(f"Forecast model returned an invalid shape for {day.date_iso}.")
        day.forecast = prediction / model_ref
    return {**weather_status, "model_artifact": model_path.name, "p_ref_kw": model_ref}


def _model_path(meta: dict[str, Any]) -> Path:
    configured = meta.get("forecast_model_artifact")
    basename = Path(str(configured or "forecast_missing.npz")).name
    candidates = []
    if configured:
        path = Path(str(configured))
        candidates.append(path if path.is_absolute() else BASE_DIR / path)
    candidates.append(FORECAST_DIR / basename)
    base = BASE_DIR.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved == base or base in resolved.parents) and resolved.is_file():
            return resolved
    raise ShadowWeatherError(
        f"Checkpoint forecast model {basename} is missing. Copy its .npz artifact from the training computer."
    )


def _day_features(day, weather: dict[datetime, np.ndarray]) -> np.ndarray:
    n = len(day.pv)
    dt_minutes = 1440.0 / n
    lags = [max(1, round(minutes / dt_minutes)) for minutes in (1, 5, 15)]
    base_date = datetime.fromisoformat(day.date_iso)
    rows = []
    for step in range(n):
        stamp = base_date + timedelta(minutes=step * dt_minutes)
        weather_stamp = stamp.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        current_weather = weather.get(weather_stamp)
        previous_weather = weather.get(weather_stamp - timedelta(hours=1), current_weather)
        if current_weather is None:
            raise ShadowWeatherError(f"Missing real weather at {weather_stamp.isoformat(timespec='minutes')}.")
        pv = float(day.pv[step])
        load = float(day.load[step])
        pv_lag = [float(day.pv[max(0, step - lag)]) for lag in lags]
        roll_start = max(0, step - lags[-1] + 1)
        angle = 2.0 * np.pi * step / n
        rows.append([
            np.sin(angle), np.cos(angle), load, pv,
            *(pv - value for value in pv_lag),
            float(np.mean(day.pv[roll_start:step + 1])),
            float(np.mean(day.load[roll_start:step + 1])),
            (stamp - weather_stamp).total_seconds() / 60.0,
            *current_weather.tolist(),
            *(current_weather - previous_weather).tolist(),
        ])
    return np.asarray(rows, dtype=np.float64)
