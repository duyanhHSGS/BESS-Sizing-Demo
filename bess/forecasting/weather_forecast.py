from __future__ import annotations

import csv
import ipaddress
import json
import socket
from datetime import datetime, timedelta
from pathlib import Path

from bess.paths import PROJECT_ROOT
from urllib.parse import urlencode, urlparse

import numpy as np
import requests

from bess.training.training_datasets import DatasetError, get_dataset_path, sanitize_dataset_id


BASE_DIR = PROJECT_ROOT
WEATHER_DIR = BASE_DIR / "user_data" / "weather"
FORECAST_DIR = BASE_DIR / "user_data" / "forecasts"
HOURLY_FIELDS = (
    "temperature_2m",
    "cloud_cover",
    "precipitation",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
)
ARTIFACT_FIELDS = (
    "fc_eff_next_1h",
    "fc_pv_next_1h",
    "fc_eff_following_2h",
    "fc_pv_following_2h",
)


class WeatherError(ValueError):
    pass


def weather_path(dataset_id: str) -> Path:
    return WEATHER_DIR / f"weather_{sanitize_dataset_id(dataset_id)}.csv"


def weather_meta_path(dataset_id: str) -> Path:
    return WEATHER_DIR / f"weather_{sanitize_dataset_id(dataset_id)}.json"


def forecast_artifact_path(tag: str) -> Path:
    return FORECAST_DIR / f"forecast_{sanitize_dataset_id(tag)}.csv"


def _dataset_dates(dataset_id: str) -> tuple[Path, list[str]]:
    path = get_dataset_path(dataset_id)
    dates = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "date_iso" not in (reader.fieldnames or []):
            raise DatasetError(
                f"{path.name} has no real date_iso column; weather cannot be aligned honestly"
            )
        for row in reader:
            value = str(row.get("date_iso") or "").strip()
            if value and (not dates or dates[-1] != value):
                datetime.fromisoformat(value)
                dates.append(value)
    if not dates:
        raise DatasetError(f"{path.name} has no dated rows")
    return path, dates


def _validate_remote_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise WeatherError("Custom weather URL must be an https URL")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except OSError as exc:
        raise WeatherError(f"Custom weather host cannot be resolved: {exc}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise WeatherError("Custom weather URL cannot target a private or local address")
    return url


def _open_meteo_url(latitude: float, longitude: float, start: str, end: str,
                    timezone: str) -> str:
    query = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": timezone,
    })
    return f"https://archive-api.open-meteo.com/v1/archive?{query}"


def _normalise_hourly(payload: dict) -> list[dict]:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise WeatherError("Weather API response must use the Open-Meteo hourly JSON shape")
    times = hourly["time"]
    rows = []
    for index, time_iso in enumerate(times):
        row = {"time_iso": str(time_iso)}
        for field in HOURLY_FIELDS:
            values = hourly.get(field)
            if not isinstance(values, list) or index >= len(values) or values[index] is None:
                raise WeatherError(f"Weather response is missing {field} at {time_iso}")
            value = float(values[index])
            if not np.isfinite(value):
                raise WeatherError(f"Weather response has non-finite {field} at {time_iso}")
            row[field] = value
        rows.append(row)
    return rows


def fetch_weather(payload: dict) -> dict:
    dataset_id = str(payload.get("dataset_id") or "").strip()
    _, dates = _dataset_dates(dataset_id)
    fetch_start = (datetime.fromisoformat(dates[0]) - timedelta(days=1)).date().isoformat()
    latitude = float(payload.get("latitude"))
    longitude = float(payload.get("longitude"))
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise WeatherError("Latitude or longitude is outside its valid range")
    timezone = str(payload.get("timezone") or "Asia/Ho_Chi_Minh").strip()
    provider = str(payload.get("provider") or "open-meteo").strip().lower()
    headers = {"Accept": "application/json"}
    if provider == "open-meteo":
        url = _open_meteo_url(latitude, longitude, fetch_start, dates[-1], timezone)
    elif provider == "custom":
        template = str(payload.get("custom_url") or "").strip()
        url = template.format(
            latitude=latitude,
            longitude=longitude,
            start_date=fetch_start,
            end_date=dates[-1],
            timezone=timezone,
            hourly=",".join(HOURLY_FIELDS),
        )
        url = _validate_remote_url(url)
        header_name = str(payload.get("api_key_header") or "").strip()
        api_key = str(payload.get("api_key") or "")
        if header_name and api_key:
            headers[header_name] = api_key
    else:
        raise WeatherError("Provider must be open-meteo or custom")
    response = requests.get(
        url,
        headers=headers,
        timeout=(10, 90),
        allow_redirects=provider == "open-meteo",
    )
    response.raise_for_status()
    rows = _normalise_hourly(response.json())
    expected_start = datetime.fromisoformat(fetch_start)
    expected_end = datetime.fromisoformat(dates[-1]) + timedelta(hours=23)
    covered = {datetime.fromisoformat(row["time_iso"]).replace(minute=0, second=0, microsecond=0) for row in rows}
    cursor = expected_start
    missing = []
    while cursor <= expected_end:
        if cursor not in covered:
            missing.append(cursor.isoformat(timespec="minutes"))
        cursor += timedelta(hours=1)
    if missing:
        raise WeatherError(
            f"Weather API returned incomplete hourly coverage: {len(missing)} missing hours; first {missing[0]}"
        )
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    output = weather_path(dataset_id)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("time_iso", *HOURLY_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "dataset_id": sanitize_dataset_id(dataset_id),
        "provider": provider,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "start_date": dates[0],
        "end_date": dates[-1],
        "weather_start_date": fetch_start,
        "hours": len(rows),
        "downloaded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "weather_file": str(output),
    }
    weather_meta_path(dataset_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def weather_status(dataset_id: str) -> dict:
    _, dates = _dataset_dates(dataset_id)
    data_path = weather_path(dataset_id)
    meta_path = weather_meta_path(dataset_id)
    if not data_path.is_file() or not meta_path.is_file():
        return {"ready": False, "dataset_id": sanitize_dataset_id(dataset_id),
                "start_date": dates[0], "end_date": dates[-1],
                "message": "No real weather file has been downloaded for this dataset."}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ready = meta.get("start_date") <= dates[0] and meta.get("end_date") >= dates[-1]
    return {**meta, "ready": bool(ready),
            "message": "Real hourly weather is ready." if ready else "Weather coverage does not match the dataset."}


def _load_weather(path: Path) -> dict[datetime, np.ndarray]:
    values = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime.fromisoformat(row["time_iso"]).replace(minute=0, second=0, microsecond=0)
            values[stamp] = np.asarray([float(row[field]) for field in HOURLY_FIELDS], dtype=np.float64)
    return values


def _samples(days, weather: dict[datetime, np.ndarray]):
    features = []
    targets = []
    owners = []
    for day_index, day in enumerate(days):
        if not day.date_iso:
            raise WeatherError("Forecast mode requires real date_iso values on every day")
        n = len(day.pv)
        dt_minutes = 1440.0 / n
        hour_steps = max(1, round(60.0 / dt_minutes))
        lags = [max(1, round(minutes / dt_minutes)) for minutes in (1, 5, 15)]
        base_date = datetime.fromisoformat(day.date_iso)
        x_day = []
        y_day = []
        for step in range(n):
            stamp = base_date + timedelta(minutes=step * dt_minutes)
            # Use the most recently completed hour. Historical hourly fields
            # can aggregate within their named hour, so this one-hour shift
            # prevents future-within-the-hour leakage.
            weather_stamp = stamp.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            current_weather = weather.get(weather_stamp)
            previous_weather = weather.get(weather_stamp - timedelta(hours=1))
            if current_weather is None:
                raise WeatherError(f"Missing real weather at {weather_stamp.isoformat(timespec='minutes')}")
            if previous_weather is None:
                previous_weather = current_weather
            pv = float(day.pv[step])
            load = float(day.load[step])
            pv_lag = [float(day.pv[max(0, step - lag)]) for lag in lags]
            roll_start = max(0, step - lags[-1] + 1)
            angle = 2.0 * np.pi * step / n
            x = [
                np.sin(angle), np.cos(angle), load, pv,
                *(pv - value for value in pv_lag),
                float(np.mean(day.pv[roll_start:step + 1])),
                float(np.mean(day.load[roll_start:step + 1])),
                (stamp - weather_stamp).total_seconds() / 60.0,
                *current_weather.tolist(),
                *(current_weather - previous_weather).tolist(),
            ]
            a0, a1 = step + 1, min(n, step + 1 + hour_steps)
            b0, b1 = a1, min(n, step + 1 + 3 * hour_steps)
            if a0 >= a1:
                first_load, first_pv = load, pv
            else:
                first_load = float(np.mean(day.load[a0:a1]))
                first_pv = float(np.mean(day.pv[a0:a1]))
            if b0 >= b1:
                second_load, second_pv = first_load, first_pv
            else:
                second_load = float(np.mean(day.load[b0:b1]))
                second_pv = float(np.mean(day.pv[b0:b1]))
            x_day.append(x)
            y_day.append([
                max(0.0, first_load - first_pv), first_pv,
                max(0.0, second_load - second_pv), second_pv,
            ])
        features.append(np.asarray(x_day, dtype=np.float64))
        targets.append(np.asarray(y_day, dtype=np.float64))
        owners.append(day_index)
    return features, targets, owners


def fit_attach_forecasts(days, weather_file: Path, train_day_count: int,
                         artifact: Path, p_ref_kw: float) -> dict:
    if train_day_count < 1 or train_day_count >= len(days):
        raise WeatherError("Forecast model needs training days and held-out days")
    features, targets, _ = _samples(days, _load_weather(weather_file))
    x_train_base = np.concatenate(features[:train_day_count], axis=0)
    y_train = np.concatenate(targets[:train_day_count], axis=0)
    # A causal "forecast error now" signal: predict current PV from time,
    # load, weather, and weather trends, then compare with measured current PV.
    # Current PV itself and PV-derived lag columns are excluded from this
    # auxiliary fit so the residual cannot collapse to a copy of its target.
    exogenous_columns = [0, 1, 2, *range(9, x_train_base.shape[1])]
    z_train = x_train_base[:, exogenous_columns]
    z_mean = z_train.mean(axis=0)
    z_scale = z_train.std(axis=0)
    z_scale[z_scale < 1e-9] = 1.0
    zs = (z_train - z_mean) / z_scale
    pv_now = x_train_base[:, 3]
    pv_now_mean = pv_now.mean()
    pv_weights = np.linalg.solve(
        zs.T @ zs + np.eye(zs.shape[1]),
        zs.T @ (pv_now - pv_now_mean),
    )
    enriched = []
    for x in features:
        z = (x[:, exogenous_columns] - z_mean) / z_scale
        current_error = x[:, 3] - (z @ pv_weights + pv_now_mean)
        enriched.append(np.column_stack((x, current_error)))
    features = enriched
    x_train = np.concatenate(features[:train_day_count], axis=0)
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale < 1e-9] = 1.0
    y_mean = y_train.mean(axis=0)
    xs = (x_train - x_mean) / x_scale
    ridge = 1.0
    gram = xs.T @ xs + ridge * np.eye(xs.shape[1])
    weights = np.linalg.solve(gram, xs.T @ (y_train - y_mean))
    predictions = [np.maximum(0.0, ((x - x_mean) / x_scale) @ weights + y_mean) for x in features]
    for day, prediction in zip(days, predictions):
        day.forecast = prediction / float(p_ref_kw)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("date_iso", "day_index", "step", *ARTIFACT_FIELDS))
        writer.writeheader()
        for day, prediction in zip(days, predictions):
            for step, row in enumerate(prediction):
                writer.writerow({"date_iso": day.date_iso, "day_index": day.day_index, "step": step,
                                 **dict(zip(ARTIFACT_FIELDS, row))})
    model_artifact = artifact.with_suffix(".npz")
    np.savez_compressed(
        model_artifact,
        x_mean=x_mean,
        x_scale=x_scale,
        weights=weights,
        y_mean=y_mean,
        weather_pv_mean=z_mean,
        weather_pv_scale=z_scale,
        weather_pv_weights=pv_weights,
        weather_pv_intercept=np.asarray([pv_now_mean]),
        exogenous_columns=np.asarray(exogenous_columns, dtype=np.int64),
        p_ref_kw=np.asarray([p_ref_kw], dtype=np.float64),
    )
    try:
        artifact_reference = str(artifact.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        artifact_reference = str(artifact.resolve())
    try:
        model_reference = str(model_artifact.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        model_reference = str(model_artifact.resolve())
    return {"model": "causal_weather_ridge_v1", "features": int(x_train.shape[1]),
            "training_rows": int(x_train.shape[0]), "artifact": artifact_reference,
            "model_artifact": model_reference}


def attach_forecast_artifact(days, artifact: Path, p_ref_kw: float) -> None:
    by_key = {}
    with artifact.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["date_iso"]), int(row["step"]))
            by_key[key] = [float(row[field]) / float(p_ref_kw) for field in ARTIFACT_FIELDS]
    for day in days:
        rows = []
        for step in range(len(day.pv)):
            key = (str(day.date_iso), step)
            if key not in by_key:
                raise WeatherError(f"Forecast artifact has no row for {key[0]} step {step}")
            rows.append(by_key[key])
        day.forecast = np.asarray(rows, dtype=np.float64)


def build_forecast_bundle(days) -> dict:
    """Create a portable, already-normalized forecast payload for a .pt file."""
    dates = []
    day_indices = []
    lengths = []
    values = []
    for day in days:
        forecast = getattr(day, "forecast", None)
        expected = (len(day.pv), 4)
        if forecast is None or np.asarray(forecast).shape != expected:
            raise WeatherError(f"Cannot package forecast for {day.date_iso}: expected {expected}")
        dates.append(str(day.date_iso))
        day_indices.append(int(day.day_index))
        lengths.append(len(day.pv))
        values.append(np.asarray(forecast, dtype=np.float32))
    return {
        "version": 1,
        "dates": dates,
        "day_indices": day_indices,
        "lengths": lengths,
        "values": np.concatenate(values, axis=0),
    }


def attach_forecast_bundle(days, bundle: dict) -> None:
    if not isinstance(bundle, dict) or int(bundle.get("version", 0)) != 1:
        raise WeatherError("Checkpoint forecast bundle is missing or unsupported")
    raw_values = bundle.get("values")
    if hasattr(raw_values, "detach"):
        raw_values = raw_values.detach().cpu().numpy()
    values = np.asarray(raw_values, dtype=np.float64)
    dates = list(bundle.get("dates") or [])
    lengths = [int(value) for value in (bundle.get("lengths") or [])]
    if len(dates) != len(lengths) or values.ndim != 2 or values.shape[1] != 4:
        raise WeatherError("Checkpoint forecast bundle has an invalid layout")
    by_date = {}
    offset = 0
    for date_iso, length in zip(dates, lengths):
        by_date[str(date_iso)] = values[offset:offset + length]
        offset += length
    if offset != len(values):
        raise WeatherError("Checkpoint forecast bundle length does not match its metadata")
    for day in days:
        forecast = by_date.get(str(day.date_iso))
        if forecast is None or len(forecast) != len(day.pv):
            raise WeatherError(
                f"Checkpoint forecast bundle does not cover {day.date_iso} at this resolution"
            )
        day.forecast = forecast.copy()
