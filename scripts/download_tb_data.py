"""Download 1-minute ThingsBoard telemetry into a Sizing_Demo training CSV.

This file is intentionally self-contained.  It imports no code from
``diseep_simulator`` or any other project folder, and it uses only Python's
standard library.

Edit ``SPEC`` and the date constants below, then run this file with the
Sizing_Demo virtual environment.  ``END_DATE`` is inclusive.
"""

from __future__ import annotations

import csv
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from bess.paths import PROJECT_ROOT
from zoneinfo import ZoneInfo


SPEC = {
    "name": "namduoc",
    "base_url": "https://solar.datainsight.vn",
    "username": "oee2024@gmail.com",
    "password": "Oee@2124",
    "device_id": "39ce5a90-84b4-11f0-afa5-2533bc830589",
    "key_load": "INVT_T:PLoad",
    "key_pv": "INVT_T:ActivePowerSum",
    "unit_scale": 1.0,
    "timezone": "Asia/Bangkok",
}

START_DATE = "2025-06-01"
END_DATE = "2026-07-01"
INTERVAL_MINUTES = 1

BASE_DIR = PROJECT_ROOT
OUTPUT_CSV = BASE_DIR / "data" / f"offline_{SPEC['name']}_1min.csv"

TOKEN_TTL_SECONDS = 2 * 60 * 60
MAX_INTERVALS_PER_REQUEST = 650
MAX_INTERPOLATION_GAP_MINUTES = 15
PV_ZERO_THRESHOLD_KW = 1.0
CSV_HEADERS = (
    "day_index",
    "date_iso",
    "day_type",
    "step",
    "P_load_kW",
    "P_pv_kW",
)

_token: str | None = None
_token_expires_at = 0.0


def _error_message(error: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(error.read().decode("utf-8"))
        return str(body.get("message") or body)[:300]
    except Exception:  # noqa: BLE001
        return f"HTTP {error.code}"


def _post_json(url: str, payload: dict, timeout: int = 20) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"ThingsBoard rejected the login ({error.code}): "
            f"{_error_message(error)}"
        ) from error


def _get_json(url: str, jwt: str, timeout: int = 30) -> dict:
    request = urllib.request.Request(
        url,
        headers={"X-Authorization": f"Bearer {jwt}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _login(force: bool = False) -> str:
    global _token, _token_expires_at

    if not force and _token and time.time() < _token_expires_at:
        return _token

    base_url = str(SPEC["base_url"]).rstrip("/")
    response = _post_json(
        f"{base_url}/api/auth/login",
        {
            "username": SPEC["username"],
            "password": SPEC["password"],
        },
    )
    try:
        _token = str(response["token"])
    except KeyError as error:
        raise RuntimeError("ThingsBoard login response did not contain a token") from error
    _token_expires_at = time.time() + TOKEN_TTL_SECONDS
    return _token


def _split_keys(value: object) -> list[str]:
    return [key.strip() for key in str(value or "").split(",") if key.strip()]


def _telemetry_keys() -> list[str]:
    keys = _split_keys(SPEC.get("key_load")) + _split_keys(SPEC.get("key_pv"))
    if not keys:
        raise ValueError("SPEC must define key_load and/or key_pv")
    return keys


def _fetch_chunk(start_ms: int, end_ms: int, interval_ms: int) -> dict:
    query = urllib.parse.urlencode(
        {
            "keys": ",".join(_telemetry_keys()),
            "startTs": start_ms,
            "endTs": end_ms,
            "interval": interval_ms,
            "agg": "AVG",
            "limit": 60000,
            "orderBy": "ASC",
        }
    )
    base_url = str(SPEC["base_url"]).rstrip("/")
    url = (
        f"{base_url}/api/plugins/telemetry/DEVICE/{SPEC['device_id']}"
        f"/values/timeseries?{query}"
    )

    try:
        return _get_json(url, _login())
    except urllib.error.HTTPError as error:
        if error.code == 401:
            return _get_json(url, _login(force=True))
        raise RuntimeError(
            f"ThingsBoard telemetry request failed ({error.code}): "
            f"{_error_message(error)}"
        ) from error


def _format_local_minute(timestamp_ms: int, timezone: ZoneInfo) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone).strftime(
        "%Y-%m-%d %H:%M"
    )


def _format_point_counts(keys: list[str], counts: dict[str, int]) -> str:
    return " | ".join(f"{key}={counts.get(key, 0)}" for key in keys)


def _fetch_range(start_iso: str, end_iso: str, interval_min: int) -> dict:
    start_day = date.fromisoformat(start_iso)
    end_day = date.fromisoformat(end_iso)
    yesterday = date.today() - timedelta(days=1)
    if end_day > yesterday:
        end_day = yesterday
        print(f"[clamp] End date moved back to {end_day}; today is incomplete.")
    if start_day > end_day:
        raise ValueError(f"Empty date range: {start_day} > {end_day}")

    timezone = ZoneInfo(str(SPEC.get("timezone") or "Asia/Bangkok"))
    start_ms = int(
        datetime.combine(start_day, datetime.min.time(), timezone).timestamp() * 1000
    )
    exclusive_end_ms = int(
        datetime.combine(
            end_day + timedelta(days=1),
            datetime.min.time(),
            timezone,
        ).timestamp()
        * 1000
    )
    interval_ms = interval_min * 60_000
    chunk_ms = MAX_INTERVALS_PER_REQUEST * interval_ms
    total_chunks = math.ceil((exclusive_end_ms - start_ms) / chunk_ms)
    telemetry_keys = list(dict.fromkeys(_telemetry_keys()))
    merged: dict[str, list] = defaultdict(list)
    total_points = {key: 0 for key in telemetry_keys}
    nonempty_chunks = 0
    empty_chunks = 0
    empty_run_start_ms: int | None = None
    empty_run_chunks = 0

    print(
        f"ThingsBoard {SPEC['base_url']} device "
        f"{str(SPEC['device_id'])[:8]}... [{start_day} -> {end_day}]"
    )
    cursor = start_ms
    chunk_number = 0
    while cursor < exclusive_end_ms:
        chunk_number += 1
        chunk_end = min(cursor + chunk_ms, exclusive_end_ms)
        part = _fetch_chunk(cursor, chunk_end, interval_ms)
        for key, values in part.items():
            merged[key].extend(values)

        chunk_counts = {
            key: len(part.get(key, []))
            for key in telemetry_keys
        }
        for key, count in chunk_counts.items():
            total_points[key] += count

        point_count = sum(len(values) for values in part.values())
        if point_count:
            if empty_run_start_ms is not None:
                print(
                    f"[gap] {empty_run_chunks} empty chunks | "
                    f"{_format_local_minute(empty_run_start_ms, timezone)} -> "
                    f"{_format_local_minute(cursor, timezone)}"
                )
                empty_run_start_ms = None
                empty_run_chunks = 0

            nonempty_chunks += 1
            progress = chunk_number / total_chunks * 100.0
            print(
                f"[{chunk_number}/{total_chunks} | {progress:.1f}%] "
                f"{_format_local_minute(cursor, timezone)} -> "
                f"{_format_local_minute(chunk_end, timezone)} | "
                f"{_format_point_counts(telemetry_keys, chunk_counts)}"
            )
        else:
            empty_chunks += 1
            if empty_run_start_ms is None:
                empty_run_start_ms = cursor
            empty_run_chunks += 1

        cursor = chunk_end

    if empty_run_start_ms is not None:
        print(
            f"[gap] {empty_run_chunks} empty chunks | "
            f"{_format_local_minute(empty_run_start_ms, timezone)} -> "
            f"{_format_local_minute(exclusive_end_ms, timezone)}"
        )

    print(
        f"Download summary: {total_chunks} chunks | "
        f"{nonempty_chunks} nonempty | {empty_chunks} empty | "
        f"{_format_point_counts(telemetry_keys, total_points)}"
    )

    if not merged:
        raise ValueError("ThingsBoard returned no telemetry for the selected range")
    return dict(merged)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _channel_values(
    telemetry: dict,
    key_spec: object,
    interval_ms: int,
    scale: float,
) -> dict[int, float]:
    keys = _split_keys(key_spec)
    if not keys:
        return {}

    channel: dict[int, float] = defaultdict(float)
    for key in keys:
        points = telemetry.get(key, [])
        if not points:
            raise ValueError(f"ThingsBoard returned no telemetry for key '{key}'")

        values_by_bucket: dict[int, list[float]] = defaultdict(list)
        for point in points:
            timestamp_ms = int(point["ts"])
            bucket_ms = timestamp_ms - (timestamp_ms % interval_ms)
            values_by_bucket[bucket_ms].append(float(point["value"]) * scale)
        for bucket_ms, values in values_by_bucket.items():
            channel[bucket_ms] += _mean(values)
    return dict(channel)


def _interpolate(
    values: list[float | None],
    max_gap_steps: int,
    zero_threshold: float | None = None,
) -> list[float] | None:
    known = [index for index, value in enumerate(values) if value is not None]
    if not known:
        return None

    result: list[float | None] = list(values)
    first = known[0]
    first_value = float(values[first])
    if first > max_gap_steps and not (
        zero_threshold is not None and abs(first_value) <= zero_threshold
    ):
        return None
    edge_value = 0.0 if (
        zero_threshold is not None and abs(first_value) <= zero_threshold
    ) else first_value
    for index in range(first):
        result[index] = edge_value

    for left, right in zip(known, known[1:]):
        left_value = float(values[left])
        right_value = float(values[right])
        missing_steps = right - left - 1
        zero_gap = (
            zero_threshold is not None
            and abs(left_value) <= zero_threshold
            and abs(right_value) <= zero_threshold
        )
        if missing_steps > max_gap_steps and not zero_gap:
            return None
        width = right - left
        for index in range(left + 1, right):
            fraction = (index - left) / width
            result[index] = left_value + (right_value - left_value) * fraction

    last = known[-1]
    last_value = float(values[last])
    trailing_steps = len(values) - last - 1
    if trailing_steps > max_gap_steps and not (
        zero_threshold is not None and abs(last_value) <= zero_threshold
    ):
        return None
    edge_value = 0.0 if (
        zero_threshold is not None and abs(last_value) <= zero_threshold
    ) else last_value
    for index in range(last + 1, len(values)):
        result[index] = edge_value

    return [float(value) for value in result]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _despike_load(load: list[float], max_run: int) -> tuple[list[float], int]:
    cleaned = list(load)
    baseline = _percentile(cleaned, 0.90)
    threshold = max(5.0 * baseline, 50.0)
    multi_threshold = max(8.0 * baseline, 100.0)
    changed = 0
    index = 0

    while index < len(cleaned):
        if cleaned[index] <= threshold:
            index += 1
            continue
        end = index
        while end < len(cleaned) and cleaned[end] > threshold:
            end += 1
        run_length = end - index
        if (
            run_length > 1
            and max(cleaned[index:end]) <= multi_threshold
        ):
            index = end
            continue
        if run_length <= max_run:
            left = cleaned[index - 1] if index > 0 else (
                cleaned[end] if end < len(cleaned) else baseline
            )
            right = cleaned[end] if end < len(cleaned) else left
            for offset in range(run_length):
                cleaned[index + offset] = (
                    left
                    + (right - left) * (offset + 1) / (run_length + 1)
                )
            changed += run_length
        index = end
    return cleaned, changed


def _build_days(
    telemetry: dict,
    interval_min: int,
) -> tuple[list[dict], int, int, int]:
    interval_ms = interval_min * 60_000
    scale = float(SPEC.get("unit_scale", 1.0))
    load_by_time = _channel_values(
        telemetry,
        SPEC.get("key_load"),
        interval_ms,
        scale,
    )
    pv_by_time = _channel_values(
        telemetry,
        SPEC.get("key_pv"),
        interval_ms,
        scale,
    )
    timezone = ZoneInfo(str(SPEC.get("timezone") or "Asia/Bangkok"))
    samples_by_day: dict[date, dict[int, tuple[float | None, float | None]]] = (
        defaultdict(dict)
    )

    for timestamp_ms in load_by_time.keys() | pv_by_time.keys():
        local_time = datetime.fromtimestamp(timestamp_ms / 1000, timezone)
        step = (local_time.hour * 60 + local_time.minute) // interval_min
        samples_by_day[local_time.date()][step] = (
            load_by_time.get(timestamp_ms),
            pv_by_time.get(timestamp_ms),
        )

    steps_per_day = 24 * 60 // interval_min
    max_gap_steps = MAX_INTERPOLATION_GAP_MINUTES // interval_min
    days = []
    bad_sensor_days = 0
    incomplete_sensor_days = 0
    despiked_points = 0
    for calendar_day in sorted(samples_by_day):
        samples = samples_by_day[calendar_day]
        load = _interpolate(
            [samples.get(step, (None, None))[0] for step in range(steps_per_day)],
            max_gap_steps,
        )
        if load is None:
            incomplete_sensor_days += 1
            print(
                f"[skip] {calendar_day}: load telemetry has a gap longer than "
                f"{MAX_INTERPOLATION_GAP_MINUTES} minutes or misses a day edge."
            )
            continue

        pv = _interpolate(
            [samples.get(step, (None, None))[1] for step in range(steps_per_day)],
            max_gap_steps,
            zero_threshold=PV_ZERO_THRESHOLD_KW,
        )
        if pv is None:
            incomplete_sensor_days += 1
            print(
                f"[skip] {calendar_day}: PV telemetry has a nonzero gap longer "
                f"than {MAX_INTERPOLATION_GAP_MINUTES} minutes or misses a "
                "nonzero day edge."
            )
            continue

        dead_load_fraction = sum(value < 1.0 for value in load) / steps_per_day
        if dead_load_fraction > 0.98 and max(pv) > 50.0:
            bad_sensor_days += 1
            continue

        load, changed = _despike_load(
            load,
            max_run=(2 * 60) // interval_min,
        )
        despiked_points += changed
        days.append(
            {
                "date": calendar_day,
                "day_type": (
                    "weekend" if calendar_day.weekday() >= 5 else "working"
                ),
                "load": load,
                "pv": pv,
            }
        )

    if not days:
        raise ValueError("No usable telemetry days were found")
    return days, bad_sensor_days, incomplete_sensor_days, despiked_points


def _write_csv(days: list[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for day_index, day in enumerate(days, start=1):
            for step, (load, pv) in enumerate(zip(day["load"], day["pv"])):
                writer.writerow(
                    {
                        "day_index": day_index,
                        "date_iso": day["date"].isoformat(),
                        "day_type": day["day_type"],
                        "step": step,
                        "P_load_kW": load,
                        "P_pv_kW": pv,
                    }
                )
                row_count += 1
    return row_count


def main() -> None:
    telemetry = _fetch_range(START_DATE, END_DATE, INTERVAL_MINUTES)
    days, bad_sensor_days, incomplete_sensor_days, despiked_points = _build_days(
        telemetry,
        INTERVAL_MINUTES,
    )
    row_count = _write_csv(days, OUTPUT_CSV)
    print(
        f"Saved {row_count} rows ({len(days)} days) to {OUTPUT_CSV}"
        f"; skipped {bad_sensor_days} dead-sensor days"
        f"; skipped {incomplete_sensor_days} incomplete-sensor days"
        f"; repaired {despiked_points} load-spike points."
    )


if __name__ == "__main__":
    main()
