"""Retry missing ThingsBoard calendar days without inventing telemetry.

This utility starts from the already-clean CSV produced by
``scripts/download_tb_data.py``. It finds calendar dates missing from the
configured requested range, retries only those days against ThingsBoard, runs
recovered telemetry through the same cleaner, and writes a separate recovered
CSV. Unrecoverable days stay missing on purpose.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from scripts import download_tb_data as tb
except ModuleNotFoundError:  # direct ``python scripts/recover_tb_missing_days.py``
    import download_tb_data as tb

SOURCE_CSV = tb.OUTPUT_CSV
RECOVERED_CSV = SOURCE_CSV.with_name(f"{SOURCE_CSV.stem}_recovered.csv")
FINE_RECOVERY_INTERVAL_MINUTES = 5

# TODO(RECOVER-DATA): after the first Youngone recovery run, keep a short report
# of which dates were recovered versus truly absent so future training uses only
# measured calendar days and never synthetic long-gap interpolation.


def _effective_requested_range() -> tuple[date, date]:
    start_day = date.fromisoformat(tb.START_DATE)
    end_day = date.fromisoformat(tb.END_DATE)
    timezone = ZoneInfo(str(tb.SPEC.get("timezone") or "Asia/Bangkok"))
    yesterday = datetime.now(timezone).date() - timedelta(days=1)
    return start_day, min(end_day, yesterday)


def _load_existing_days(path: Path, interval_min: int) -> list[dict]:
    expected_steps = 24 * 60 // interval_min
    grouped: dict[date, dict[int, tuple[float, float, str]]] = defaultdict(dict)

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"date_iso", "day_type", "step", "P_load_kW", "P_pv_kW"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            calendar_day = date.fromisoformat(row["date_iso"])
            step = int(row["step"])
            if step in grouped[calendar_day]:
                raise ValueError(f"{path.name} duplicates {calendar_day} step {step}")
            grouped[calendar_day][step] = (
                float(row["P_load_kW"]),
                float(row["P_pv_kW"]),
                row["day_type"],
            )

    days: list[dict] = []
    expected_step_set = set(range(expected_steps))
    for calendar_day in sorted(grouped):
        samples = grouped[calendar_day]
        if set(samples) != expected_step_set:
            raise ValueError(
                f"{path.name} has incomplete existing day {calendar_day}: "
                f"{len(samples)}/{expected_steps} steps"
            )
        day_types = {sample[2] for sample in samples.values()}
        if len(day_types) != 1:
            raise ValueError(f"{path.name} has mixed day_type values on {calendar_day}")

        load = [samples[step][0] for step in range(expected_steps)]
        pv = [samples[step][1] for step in range(expected_steps)]
        quality_issue = tb._sensor_quality_issue(load, pv, interval_min)
        if quality_issue is not None:
            raise ValueError(
                f"{path.name} existing day {calendar_day} fails cleaner: {quality_issue}"
            )
        days.append(
            {
                "date": calendar_day,
                "day_type": next(iter(day_types)),
                "load": load,
                "pv": pv,
            }
        )
    return days


def _missing_dates(existing_days: list[dict], start_day: date, end_day: date) -> list[date]:
    present = {day["date"] for day in existing_days}
    result: list[date] = []
    cursor = start_day
    while cursor <= end_day:
        if cursor not in present:
            result.append(cursor)
        cursor += timedelta(days=1)
    return result


def _day_window_ms(calendar_day: date) -> tuple[int, int]:
    timezone = ZoneInfo(str(tb.SPEC.get("timezone") or "Asia/Bangkok"))
    start = datetime.combine(calendar_day, datetime.min.time(), timezone)
    end = datetime.combine(calendar_day + timedelta(days=1), datetime.min.time(), timezone)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000) - 1


def _fetch_day(calendar_day: date, interval_min: int) -> dict:
    start_ms, end_ms = _day_window_ms(calendar_day)
    return tb._fetch_chunk(start_ms, end_ms, interval_min * 60_000)


def _clean_exact_day(telemetry: dict, calendar_day: date, interval_min: int) -> dict | None:
    if not telemetry:
        return None
    try:
        days, _bad, _incomplete, _despiked = tb._build_days(telemetry, interval_min)
    except ValueError:
        return None
    for day in days:
        if day["date"] == calendar_day:
            return day
    return None


def _recover_one_day(
    calendar_day: date,
    target_interval_min: int,
    fetch_day: Callable[[date, int], dict] = _fetch_day,
) -> tuple[dict | None, int | None]:
    intervals = (target_interval_min, FINE_RECOVERY_INTERVAL_MINUTES)
    tried: set[int] = set()
    for query_interval in intervals:
        if query_interval in tried:
            continue
        tried.add(query_interval)
        telemetry = fetch_day(calendar_day, query_interval)
        recovered = _clean_exact_day(telemetry, calendar_day, target_interval_min)
        if recovered is not None:
            return recovered, query_interval
    return None, None


def _merge_days(existing_days: list[dict], recovered_days: list[dict]) -> list[dict]:
    merged = {day["date"]: day for day in existing_days}
    duplicates = sorted(day["date"] for day in recovered_days if day["date"] in merged)
    if duplicates:
        raise ValueError(f"Recovery attempted to overwrite existing day(s): {duplicates}")
    for day in recovered_days:
        merged[day["date"]] = day
    return [merged[calendar_day] for calendar_day in sorted(merged)]


def _date_ranges(days: list[date]) -> list[tuple[date, date]]:
    if not days:
        return []
    ordered = sorted(set(days))
    ranges: list[tuple[date, date]] = []
    start = previous = ordered[0]
    for calendar_day in ordered[1:]:
        if calendar_day == previous + timedelta(days=1):
            previous = calendar_day
            continue
        ranges.append((start, previous))
        start = previous = calendar_day
    ranges.append((start, previous))
    return ranges


def main() -> None:
    start_day, end_day = _effective_requested_range()
    existing_days = _load_existing_days(SOURCE_CSV, tb.INTERVAL_MINUTES)
    missing = _missing_dates(existing_days, start_day, end_day)
    print(
        f"Recovery scan [{start_day} -> {end_day}] | "
        f"{len(existing_days)} existing clean days | {len(missing)} missing days"
    )

    recovered_days: list[dict] = []
    unrecoverable: list[date] = []
    recovered_by_interval: dict[int, int] = defaultdict(int)

    for index, calendar_day in enumerate(missing, start=1):
        recovered, query_interval = _recover_one_day(calendar_day, tb.INTERVAL_MINUTES)
        if recovered is None or query_interval is None:
            unrecoverable.append(calendar_day)
            print(f"[recover {index}/{len(missing)}] {calendar_day}: UNRECOVERABLE")
            continue
        recovered_days.append(recovered)
        recovered_by_interval[query_interval] += 1
        print(
            f"[recover {index}/{len(missing)}] {calendar_day}: recovered "
            f"using {query_interval}-minute query"
        )

    merged_days = _merge_days(existing_days, recovered_days)
    row_count = tb._write_csv(merged_days, RECOVERED_CSV)

    print(
        f"Saved {row_count} rows ({len(merged_days)} days) to {RECOVERED_CSV}; "
        f"recovered {len(recovered_days)}/{len(missing)} missing days."
    )
    if recovered_by_interval:
        detail = ", ".join(
            f"{interval}min={count}"
            for interval, count in sorted(recovered_by_interval.items(), reverse=True)
        )
        print(f"Recovery sources: {detail}")
    if unrecoverable:
        print("Unrecoverable calendar ranges (left missing; no fake fill):")
        for first, last in _date_ranges(unrecoverable):
            span = (last - first).days + 1
            label = str(first) if first == last else f"{first} -> {last}"
            print(f"  {label} ({span} day{'s' if span != 1 else ''})")


if __name__ == "__main__":
    main()
