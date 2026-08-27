from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts import download_tb_data as tb
from scripts.recover_tb_missing_days import (
    FINE_RECOVERY_INTERVAL_MINUTES,
    _date_ranges,
    _load_existing_days,
    _merge_days,
    _missing_dates,
    _recover_one_day,
)

TARGET_INTERVAL_MINUTES = 15
SAMPLES_PER_DAY = 24 * 60 // TARGET_INTERVAL_MINUTES


def _clean_day(calendar_day: date, base_load: float = 100.0) -> dict:
    load = [base_load + (step % 12) * 2.0 for step in range(SAMPLES_PER_DAY)]
    pv = [max(0.0, 180.0 - abs(step - 48) * 8.0) for step in range(SAMPLES_PER_DAY)]
    return {
        "date": calendar_day,
        "day_type": "weekend" if calendar_day.weekday() >= 5 else "working",
        "load": load,
        "pv": pv,
    }


def _telemetry(calendar_day: date, interval_min: int) -> dict:
    timezone = ZoneInfo(str(tb.SPEC.get("timezone") or "Asia/Bangkok"))
    start = datetime.combine(calendar_day, datetime.min.time(), timezone)
    steps = 24 * 60 // interval_min
    load_key = str(tb.SPEC["key_load"])
    pv_key = str(tb.SPEC["key_pv"])
    load_points: list[dict] = []
    pv_points: list[dict] = []
    for step in range(steps):
        timestamp = start + timedelta(minutes=step * interval_min)
        ts = int(timestamp.timestamp() * 1000)
        load_points.append({"ts": ts, "value": 100.0 + (step % 12) * 2.0})
        pv_points.append({"ts": ts, "value": max(0.0, 180.0 - abs(step - steps // 2) * 2.0)})
    return {load_key: load_points, pv_key: pv_points}


def test_missing_dates_finds_edge_and_internal_holes() -> None:
    existing = [
        _clean_day(date(2026, 1, 2)),
        _clean_day(date(2026, 1, 4)),
    ]

    assert _missing_dates(existing, date(2026, 1, 1), date(2026, 1, 5)) == [
        date(2026, 1, 1),
        date(2026, 1, 3),
        date(2026, 1, 5),
    ]


def test_missing_dates_returns_empty_for_complete_range() -> None:
    existing = [_clean_day(date(2026, 1, day)) for day in range(1, 4)]

    assert _missing_dates(existing, date(2026, 1, 1), date(2026, 1, 3)) == []


def test_recover_one_day_uses_target_interval_when_clean() -> None:
    calendar_day = date(2026, 1, 10)
    calls: list[int] = []

    def fetcher(day: date, interval_min: int) -> dict:
        assert day == calendar_day
        calls.append(interval_min)
        return _telemetry(day, interval_min)

    recovered, query_interval = _recover_one_day(
        calendar_day,
        TARGET_INTERVAL_MINUTES,
        fetcher,
    )

    assert recovered is not None
    assert recovered["date"] == calendar_day
    assert len(recovered["load"]) == SAMPLES_PER_DAY
    assert query_interval == TARGET_INTERVAL_MINUTES
    assert calls == [TARGET_INTERVAL_MINUTES]


def test_recover_one_day_falls_back_to_finer_query() -> None:
    calendar_day = date(2026, 1, 10)
    calls: list[int] = []

    def fetcher(day: date, interval_min: int) -> dict:
        calls.append(interval_min)
        if interval_min == TARGET_INTERVAL_MINUTES:
            telemetry = _telemetry(day, interval_min)
            telemetry[str(tb.SPEC["key_load"])] = []
            return telemetry
        return _telemetry(day, interval_min)

    recovered, query_interval = _recover_one_day(
        calendar_day,
        TARGET_INTERVAL_MINUTES,
        fetcher,
    )

    assert recovered is not None
    assert query_interval == FINE_RECOVERY_INTERVAL_MINUTES
    assert calls == [TARGET_INTERVAL_MINUTES, FINE_RECOVERY_INTERVAL_MINUTES]


def test_recover_one_day_never_invents_unrecoverable_data() -> None:
    calendar_day = date(2026, 1, 10)
    calls: list[int] = []

    def fetcher(_day: date, interval_min: int) -> dict:
        calls.append(interval_min)
        return {}

    recovered, query_interval = _recover_one_day(
        calendar_day,
        TARGET_INTERVAL_MINUTES,
        fetcher,
    )

    assert recovered is None
    assert query_interval is None
    assert calls == [TARGET_INTERVAL_MINUTES, FINE_RECOVERY_INTERVAL_MINUTES]


def test_recover_one_day_rejects_frozen_recovery() -> None:
    calendar_day = date(2026, 1, 10)

    def fetcher(day: date, interval_min: int) -> dict:
        telemetry = _telemetry(day, interval_min)
        load_key = str(tb.SPEC["key_load"])
        for point in telemetry[load_key]:
            point["value"] = 0.07
        return telemetry

    recovered, query_interval = _recover_one_day(
        calendar_day,
        TARGET_INTERVAL_MINUTES,
        fetcher,
    )

    assert recovered is None
    assert query_interval is None


def test_merge_days_sorts_recovered_dates_chronologically() -> None:
    existing = [_clean_day(date(2026, 1, 2)), _clean_day(date(2026, 1, 4))]
    recovered = [_clean_day(date(2026, 1, 3)), _clean_day(date(2026, 1, 1))]

    merged = _merge_days(existing, recovered)

    assert [day["date"] for day in merged] == [
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
        date(2026, 1, 4),
    ]


def test_merge_days_refuses_overwrite_of_existing_measurement() -> None:
    calendar_day = date(2026, 1, 2)
    try:
        _merge_days([_clean_day(calendar_day)], [_clean_day(calendar_day, 999.0)])
    except ValueError as error:
        assert "overwrite existing day" in str(error)
    else:
        raise AssertionError("Expected duplicate recovery to be rejected")


def test_date_ranges_compacts_consecutive_unrecoverable_days() -> None:
    assert _date_ranges(
        [
            date(2026, 1, 1),
            date(2026, 1, 2),
            date(2026, 1, 4),
            date(2026, 1, 7),
            date(2026, 1, 6),
        ]
    ) == [
        (date(2026, 1, 1), date(2026, 1, 2)),
        (date(2026, 1, 4), date(2026, 1, 4)),
        (date(2026, 1, 6), date(2026, 1, 7)),
    ]


def test_load_existing_days_round_trips_clean_canonical_csv(tmp_path: Path) -> None:
    source = tmp_path / "youngone.csv"
    original = [_clean_day(date(2026, 1, 2)), _clean_day(date(2026, 1, 3))]
    tb._write_csv(original, source)

    loaded = _load_existing_days(source, TARGET_INTERVAL_MINUTES)

    assert [day["date"] for day in loaded] == [date(2026, 1, 2), date(2026, 1, 3)]
    assert loaded[0]["load"] == original[0]["load"]
    assert loaded[1]["pv"] == original[1]["pv"]


def test_load_existing_days_rechecks_cleaner_before_recovery(tmp_path: Path) -> None:
    source = tmp_path / "youngone.csv"
    corrupt = _clean_day(date(2026, 1, 2))
    corrupt["load"] = [0.07] * SAMPLES_PER_DAY
    corrupt["pv"] = [0.07] * SAMPLES_PER_DAY
    tb._write_csv([corrupt], source)

    # TODO(RECOVER-DATA): never let the rescue utility preserve an already-corrupt
    # source day merely because that calendar date is technically present.
    try:
        _load_existing_days(source, TARGET_INTERVAL_MINUTES)
    except ValueError as error:
        assert "fails cleaner" in str(error)
    else:
        raise AssertionError("Expected corrupt existing CSV day to be rejected")
