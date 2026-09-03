from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from scripts.normalize_day_type import day_type_from_date, normalize_csv


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("day_index", "date_iso", "day_type", "step", "P_load_kW", "P_pv_kW"),
        )
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("calendar_day", "expected"),
    (
        (date(2026, 8, 28), "working"),  # Friday
        (date(2026, 8, 29), "working"),  # Saturday
        (date(2026, 8, 30), "weekend"),  # Sunday
        (date(2026, 8, 31), "working"),  # Monday
    ),
)
def test_only_sunday_is_weekend(calendar_day: date, expected: str) -> None:
    assert day_type_from_date(calendar_day) == expected


def test_normalize_csv_changes_only_day_type_and_supports_in_place(tmp_path: Path) -> None:
    source = tmp_path / "telemetry.csv"
    rows = [
        {
            "day_index": "1", "date_iso": "2026-08-29", "day_type": "weekend",
            "step": "0", "P_load_kW": "123.45", "P_pv_kW": "6.7",
        },
        {
            "day_index": "2", "date_iso": "2026-08-30", "day_type": "working",
            "step": "0", "P_load_kW": "99", "P_pv_kW": "0",
        },
    ]
    _write_rows(source, rows)

    result = normalize_csv(source, source)

    with source.open("r", encoding="utf-8", newline="") as csv_file:
        cleaned = list(csv.DictReader(csv_file))
    assert result.rows == 2
    assert result.changed_rows == 2
    assert result.working_rows == 1
    assert result.weekend_rows == 1
    assert cleaned[0] == {**rows[0], "day_type": "working"}
    assert cleaned[1] == {**rows[1], "day_type": "weekend"}


def test_normalize_csv_rejects_missing_required_columns(tmp_path: Path) -> None:
    source = tmp_path / "missing.csv"
    source.write_text("date_iso,step\n2026-08-29,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns: day_type"):
        normalize_csv(source, source)


def test_normalize_csv_rejects_invalid_date_without_replacing_source(tmp_path: Path) -> None:
    source = tmp_path / "invalid.csv"
    original = "date_iso,day_type\nnot-a-date,weekend\n"
    source.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid date_iso at CSV line 2"):
        normalize_csv(source, source)

    assert source.read_text(encoding="utf-8") == original
