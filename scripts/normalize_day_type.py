"""Normalize CSV ``day_type`` values from ``date_iso``.

Youngone operates Monday through Saturday.  Sunday is the only weekend day.
The script rewrites only ``day_type`` and preserves the original column order
and every other cell.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path


REQUIRED_HEADERS = frozenset({"date_iso", "day_type"})


@dataclass(frozen=True, slots=True)
class CleanupResult:
    rows: int
    changed_rows: int
    working_rows: int
    weekend_rows: int


def day_type_from_date(calendar_day: date) -> str:
    """Return Youngone's calendar class: Sunday only is weekend."""
    return "weekend" if calendar_day.weekday() == 6 else "working"


def normalize_csv(input_path: Path, output_path: Path) -> CleanupResult:
    """Write a normalized copy, replacing ``output_path`` atomically."""
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    changed_rows = 0
    working_rows = 0
    weekend_rows = 0
    row_count = 0

    # TODO(DAY-TYPE): Keep date_iso authoritative if more site calendars are added.
    # Validate and render in memory first. One filesystem write is dramatically
    # faster than 30,000 tiny writes under Windows filesystem monitoring.
    with source.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source}")
        missing = REQUIRED_HEADERS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        rendered = io.StringIO(newline="")
        # The tracked telemetry CSV uses CRLF; preserve that canonical format.
        writer = csv.DictWriter(rendered, fieldnames=reader.fieldnames, lineterminator="\r\n")
        writer.writeheader()
        for line_number, row in enumerate(reader, start=2):
            raw_date = (row.get("date_iso") or "").strip()
            try:
                calendar_day = date.fromisoformat(raw_date)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid date_iso at CSV line {line_number}: {raw_date!r}"
                ) from exc

            expected = day_type_from_date(calendar_day)
            changed_rows += row.get("day_type") != expected
            row["day_type"] = expected
            working_rows += expected == "working"
            weekend_rows += expected == "weekend"
            row_count += 1
            writer.writerow(row)

    temporary_path = destination.with_name(
        f"{destination.stem}.day-type-{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("x", encoding="utf-8", newline="") as temporary_file:
            temporary_file.write(rendered.getvalue())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return CleanupResult(
        rows=row_count,
        changed_rows=changed_rows,
        working_rows=working_rows,
        weekend_rows=weekend_rows,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set day_type from date_iso with Sunday as the only weekend day."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Omit to clean the input file in place.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or args.input_csv
    result = normalize_csv(args.input_csv, output)
    print(
        f"[day-type] wrote {output}: {result.rows} rows, "
        f"{result.changed_rows} changed, {result.working_rows} working, "
        f"{result.weekend_rows} weekend"
    )


if __name__ == "__main__":
    main()
