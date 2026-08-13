from __future__ import annotations

import csv
import json
import os
from pathlib import Path


CURVE_FIELDS = [
    "steps",
    "val_cost_vnd",
    "oracle_gap_pct",
    "saving_vs_nobess_pct",
]


def write_curve(path: Path, points: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(points)
    os.replace(temp, path)


def write_report(path: Path, report: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temp, path)
