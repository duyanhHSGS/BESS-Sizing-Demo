"""Small PPO2-owned report persistence helper."""
from __future__ import annotations

import json
import os
from pathlib import Path


def write_report(path: Path, report: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temp, path)
