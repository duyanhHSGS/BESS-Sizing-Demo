"""Strict discovery and reporting for Brain 3 artifacts only."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from bess.paths import PROJECT_ROOT

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def _safe(name: str, prefix: str) -> str:
    clean = Path(str(name)).name
    if clean != name or not clean.startswith(prefix) or not clean.endswith(".pt"):
        raise ValueError("invalid Brain 3 checkpoint name")
    return clean


def _deployment_row(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint root is not an object")
        if payload.get("schema_version") != 1 or payload.get("algorithm") != "brain3_dqn":
            raise ValueError("unsupported checkpoint schema")
        if payload.get("observation_dim") != 7 or tuple(payload.get("action_values", ())) != (-1.0, 0.0, 1.0):
            raise ValueError("incompatible seven-eye/three-action contract")
        meta = dict(payload.get("meta") or {})
        fingerprint = meta.get("environment_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("missing environment fingerprint")
        if not isinstance(meta.get("environment"), dict):
            raise ValueError("missing sizing/economics contract")
        if int(meta.get("native_steps_per_action", 0)) <= 0:
            raise ValueError("missing sampling contract")
        return {
            "id": f"brain3:{path.name}",
            "name": path.name,
            "display_name": f"Brain 3 - {path.stem.removeprefix('brain3_')}",
            "algo": "brain3_dqn",
            "meta": meta,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "id": f"brain3:{path.name}",
            "name": path.name,
            "display_name": path.name,
            "algo": "brain3_dqn",
            "meta": {},
            "error": str(exc),
        }


def list_checkpoints(checkpoint_dir: Path = CHECKPOINT_DIR) -> list[dict]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return [
        _deployment_row(path)
        for path in sorted(checkpoint_dir.glob("brain3_*.pt"))
        if path.is_file() and not path.name.startswith("brain3_resume_")
    ]


def list_resume_checkpoints(checkpoint_dir: Path = CHECKPOINT_DIR) -> list[str]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return [path.name for path in sorted(checkpoint_dir.glob("brain3_resume_*.pt")) if path.is_file()]


def list_compatible_checkpoints(environment_fingerprint: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> list[dict]:
    return [
        row
        for row in list_checkpoints(checkpoint_dir)
        if row.get("error") is None
        and row.get("meta", {}).get("environment_fingerprint") == environment_fingerprint
    ]


def get_checkpoint_report(name: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> dict:
    safe = _safe(name, "brain3_")
    known = {row["name"]: row for row in list_checkpoints(checkpoint_dir)}
    if safe not in known:
        raise FileNotFoundError(safe)
    tag = Path(safe).stem.removeprefix("brain3_")
    report_path = checkpoint_dir / f"brain3_report_{tag}.json"
    settings_path = checkpoint_dir / f"training_settings_{tag}.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}
    return {
        "checkpoint": known[safe],
        "training": report,
        "curve": report.get("curve", []),
        "run_settings": settings,
        "artifacts": {
            "report": report_path.name if report_path.is_file() else None,
            "settings": settings_path.name if settings_path.is_file() else None,
            "resume": f"brain3_resume_{tag}.pt" if (checkpoint_dir / f"brain3_resume_{tag}.pt").is_file() else None,
        },
        "warnings": [],
    }
