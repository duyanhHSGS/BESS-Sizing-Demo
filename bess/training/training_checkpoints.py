from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from bess.paths import PROJECT_ROOT


BASE_DIR = PROJECT_ROOT
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CURVE_FIELDS = (
    "steps",
    "val_cost_vnd",
    "oracle_gap_pct",
    "saving_vs_nobess_pct",
)


def _load_checkpoint_meta(path: Path) -> tuple[str, dict, str | None]:
    try:
        import torch

        raw = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return ("unknown", {}, str(exc))

    if not isinstance(raw, dict):
        return ("legacy", {}, None)
    meta = raw.get("meta", {}) or {}
    algo = raw.get("algo") or meta.get("algo")
    if not algo:
        if "state_dict" in raw:
            algo = "ppo"
        else:
            algo = "removed_legacy"
    return (str(algo), dict(meta), None)


def list_checkpoints(checkpoint_dir: Path = CHECKPOINT_DIR) -> list[dict]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(checkpoint_dir.glob("policy_*.pt")):
        if not path.is_file() or "archive" in path.parts:
            continue
        algo, meta, error = _load_checkpoint_meta(path)
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "algo": algo,
                "e_cap_kwh": meta.get("e_cap_kwh"),
                "p_rated_kw": meta.get("p_rated_kw"),
                "billing_mode": meta.get("billing_mode"),
                "test_saving_pct": meta.get("test_saving_pct"),
                "trained": meta.get("trained"),
                "meta": meta,
                "error": error,
            }
        )
    return rows


def _checkpoint_artifacts(path: Path) -> tuple[Path, Path]:
    tag = path.stem.removeprefix("policy_")
    return (
        path.with_name(f"training_curve_{tag}.csv"),
        path.with_name(f"training_report_{tag}.json"),
    )


def _checkpoint_settings_artifact(path: Path) -> Path:
    tag = path.stem.removeprefix("policy_")
    return path.with_name(f"training_settings_{tag}.json")


def _finite_float(value, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite")
    return number


def _load_curve(path: Path) -> tuple[list[dict], list[str]]:
    if not path.is_file():
        return [], ["Learning curve unavailable; training may have stopped before the first persisted validation point."]
    points = []
    warnings = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                try:
                    points.append(
                        {
                            "steps": int(_finite_float(row.get("steps"), "steps")),
                            "val_cost_vnd": _finite_float(row.get("val_cost_vnd"), "val_cost_vnd"),
                            "oracle_gap_pct": _finite_float(row.get("oracle_gap_pct"), "oracle_gap_pct"),
                            "saving_vs_nobess_pct": _finite_float(
                                row.get("saving_vs_nobess_pct"), "saving_vs_nobess_pct"
                            ),
                            **{
                                field: _finite_float(row.get(field), field)
                                for field in CURVE_FIELDS[4:]
                                if row.get(field) not in (None, "")
                            },
                        }
                    )
                except (TypeError, ValueError) as exc:
                    warnings.append(f"Skipped malformed curve row {row_number}: {exc}")
    except (OSError, csv.Error) as exc:
        return [], [f"Learning curve could not be read: {exc}"]
    if not points and not warnings:
        warnings.append("Learning curve is empty.")
    return points, warnings


def _load_report(path: Path) -> tuple[dict, list[str]]:
    if not path.is_file():
        return {}, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("report root must be an object")
        return payload, []
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"Training report could not be read: {exc}"]


def _curve_summary(points: list[dict]) -> dict:
    if not points:
        return {}
    best_bill = min(points, key=lambda point: point["val_cost_vnd"])
    best_saving = max(points, key=lambda point: point["saving_vs_nobess_pct"])
    best_gap = min(points, key=lambda point: point["oracle_gap_pct"])
    latest = max(points, key=lambda point: point["steps"])
    saving_fraction = best_bill["saving_vs_nobess_pct"] / 100.0
    gap_fraction = best_bill["oracle_gap_pct"] / 100.0
    val_no_bess = (
        best_bill["val_cost_vnd"] / (1.0 - saving_fraction)
        if abs(1.0 - saving_fraction) > 1e-12
        else None
    )
    val_oracle = (
        best_bill["val_cost_vnd"] / (1.0 + gap_fraction)
        if abs(1.0 + gap_fraction) > 1e-12
        else None
    )
    return {
        "best_bill": best_bill,
        "best_saving": best_saving,
        "best_oracle_gap": best_gap,
        "latest": latest,
        "max_steps": max(point["steps"] for point in points),
        "val_no_bess_vnd": val_no_bess,
        "val_oracle_vnd": val_oracle,
    }


def get_checkpoint_report(
    checkpoint_name: str,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> dict:
    safe_name = Path(str(checkpoint_name)).name
    if safe_name != checkpoint_name or not safe_name.startswith("policy_") or not safe_name.endswith(".pt"):
        raise ValueError("invalid checkpoint name")
    known = {row["name"]: row for row in list_checkpoints(checkpoint_dir)}
    if safe_name not in known:
        raise FileNotFoundError(safe_name)
    checkpoint = known[safe_name]
    checkpoint_path = checkpoint_dir.resolve() / safe_name
    curve_path, report_path = _checkpoint_artifacts(checkpoint_path)
    settings_path = _checkpoint_settings_artifact(checkpoint_path)
    points, warnings = _load_curve(curve_path)
    saved_report, report_warnings = _load_report(report_path)
    warnings.extend(report_warnings)
    run_settings, settings_warnings = _load_report(settings_path)
    warnings.extend(settings_warnings)
    if not settings_path.is_file():
        warnings.append(
            "Frozen launch settings are unavailable for this legacy run; "
            "current UI values are not proof of its historical configuration."
        )
    required_sampling = {
        "native_dt_minutes",
        "control_dt_minutes",
        "native_steps_per_action",
    }
    if not required_sampling.issubset(checkpoint.get("meta", {})):
        warnings.append(
            "Legacy checkpoint lacks required control sampling delta-t "
            "metadata and is blocked from Dispatch; retrain it."
        )
    return {
        "checkpoint": checkpoint,
        "curve": points,
        "summary": _curve_summary(points),
        "training": saved_report,
        "run_settings": run_settings,
        "artifacts": {
            "curve": curve_path.name if curve_path.is_file() else None,
            "report": report_path.name if report_path.is_file() else None,
            "settings": settings_path.name if settings_path.is_file() else None,
        },
        "warnings": warnings,
    }
