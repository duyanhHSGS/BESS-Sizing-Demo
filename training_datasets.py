from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REQUIRED_HEADERS = ("day_index", "step", "P_load_kW", "P_pv_kW", "day_type")
TRAINING_HEADERS = ("date_iso", "day_type", "step", "P_load_kW", "P_pv_kW")


class DatasetError(ValueError):
    pass


def _dataset_paths(base_dir: Path = BASE_DIR) -> dict[str, Path]:
    data_dir = base_dir / "data"
    source_dir = data_dir if data_dir.exists() else base_dir
    paths = {
        "youngone": source_dir / "offline_data_Youngone.csv",
        "youngone_grepo": source_dir / "offline_Youngone_grepo.csv",
    }
    for path in sorted(source_dir.glob("*.csv")):
        if path.is_file():
            paths.setdefault(sanitize_dataset_id(path.stem), path)
            paths.setdefault(sanitize_dataset_id(path.name), path)
    return paths


def sanitize_dataset_id(dataset_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in dataset_id).strip("_").lower()


def _read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        return next(reader, [])


def count_days(path: Path) -> int:
    days = set()
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [header for header in REQUIRED_HEADERS if header not in (reader.fieldnames or [])]
        if missing:
            raise DatasetError(f"{path.name} missing columns: {', '.join(missing)}")
        for row in reader:
            days.add(int(row["day_index"]))
    return len(days)


def detect_resolution_minutes(path: Path) -> int:
    counts: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [header for header in REQUIRED_HEADERS if header not in (reader.fieldnames or [])]
        if missing:
            raise DatasetError(f"{path.name} missing columns: {', '.join(missing)}")
        for row in reader:
            day = int(row["day_index"])
            counts[day] = counts.get(day, 0) + 1
    step_counts = list(counts.values())
    if not step_counts:
        raise DatasetError(f"{path.name} has no rows")
    steps_per_day = max(set(step_counts), key=step_counts.count)
    return round(24 * 60 / steps_per_day)


def list_datasets(base_dir: Path = BASE_DIR) -> list[dict]:
    out = []
    seen_paths = set()
    for dataset_id, path in _dataset_paths(base_dir).items():
        if not path.exists():
            continue
        resolved_path = path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        try:
            n_days = count_days(path)
            res_min = detect_resolution_minutes(path)
            status = "ok"
            error = ""
        except Exception as exc:  # noqa: BLE001
            n_days = 0
            res_min = 0
            status = "error"
            error = str(exc)
        out.append(
            {
                "id": dataset_id,
                "source": path.name,
                "path": str(path),
                "n_days": n_days,
                "res_min": res_min,
                "status": status,
                "error": error,
            }
        )
    return out


def get_dataset_path(dataset_id: str, base_dir: Path = BASE_DIR) -> Path:
    dataset_id = sanitize_dataset_id(dataset_id)
    paths = _dataset_paths(base_dir)
    path = paths.get(dataset_id)
    if not path or not path.exists():
        raise DatasetError(f"unknown dataset: {dataset_id}")
    return path


def require_min_days(path: Path, min_days: int = 90) -> int:
    n_days = count_days(path)
    if n_days < min_days:
        raise DatasetError(f"{path.name} has {n_days} days; training needs at least {min_days}")
    return n_days


def export_training_csv(dataset_id: str, output_dir: Path, base_dir: Path = BASE_DIR, min_days: int = 90) -> Path:
    source = get_dataset_path(dataset_id, base_dir)
    require_min_days(source, min_days)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"train_{sanitize_dataset_id(dataset_id)}.csv"
    start = date(2026, 1, 1)

    with source.open(newline="", encoding="utf-8-sig") as src, target.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        missing = [header for header in REQUIRED_HEADERS if header not in (reader.fieldnames or [])]
        if missing:
            raise DatasetError(f"{source.name} missing columns: {', '.join(missing)}")
        writer = csv.DictWriter(dst, fieldnames=TRAINING_HEADERS)
        writer.writeheader()
        for row in reader:
            day_index = int(row["day_index"])
            writer.writerow(
                {
                    "date_iso": (start + timedelta(days=day_index - 1)).isoformat(),
                    "day_type": row.get("day_type") or "working",
                    "step": row["step"],
                    "P_load_kW": row["P_load_kW"],
                    "P_pv_kW": row["P_pv_kW"],
                }
            )
    return target
