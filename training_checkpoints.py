from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


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
        if "actor" in raw and "critic" in raw:
            algo = "grepo"
        elif "state_dict" in raw:
            algo = "ppo"
        else:
            algo = "unknown"
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
