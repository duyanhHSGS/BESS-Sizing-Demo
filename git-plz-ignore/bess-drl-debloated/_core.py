"""Locate the existing math/training core without importing service infrastructure.

The debloated runner intentionally reuses the proven PPO/physics implementation
from ``../bess-drl/src``.  It does NOT import FastAPI, MongoDB/Beanie,
repositories, controllers, or the training JobManager.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parent
DEFAULT_CORE_SRC = WORKSPACE_ROOT / "bess-drl" / "src"


def core_src() -> Path:
    override = os.environ.get("BESS_DRL_CORE_SRC", "").strip()
    path = Path(override).expanduser().resolve() if override else DEFAULT_CORE_SRC.resolve()
    package = path / "bess_drl"
    if not package.is_dir():
        raise RuntimeError(
            "Cannot find bess_drl core package. Expected "
            f"{package}. Set BESS_DRL_CORE_SRC to the original bess-drl/src folder."
        )
    return path


def bootstrap_core() -> Path:
    path = core_src()
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    return path


def training_script() -> Path:
    return (
        core_src()
        / "bess_drl"
        / "training"
        / "drl_engine"
        / "run_train_dataset.py"
    )
