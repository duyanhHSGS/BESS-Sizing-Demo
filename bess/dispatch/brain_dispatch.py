"""Brain-native measured-data dispatch orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from bess.brain.runtime import run_controllers
from bess.evaluation.benchmark import selected_data_path
from bess.training.brain3_checkpoints import CHECKPOINT_DIR


class DispatchRunWarning(RuntimeError):
    pass


def run_dispatch(
    controller_ids: list[str],
    parameters: dict[str, Any],
    *,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> tuple[dict[str, Any], list[str]]:
    if not controller_ids:
        raise DispatchRunWarning("select at least one brain")
    return run_controllers(
        controller_ids,
        selected_data_path(parameters),
        parameters,
        checkpoint_dir,
    )
