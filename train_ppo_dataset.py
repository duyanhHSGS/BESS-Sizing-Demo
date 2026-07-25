from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def main():
    checkpoint_dir = Path(os.environ.get("SIZING_DEMO_CHECKPOINT_DIR", Path(__file__).resolve().parent / "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "diseep_simulator" / "tool_c" / "experiments" / "run_train_dataset.py"
    spec = importlib.util.spec_from_file_location("sizing_demo_tool_c_train_dataset", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trainer: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bridge.RESULTS_DIR = checkpoint_dir
    module.main()


if __name__ == "__main__":
    main()
