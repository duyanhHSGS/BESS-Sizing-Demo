from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _ensure_import_path(path: Path) -> None:
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def main():
    checkpoint_dir = Path(os.environ.get("SIZING_DEMO_CHECKPOINT_DIR", Path(__file__).resolve().parent / "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "SADRBC_Verification" / "drl" / "train_grepo.py"
    _ensure_import_path(script.parent)
    spec = importlib.util.spec_from_file_location("sizing_demo_train_grepo", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trainer: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RESULTS_DIR = checkpoint_dir
    module.main()


if __name__ == "__main__":
    main()
