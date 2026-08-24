"""Thin direct launcher for the existing PPO trainer.

No DB job records, no API service, no policy repository.  Arguments are passed
straight through to run_train_dataset.py and artifacts land in ./results.
"""
from __future__ import annotations

import os
import subprocess
import sys

from _core import HERE, core_src, training_script


def run_training(args: list[str]) -> int:
    script = training_script()
    if not script.is_file():
        raise RuntimeError(f"Training script not found: {script}")

    results_dir = (HERE / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src = str(core_src())
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not old_pythonpath else os.pathsep.join((src, old_pythonpath))
    env["DRL_RESULTS_DIR"] = str(results_dir)

    cmd = [sys.executable, "-X", "utf8", "-u", str(script), *args]
    print("[debloated] PPO trainer:", " ".join(cmd), flush=True)
    print(f"[debloated] results -> {results_dir}", flush=True)
    return subprocess.call(cmd, cwd=str(script.parent), env=env)


def main() -> None:
    raise SystemExit(run_training(sys.argv[1:]))


if __name__ == "__main__":
    main()
