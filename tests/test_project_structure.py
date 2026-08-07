from __future__ import annotations

import importlib
from pathlib import Path

from bess.paths import PROJECT_ROOT
from bess.training.training_launcher import TRAINING_MODULES


OLD_ROOT_MODULES = {
    "common.py",
    "settings.py",
    "bess_env.py",
    "scenario_gen.py",
    "ppo_agent.py",
    "ppo2_agent.py",
    "grepo_agent.py",
    "grepro_agent.py",
    "pro_agent.py",
    "sadrbc.py",
    "training_launcher.py",
    "benchmark.py",
    "dispatch_runner.py",
    "thingsboard_connector.py",
}


def test_project_root_anchor_is_stable() -> None:
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert (PROJECT_ROOT / "data").is_dir()
    assert (PROJECT_ROOT / "bess").is_dir()


def test_old_root_modules_stay_moved() -> None:
    leftovers = sorted(name for name in OLD_ROOT_MODULES if (PROJECT_ROOT / name).exists())
    assert leftovers == []


def test_training_runners_are_importable_package_modules() -> None:
    assert set(TRAINING_MODULES) == {"ppo", "ppo2", "grepo", "grepro", "pro"}
    for module_name in TRAINING_MODULES.values():
        importlib.import_module(module_name)


def test_flask_entrypoint_uses_moved_template_directory() -> None:
    main = importlib.import_module("main")
    assert main.app.template_folder == "web/templates"
    assert (PROJECT_ROOT / "web" / "templates" / "index.html").is_file()
