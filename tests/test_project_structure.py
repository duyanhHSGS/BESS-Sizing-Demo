from pathlib import Path

from bess.paths import PROJECT_ROOT
from bess.training.brain3_launcher import TRAINING_MODULES


def test_brain_hq_is_the_only_active_controller_stack() -> None:
    assert TRAINING_MODULES == {"brain3_dqn": "bess.training.runners.train_brain3"}
    assert (PROJECT_ROOT / "bess" / "brain" / "brain_env.py").is_file()
    assert not (PROJECT_ROOT / "EXPERIMENT_FIELD").exists()
    assert not (PROJECT_ROOT / "bess" / "agents").exists()
    assert not (PROJECT_ROOT / "bess" / "forecasting").exists()
    assert not (PROJECT_ROOT / "bess" / "core" / "bess_env.py").exists()


def test_flask_entrypoint_uses_brain_hq_assets() -> None:
    source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "from bess.webapp import app" in source
    assert "/assets/brain_hq.css" in html
    assert "/assets/brain_hq_overrides.css" in html
    assert "/assets/brain_hq.js" in html
    assert len(html.split('class="panel')) == 8


def test_legacy_algorithm_vocabulary_is_absent_from_active_sources() -> None:
    forbidden = ("ppo", "grepo", "grepro", "sadrbc")
    active = [
        *PROJECT_ROOT.joinpath("bess").rglob("*.py"),
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "web" / "templates" / "index.html",
        PROJECT_ROOT / "web" / "brain_hq.js",
    ]
    for path in active:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(word in text for word in forbidden), path
