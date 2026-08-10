from pathlib import Path

from bess.paths import PROJECT_ROOT


def test_brain_env_tab_contains_required_controls_charts_and_diagnostics() -> None:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")

    required_ids = {
        "tab-brain-env",
        "brain-day",
        "brain-starting-peak",
        "brain-action",
        "btn-brain-start",
        "btn-brain-step",
        "btn-brain-auto",
        "btn-brain-pause",
        "btn-brain-reset",
        "brain-grid-chart",
        "brain-soc-chart",
        "brain-diagnostics",
    }
    for element_id in required_ids:
        assert f'id="{element_id}"' in html


def test_brain_env_browser_loop_is_sequential_reconnectable_and_pause_safe() -> None:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")

    assert '"/api/brain-env/context"' in html
    assert '"/api/brain-env/sessions"' in html
    assert "/step`" in html
    assert "await stepBrainSession()" in html
    assert "window.setTimeout(runBrainAutoTick" in html
    assert "if (brainStepPending) return" in html
    assert "pauseBrainAuto();" in html
    assert "function storedBrainSessionId()" in html
    assert "function storeBrainSessionId(sessionId)" in html
    assert "function clearStoredBrainSessionId()" in html


def test_brain_action_keeps_its_real_bounded_range_slider() -> None:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")

    assert 'id="brain-action" data-keep-range type="range" min="-1" max="1"' in html
    assert 'input[type="range"]:not([data-keep-range])' in html
