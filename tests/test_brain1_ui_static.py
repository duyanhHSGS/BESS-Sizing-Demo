from bess.paths import PROJECT_ROOT


def _brain1_section() -> str:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    return html.split('<section id="tab-brain1"', 1)[1].split(
        '<section id="tab-brain2"', 1
    )[0]


def test_brain1_has_separate_spectator_tab_without_human_action_or_auto_run() -> None:
    section = _brain1_section()

    for element_id in (
        "brain1-day",
        "brain1-starting-peak",
        "btn-brain1-start",
        "btn-brain1-step",
        "btn-brain1-reset",
        "brain1-eyes",
        "brain1-decision",
        "brain1-reward-chart",
        "brain1-grid-chart",
        "brain1-soc-chart",
    ):
        assert f'id="{element_id}"' in section
    assert "brain1-action" not in section
    assert "btn-brain1-auto" not in section


def test_brain1_ui_previews_decisions_and_steps_without_action_payload() -> None:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")

    assert '"/api/brain1/context"' in html
    assert '"/api/brain1/sessions"' in html
    assert "renderBrain1Eyes(preview)" in html
    assert "renderBrain1Decision(preview)" in html
    assert "stepBrain1Session" in html
    assert 'method: "POST"' in html
    assert "body: JSON.stringify({ action" not in _brain1_section()
    assert "storedBrain1SessionId" in html


def test_brain_chart_line_keeps_a_single_latest_sample_visible() -> None:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    draw_line = html.split("function drawBrainLine", 1)[1].split(
        "function drawBrainAxes", 1
    )[0]

    assert "const latestIndex = values.length - 1" in draw_line
    assert "canvasContext.arc(" in draw_line
    assert "canvasContext.fill()" in draw_line


def test_brain1_graph_uses_executed_step_progress_and_guards_completion() -> None:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")

    assert "function spectatorXForTraceIndex" in html
    assert "recordedStep + 1" in html
    assert "function drawSpectatorProgress" in html
    assert "traceSteps === totalSteps" in html
    assert "brain1Trace.length === totalSteps" in html
    assert "the graph owns ${brain1Trace.length}/${totalSteps} executed steps" in html
