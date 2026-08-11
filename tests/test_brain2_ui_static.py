from bess.paths import PROJECT_ROOT


def _brain2_section() -> str:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    return html.split('<section id="tab-brain2"', 1)[1].split(
        '<section id="tab-benchmarking"', 1
    )[0]


def test_brain2_has_separate_spectator_tab_without_human_action_or_auto_run() -> None:
    section = _brain2_section()

    for element_id in (
        "brain2-day",
        "brain2-starting-peak",
        "btn-brain2-start",
        "btn-brain2-step",
        "btn-brain2-reset",
        "brain2-eyes",
        "brain2-decision",
        "brain2-schedule",
        "brain2-reward-chart",
        "brain2-grid-chart",
        "brain2-soc-chart",
    ):
        assert f'id="{element_id}"' in section
    assert "brain2-action" not in section
    assert "btn-brain2-auto" not in section


def test_brain2_ui_previews_schedule_and_steps_without_action_payload() -> None:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")

    assert '"/api/brain2/context"' in html
    assert '"/api/brain2/sessions"' in html
    assert "renderBrain2Eyes(preview)" in html
    assert "renderBrain2Decision(preview)" in html
    assert "renderBrain2Schedule()" in html
    assert "stepBrain2Session" in html
    assert "body: JSON.stringify({ action" not in _brain2_section()
    assert "storedBrain2SessionId" in html


def test_brain2_graph_uses_shared_executed_progress_and_guards_completion() -> None:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")

    assert "spectatorXForTraceIndex(rewardPlot, index, totalSteps, brain2Trace)" in html
    assert "drawSpectatorProgress" in html
    assert "brain2Trace.length === totalSteps" in html
    assert "the graph owns ${brain2Trace.length}/${totalSteps} executed steps" in html
