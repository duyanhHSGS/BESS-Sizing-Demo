from __future__ import annotations

from pathlib import Path

from runtime import DebloatedRuntime, RuntimeOptions, _billing_demand_transition
from webui import _display_command, create_app


def test_billing_transition_matches_full_runtime_rules() -> None:
    assert _billing_demand_transition(
        current_slot=0, day_of_month=1, completed_demand_kw=123.0
    ) == (None, None)
    assert _billing_demand_transition(
        current_slot=0, day_of_month=2, completed_demand_kw=123.0
    ) == (None, 123.0)
    assert _billing_demand_transition(
        current_slot=1, day_of_month=2, completed_demand_kw=123.0
    ) == (None, None)
    assert _billing_demand_transition(
        current_slot=2, day_of_month=2, completed_demand_kw=123.0
    ) == (123.0, 123.0)
    assert _billing_demand_transition(
        current_slot=3, day_of_month=2, completed_demand_kw=123.0
    ) == (123.0, None)


def test_standalone_overlay_is_96_slots_and_marks_current_slot_arb() -> None:
    overlay = DebloatedRuntime._build_overlay(
        base_plan=None,
        today_str="2026-08-14",
        current_slot=17,
        p_drl_kw=321.0,
    )
    assert len(overlay.p_plan) == 96
    assert len(overlay.dispatch_sources) == 96
    assert overlay.p_plan[17] == 321.0
    assert overlay.dispatch_sources[17] == "arb"
    assert all(
        source == "standby"
        for index, source in enumerate(overlay.dispatch_sources)
        if index != 17
    )


def test_options_reject_nonexistent_policy_and_config(tmp_path: Path) -> None:
    options = RuntimeOptions(
        policy_path=tmp_path / "missing.pt",
        config_path=tmp_path / "missing.json",
    )
    try:
        options.validate()
    except FileNotFoundError as exc:
        assert "Policy not found" in str(exc)
    else:
        raise AssertionError("missing policy must fail loudly")


def test_flask_ui_boots_and_reports_plain_file_status() -> None:
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()
    index = client.get("/")
    assert index.status_code == 200
    assert b"<title>Sizing Demo</title>" in index.data
    assert b"Training Lab" in index.data
    assert b"Benchmarking" in index.data
    assert b"Shadow Running" in index.data

    datasets = client.get("/api/training/datasets")
    assert datasets.status_code == 200
    assert isinstance(datasets.get_json(), list)

    status = client.get("/api/status")
    assert status.status_code == 200
    payload = status.get_json()
    assert payload["ok"] is True
    assert "training" in payload
    assert "runtime" in payload
    assert "artifacts" in payload


def test_ui_command_display_redacts_secrets() -> None:
    text = _display_command(
        ["python", "main.py", "run", "--api-key", "very-secret", "--mqtt-password", "also-secret"]
    )
    assert "very-secret" not in text
    assert "also-secret" not in text
    assert text.count("***") == 2
