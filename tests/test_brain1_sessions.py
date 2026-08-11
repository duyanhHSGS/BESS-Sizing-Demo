from __future__ import annotations

import csv
from pathlib import Path

import pytest

import EXPERIMENT_FIELD.brain1_sessions as brain1_sessions
import EXPERIMENT_FIELD.brain_env_sessions as human_sessions


def _parameters(**overrides) -> dict[str, object]:
    parameters: dict[str, object] = {
        "selected_data_csv": "brain1-test.csv",
        "battery_capacity_kWh": "1000",
        "battery_power_limit_kW": "100",
        "charge_efficiency": "0.9",
        "discharge_efficiency": "0.9",
        "battery_wear_cost": "5",
        "minimum_soc": "0.2",
        "maximum_soc": "0.9",
        "required_final_soc": "0.5",
        "billing_mode": "2tc",
        "billing_sunday": True,
        "billing_expensive": "30",
        "billing_normal": "20",
        "billing_cheap": "10",
        "billing_peak_penalty": "100",
        "billing_windows_expensive": "01:00-02:00",
        "billing_windows_cheap": "00:00-01:00",
    }
    parameters.update(overrides)
    return parameters


@pytest.fixture
def brain1_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "brain1-test.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["day_index", "step", "P_load_kW", "P_pv_kW", "day_type", "date_iso"],
        )
        writer.writeheader()
        for day_index, date_iso in ((1, "2026-01-03"), (2, "2026-01-04")):
            for step in range(48):
                writer.writerow(
                    {
                        "day_index": day_index,
                        "step": step,
                        "P_load_kW": 200.0,
                        "P_pv_kW": 50.0,
                        "day_type": "working" if day_index == 1 else "weekend",
                        "date_iso": date_iso,
                    }
                )

    monkeypatch.setattr(brain1_sessions, "selected_data_path", lambda _parameters: path)
    monkeypatch.setattr(brain1_sessions, "selected_data_filename", lambda _parameters: path.name)
    with brain1_sessions._SESSIONS_LOCK:
        brain1_sessions._SESSIONS.clear()
    with human_sessions._SESSIONS_LOCK:
        human_sessions._SESSIONS.clear()
    return path


def test_context_derives_thresholds_from_applied_ui_tariffs(brain1_csv: Path) -> None:
    payload = brain1_sessions.context(_parameters())

    assert payload["dataset_name"] == brain1_csv.name
    assert payload["thresholds"]["normalization_vnd_per_kwh"] == pytest.approx(30.0)
    assert payload["thresholds"]["cheap_normalized"] == pytest.approx(1.0 / 3.0)
    assert payload["thresholds"]["expensive_normalized"] == pytest.approx(1.0)


def test_sunday_normal_tariff_does_not_cosplay_as_expensive(brain1_csv: Path) -> None:
    session = brain1_sessions.create_session(_parameters(), day_index=2)

    assert session.preview()["decision"]["label"] == "CHARGE"
    session.step()
    session.step()
    preview = session.preview()

    assert preview["tariff_vnd_per_kwh"] == pytest.approx(20.0)
    assert preview["observation"]["normalized_tariff"] == pytest.approx(2.0 / 3.0)
    assert preview["decision"]["label"] == "IDLE"


def test_session_previews_then_executes_only_brain1_action(brain1_csv: Path) -> None:
    session = brain1_sessions.create_session(
        _parameters(), day_index=1, starting_peak_kw=125.0
    )
    preview = session.preview()

    assert preview["observation"]["normalized_monthly_peak"] == pytest.approx(0.125)
    assert preview["decision"]["action"] == pytest.approx(-1.0)
    assert session.env.bess_world.meter_state.monthly_peak_kw == pytest.approx(125.0)
    assert session.env.raw_world.meter_state.monthly_peak_kw == pytest.approx(125.0)

    entry, status = session.step()

    assert entry["executed_action"] == preview["decision"]["action"]
    assert entry["reward"]["timestep_savings_vnd"] == pytest.approx(
        entry["net_battery_savings_vnd"]
    )
    assert status["step_index"] == 1
    assert status["preview"]["step_index"] == 1
    assert len(session.trace) == 1


def test_brain1_registry_is_independent_from_human_brain_env_registry(brain1_csv: Path) -> None:
    session = brain1_sessions.create_session(_parameters(), day_index=1)

    assert brain1_sessions.get_session(session.session_id) is session
    assert human_sessions._SESSIONS == {}
    with pytest.raises(human_sessions.BrainEnvSessionNotFound):
        human_sessions.get_session(session.session_id)


def test_non_distinct_applied_tariffs_fail_visibly(brain1_csv: Path) -> None:
    parameters = _parameters(billing_cheap="30", billing_expensive="30")

    with pytest.raises(brain1_sessions.Brain1SessionError, match="cheap tariff"):
        brain1_sessions.context(parameters)
    with pytest.raises(brain1_sessions.Brain1SessionError, match="cheap tariff"):
        brain1_sessions.create_session(parameters, day_index=1)


def test_completion_has_no_fake_future_preview(brain1_csv: Path) -> None:
    session = brain1_sessions.create_session(_parameters(), day_index=1)
    for _ in range(48):
        session.step()

    assert session.status()["complete"] is True
    assert session.status()["preview"] is None
    assert len(session.trace) == session.status()["total_steps"]
    assert session.trace[-1]["done"] is True
    assert session.trace[-1]["next_observation"] is None
    with pytest.raises(human_sessions.BrainEnvSessionComplete):
        session.step()


def test_brain1_flask_api_lifecycle(brain1_csv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    monkeypatch.setattr(main, "PARAMETERS", _parameters())
    client = main.app.test_client()

    assert client.get("/api/brain1/context").status_code == 200
    created = client.post(
        "/api/brain1/sessions",
        json={"day_index": 1, "starting_peak_kw": 75.0},
    )
    assert created.status_code == 201
    session_id = created.get_json()["status"]["session_id"]
    assert created.get_json()["status"]["preview"]["decision"]["label"] == "CHARGE"

    stepped = client.post(f"/api/brain1/sessions/{session_id}/step")
    assert stepped.status_code == 200
    assert stepped.get_json()["entry"]["executed_action"] == pytest.approx(-1.0)
    assert client.get(f"/api/brain1/sessions/{session_id}").status_code == 200
    assert client.delete(f"/api/brain1/sessions/{session_id}").status_code == 200
    assert client.get(f"/api/brain1/sessions/{session_id}").status_code == 404
