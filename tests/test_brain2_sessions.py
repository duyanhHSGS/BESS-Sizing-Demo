from __future__ import annotations

import csv
from pathlib import Path

import pytest

import EXPERIMENT_FIELD.brain1_sessions as brain1_sessions
import EXPERIMENT_FIELD.brain2_sessions as brain2_sessions
import EXPERIMENT_FIELD.brain_env_sessions as human_sessions


def _parameters(**overrides) -> dict[str, object]:
    parameters: dict[str, object] = {
        "selected_data_csv": "brain2-test.csv",
        "battery_capacity_kWh": "100",
        "battery_power_limit_kW": "100",
        "charge_efficiency": "0.9",
        "discharge_efficiency": "0.9",
        "battery_wear_cost": "5",
        "minimum_soc": "0.2",
        "maximum_soc": "0.9",
        "required_final_soc": "0.2",
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
def brain2_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "brain2-test.csv"
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

    monkeypatch.setattr(brain2_sessions, "selected_data_path", lambda _parameters: path)
    monkeypatch.setattr(brain2_sessions, "selected_data_filename", lambda _parameters: path.name)
    with brain2_sessions._SESSIONS_LOCK:
        brain2_sessions._SESSIONS.clear()
    with brain1_sessions._SESSIONS_LOCK:
        brain1_sessions._SESSIONS.clear()
    with human_sessions._SESSIONS_LOCK:
        human_sessions._SESSIONS.clear()
    return path


def test_context_exposes_applied_brain2_schedule(brain2_csv: Path) -> None:
    payload = brain2_sessions.context(_parameters())

    assert payload["dataset_name"] == brain2_csv.name
    assert payload["schedule"]["usable_capacity_kwh"] == pytest.approx(70.0)
    assert payload["schedule"]["cheap_steps"] == 2
    assert payload["schedule"]["normal_action"] > 0.0
    assert payload["schedule"]["expensive_action"] > payload["schedule"]["normal_action"]


def test_session_previews_then_executes_only_brain2_action(brain2_csv: Path) -> None:
    session = brain2_sessions.create_session(
        _parameters(), day_index=1, starting_peak_kw=125.0
    )
    preview = session.preview()

    assert preview["decision"]["tariff_period"] == "cheap"
    assert preview["decision"]["action"] == pytest.approx(-0.7)
    assert preview["decision"]["remaining_cheap_steps"] == 2
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


def test_brain2_registry_is_independent_from_other_playgrounds(brain2_csv: Path) -> None:
    session = brain2_sessions.create_session(_parameters(), day_index=1)

    assert brain2_sessions.get_session(session.session_id) is session
    assert brain1_sessions._SESSIONS == {}
    assert human_sessions._SESSIONS == {}
    with pytest.raises(human_sessions.BrainEnvSessionNotFound):
        brain1_sessions.get_session(session.session_id)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"billing_windows_cheap": "00:00-01:00,02:00-03:00"}, "exactly one cheap"),
        ({"billing_windows_cheap": "23:00-01:00"}, "must not wrap midnight"),
        ({"billing_cheap": "30"}, "0 < cheap < normal < expensive"),
    ],
)
def test_incompatible_applied_schedule_fails_visibly(
    brain2_csv: Path, overrides: dict[str, object], message: str
) -> None:
    parameters = _parameters(**overrides)

    with pytest.raises((brain2_sessions.Brain2SessionError, ValueError), match=message):
        brain2_sessions.context(parameters)
    with pytest.raises((brain2_sessions.Brain2SessionError, ValueError), match=message):
        brain2_sessions.create_session(parameters, day_index=1)


def test_completion_has_full_trace_and_no_fake_future_preview(brain2_csv: Path) -> None:
    session = brain2_sessions.create_session(_parameters(), day_index=1)
    for _ in range(48):
        session.step()

    status = session.status()
    assert status["complete"] is True
    assert status["preview"] is None
    assert len(session.trace) == status["total_steps"]
    assert session.trace[-1]["done"] is True
    assert session.trace[-1]["next_observation"] is None
    with pytest.raises(human_sessions.BrainEnvSessionComplete):
        session.step()


def test_brain2_flask_api_lifecycle(brain2_csv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    monkeypatch.setattr(main, "PARAMETERS", _parameters())
    client = main.app.test_client()

    assert client.get("/api/brain2/context").status_code == 200
    created = client.post(
        "/api/brain2/sessions",
        json={"day_index": 1, "starting_peak_kw": 75.0},
    )
    assert created.status_code == 201
    session_id = created.get_json()["status"]["session_id"]
    assert created.get_json()["status"]["preview"]["decision"]["tariff_period"] == "cheap"

    stepped = client.post(f"/api/brain2/sessions/{session_id}/step")
    assert stepped.status_code == 200
    assert stepped.get_json()["entry"]["executed_action"] == pytest.approx(-0.7)
    assert client.get(f"/api/brain2/sessions/{session_id}").status_code == 200
    assert client.delete(f"/api/brain2/sessions/{session_id}").status_code == 200
    assert client.get(f"/api/brain2/sessions/{session_id}").status_code == 404
