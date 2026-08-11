from __future__ import annotations

import csv
from pathlib import Path

import pytest

import bess.brain.brain_env_sessions as brain_sessions
from bess.brain.brain_env import action_to_requested_battery_power_kw


def _parameters(*, billing_mode: str = "2tc") -> dict[str, object]:
    return {
        "selected_data_csv": "brain-test.csv",
        "battery_capacity_kWh": "1000",
        "battery_power_limit_kW": "100",
        "charge_efficiency": "0.9",
        "discharge_efficiency": "0.9",
        "battery_wear_cost": "5",
        "minimum_soc": "0.2",
        "maximum_soc": "0.9",
        "required_final_soc": "0.5",
        "billing_mode": billing_mode,
        "billing_sunday": True,
        "billing_expensive": "30",
        "billing_normal": "20",
        "billing_cheap": "10",
        "billing_peak_penalty": "100",
        "billing_windows_expensive": "01:00-02:00",
        "billing_windows_cheap": "00:00-01:00",
    }


@pytest.fixture
def brain_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "brain-test.csv"
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
                        "P_load_kW": 200 + step,
                        "P_pv_kW": 50,
                        "day_type": "working" if day_index == 1 else "weekend",
                        "date_iso": date_iso,
                    }
                )

    monkeypatch.setattr(brain_sessions, "selected_data_path", lambda _parameters: path)
    monkeypatch.setattr(brain_sessions, "selected_data_filename", lambda _parameters: path.name)
    with brain_sessions._SESSIONS_LOCK:
        brain_sessions._SESSIONS.clear()
    return path


def test_context_lists_csv_days_and_applied_settings(brain_csv: Path) -> None:
    payload = brain_sessions.context(_parameters())

    assert payload["dataset_name"] == brain_csv.name
    assert payload["timestep_hours"] == pytest.approx(0.5)
    assert [day["day_index"] for day in payload["days"]] == [1, 2]
    assert payload["days"][0]["step_count"] == 48
    assert payload["settings"]["initial_soc"] == pytest.approx(0.5)
    assert payload["settings"]["battery_wear_vnd_per_kwh"] == pytest.approx(5.0)


def test_session_uses_net_load_equal_carry_in_peaks_and_exact_step_accounting(
    brain_csv: Path,
) -> None:
    session = brain_sessions.create_session(
        _parameters(), day_index=1, starting_peak_kw=125.0
    )

    assert session.env.bess_world.state_of_charge == pytest.approx(0.5)
    assert session.env.bess_world.meter_state.monthly_peak_kw == pytest.approx(125.0)
    assert session.env.raw_world.meter_state.monthly_peak_kw == pytest.approx(125.0)
    assert session.net_load_kw[0] == pytest.approx(150.0)

    entry, status = session.step(1.0)

    assert entry["raw_grid_import_kw"] == pytest.approx(150.0)
    assert entry["requested_battery_kw"] == pytest.approx(100.0)
    assert entry["bess_grid_import_kw"] == pytest.approx(60.0)
    assert entry["next_soc"] == pytest.approx(0.45)
    assert status["step_index"] == 1
    assert status["summary"]["net_battery_savings_vnd"] == pytest.approx(
        status["summary"]["energy_savings_vnd"]
        + status["summary"]["demand_savings_vnd"]
        - status["summary"]["battery_wear_cost_vnd"]
    )


def test_tariff_snapshot_honors_cheap_precedence_and_sunday_no_peak(brain_csv: Path) -> None:
    saturday = brain_sessions.create_session(_parameters(), day_index=1)
    sunday = brain_sessions.create_session(_parameters(), day_index=2)

    assert saturday.tariffs_vnd_per_kwh[:4] == pytest.approx((10.0, 10.0, 30.0, 30.0))
    assert sunday.tariffs_vnd_per_kwh[:4] == pytest.approx((10.0, 10.0, 20.0, 20.0))


def test_energy_only_mode_keeps_peak_meter_but_removes_demand_money(brain_csv: Path) -> None:
    session = brain_sessions.create_session(_parameters(billing_mode="tou"), day_index=1)

    session.step(0.0)
    entry, status = session.step(0.0)

    assert entry["raw_block_completed"] is True
    assert entry["raw_monthly_peak_kw"] > 0.0
    assert entry["raw_demand_cost_vnd"] == pytest.approx(0.0)
    assert status["summary"]["raw_demand_cost_vnd"] == pytest.approx(0.0)


def test_rejected_inputs_do_not_advance_or_mutate_session(brain_csv: Path) -> None:
    with pytest.raises(brain_sessions.BrainEnvSessionError, match="does not exist"):
        brain_sessions.create_session(_parameters(), day_index=999)
    with pytest.raises(brain_sessions.BrainEnvSessionError, match="at least 0"):
        brain_sessions.create_session(_parameters(), day_index=1, starting_peak_kw=-1)

    session = brain_sessions.create_session(_parameters(), day_index=1)
    with pytest.raises(brain_sessions.BrainEnvSessionError, match="between -1 and 1"):
        session.step(1.01)
    with pytest.raises(brain_sessions.BrainEnvSessionError, match="finite"):
        session.step(float("nan"))

    assert session.step_index == 0
    assert session.trace == []
    assert session.env.bess_world.total_operating_cost_vnd == pytest.approx(0.0)
    assert session.env.raw_world.total_operating_cost_vnd == pytest.approx(0.0)


def test_low_level_brain_action_conversion_clips_to_physical_action_contract() -> None:
    assert action_to_requested_battery_power_kw(2.0, 100.0) == pytest.approx(100.0)
    assert action_to_requested_battery_power_kw(-2.0, 100.0) == pytest.approx(-100.0)


def test_day_completion_rejects_extra_step_without_mutation(brain_csv: Path) -> None:
    session = brain_sessions.create_session(_parameters(), day_index=1)
    for _ in range(48):
        session.step(0.0)

    before = session.detail()
    assert before["status"]["complete"] is True
    with pytest.raises(brain_sessions.BrainEnvSessionComplete, match="already complete"):
        session.step(0.0)
    assert session.detail() == before


def test_session_lookup_and_deletion_are_explicit(brain_csv: Path) -> None:
    session = brain_sessions.create_session(_parameters(), day_index=1)

    assert brain_sessions.get_session(session.session_id) is session
    assert brain_sessions.drop_session(session.session_id) is True
    assert brain_sessions.drop_session(session.session_id) is False
    with pytest.raises(brain_sessions.BrainEnvSessionNotFound):
        brain_sessions.get_session(session.session_id)


def test_brain_env_flask_api_lifecycle(brain_csv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    monkeypatch.setattr(main, "PARAMETERS", _parameters())
    client = main.app.test_client()

    context_response = client.get("/api/brain-env/context")
    assert context_response.status_code == 200
    assert context_response.get_json()["days"][0]["day_index"] == 1

    create_response = client.post(
        "/api/brain-env/sessions",
        json={"day_index": 1, "starting_peak_kw": 75},
    )
    assert create_response.status_code == 201
    session_id = create_response.get_json()["status"]["session_id"]

    step_response = client.post(
        f"/api/brain-env/sessions/{session_id}/step",
        json={"action": 0.25},
    )
    assert step_response.status_code == 200
    assert step_response.get_json()["status"]["step_index"] == 1

    detail_response = client.get(f"/api/brain-env/sessions/{session_id}")
    assert detail_response.status_code == 200
    assert len(detail_response.get_json()["trace"]) == 1

    assert client.delete(f"/api/brain-env/sessions/{session_id}").status_code == 200
    assert client.get(f"/api/brain-env/sessions/{session_id}").status_code == 404


def test_brain_env_api_reports_invalid_create_and_step(brain_csv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    monkeypatch.setattr(main, "PARAMETERS", _parameters())
    client = main.app.test_client()

    assert client.post("/api/brain-env/sessions", json={"day_index": 999}).status_code == 422
    created = client.post("/api/brain-env/sessions", json={"day_index": 1}).get_json()
    session_id = created["status"]["session_id"]
    rejected = client.post(
        f"/api/brain-env/sessions/{session_id}/step",
        json={"action": 2},
    )
    assert rejected.status_code == 422
    assert rejected.get_json()["status"]["step_index"] == 0
