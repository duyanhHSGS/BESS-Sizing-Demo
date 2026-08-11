from pathlib import Path
from types import SimpleNamespace

from bess.brain.runtime import BrainDay, BrainPeriod
from bess.core.config import BrainConfig
from bess.dispatch import brain_dispatch
from bess.evaluation import brain_benchmarking
from bess.integrations.thingsboard_connector import TelemetryDay
from bess.shadow import brain_live_runs, brain_shadow_runs
from bess.training.runners.train_brain3 import _decision_rollout


def _result(controller_id: str) -> dict:
    return {
        "controller": controller_id,
        "meta": {"display_name": controller_id},
        "trace": [{"day_index": 1, "date_iso": "2026-01-01", "grid_import_kw": 1.0}],
        "kpi": {"bess_cost_vnd": 1.0},
    }


def test_dispatch_forwards_controller_ids_to_the_shared_runtime(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "source.csv"
    captured = {}
    monkeypatch.setattr(brain_dispatch, "selected_data_path", lambda _parameters: csv_path)

    def fake_run(controller_ids, source, parameters, checkpoint_dir):
        captured.update(ids=controller_ids, source=source, parameters=parameters, checkpoints=checkpoint_dir)
        return {controller_id: _result(controller_id) for controller_id in controller_ids}, []

    monkeypatch.setattr(brain_dispatch, "run_controllers", fake_run)
    results, warnings = brain_dispatch.run_dispatch(["brain1", "brain2"], {"site": "same"})
    assert list(results) == ["brain1", "brain2"]
    assert warnings == []
    assert captured["ids"] == ["brain1", "brain2"]
    assert captured["source"] == csv_path


def test_live_reveals_the_same_day_for_each_independent_controller(monkeypatch) -> None:
    results = {controller_id: _result(controller_id) for controller_id in ("brain1", "brain2")}
    monkeypatch.setattr(brain_live_runs, "selected_data_path", lambda _parameters: Path("same.csv"))
    monkeypatch.setattr(brain_live_runs, "run_controllers", lambda *_args: (results, []))
    session = brain_live_runs.create_session({"controllers": ["brain1", "brain2"]}, {})
    day = session.step()
    assert day is not None
    assert set(day["controllers"]) == {"brain1", "brain2"}
    assert session.public()["complete"] is True
    brain_live_runs.drop_session(session.id)


def test_benchmark_saves_comparable_controller_results(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "same.csv"
    csv_path.write_text("measured", encoding="utf-8")
    monkeypatch.setattr(brain_benchmarking, "selected_data_path", lambda _parameters: csv_path)
    monkeypatch.setattr(
        brain_benchmarking,
        "roster",
        lambda _parameters: [
            {"id": "brain1", "display_name": "Brain 1"},
            {"id": "brain2", "display_name": "Brain 2"},
        ],
    )
    monkeypatch.setattr(
        brain_benchmarking,
        "run_controllers",
        lambda ids, *_args: ({ids[0]: _result(ids[0])}, []),
    )
    monkeypatch.setattr(brain_benchmarking.oracle_cache, "selected_csv_has_cache", lambda _p: False)
    monkeypatch.setattr(brain_benchmarking.benchmark_store, "save_result", lambda result: {"id": "saved", **result})
    result = brain_benchmarking.run_and_save({}, ["brain1", "brain2"], lambda *_args: None, lambda: False)
    assert result["id"] == "saved"
    assert [row["id"] for row in result["leaderboard"]] == ["brain1", "brain2"]
    assert result["snapshot"]["controllers"] == ["brain1", "brain2"]


def test_thingsboard_shadow_converts_only_connector_days_to_brain_days(monkeypatch) -> None:
    telemetry = TelemetryDay(load=[2.0] * 48, pv=[0.5] * 48, day_type="working", date_iso="2026-01-01")
    monkeypatch.setattr(
        brain_shadow_runs.thingsboard_connector,
        "fetch_days",
        lambda _start, _end: ([telemetry], {}, {}),
    )
    days = brain_shadow_runs._source_days(
        {"source_kind": "thingsboard", "parameters": {}}, "2026-01-01", "2026-01-01"
    )
    assert len(days) == 1
    assert days[0].net_load_kw == (1.5,) * 48


def test_brain3_held_decisions_create_one_summed_transition_per_control_interval() -> None:
    config = BrainConfig(
        battery_capacity_kwh=100.0,
        battery_power_kw=20.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        minimum_soc=0.1,
        maximum_soc=0.9,
        required_final_soc=0.5,
        timestep_hours=0.5,
        battery_wear_vnd_per_kwh=0.0,
        demand_charge_vnd_per_kw=0.0,
        cheap_tariff_vnd_per_kwh=1.0,
        normal_tariff_vnd_per_kwh=2.0,
        expensive_tariff_vnd_per_kwh=3.0,
        cheap_windows="00:00-06:00",
        expensive_windows="17:00-22:00",
        sunday_no_peak=False,
        billing_mode="tou",
    )
    day = BrainDay(1, None, "working", (10.0,) * 48, (0.0,) * 48)
    period = BrainPeriod("period-001", (day,))

    class RecordingAgent:
        def __init__(self):
            self.transitions = []

        def decide(self, _observation, *, explore):
            assert explore is True
            return SimpleNamespace(action=0.0, action_index=1)

        def remember(self, *transition):
            self.transitions.append(transition)

        def learn(self):
            return None

    agent = RecordingAgent()
    decisions, savings, losses = _decision_rollout(agent, period, config, 4, learn=True)
    assert decisions == 12
    assert len(agent.transitions) == 12
    assert sum(transition[2] for transition in agent.transitions) == savings == 0.0
    assert agent.transitions[-1][-1] is True
    assert losses == []
