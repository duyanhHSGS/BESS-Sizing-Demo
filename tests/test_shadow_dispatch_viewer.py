from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from bess.core.scenario_gen import DayData, MonthData
from bess.paths import PROJECT_ROOT
from bess.shadow.shadow_runs import (
    ShadowRunError,
    _build_rollouts,
    _shadow_viewer_trace,
)


def _day(load, pv=None) -> DayData:
    load_array = np.asarray(load, dtype=np.float64)
    pv_array = np.zeros_like(load_array) if pv is None else np.asarray(pv, dtype=np.float64)
    return DayData(
        load=load_array,
        pv=pv_array,
        day_type="working",
        weather="test",
        day_index=1,
        date_iso="2026-01-01",
    )


def test_shadow_viewer_trace_uses_fixed_30_minute_meter_series_and_preserves_native_grids() -> None:
    day = _day([100.0, 300.0, 200.0, 400.0])
    cfg = SimpleNamespace(dt=0.25)
    no_bess = np.array([100.0, 300.0, 200.0, 400.0])
    policy_grid = np.array([80.0, 240.0, 160.0, 320.0])
    policy_soc = np.array([0.20, 0.30, 0.40, 0.50])

    trace = _shadow_viewer_trace(
        0,
        day,
        cfg,
        {},
        no_bess,
        policy_grid,
        policy_soc,
    )

    assert trace["no_bess_grid"] == [100.0, 300.0, 200.0, 400.0]
    assert trace["policy_grid"] == [80.0, 240.0, 160.0, 320.0]
    assert trace["no_bess_rolling_grid"] == [200.0, 200.0, 300.0, 300.0]
    assert trace["policy_rolling_grid"] == [160.0, 160.0, 240.0, 240.0]


def test_shadow_viewer_trace_pads_short_soc_for_plot_without_hiding_terminal_value() -> None:
    day = _day([100.0, 100.0, 100.0, 100.0])
    cfg = SimpleNamespace(dt=0.25)
    policy_soc = np.array([0.20, 0.35])

    trace = _shadow_viewer_trace(
        0,
        day,
        cfg,
        {},
        np.full(4, 100.0),
        np.full(4, 90.0),
        policy_soc,
    )

    assert trace["policy_soc"] == [20.0, 35.0, 35.0, 35.0]
    assert policy_soc[-1] == pytest.approx(0.35)


def test_shadow_viewer_trace_carries_eye6_and_planned_peak_as_separate_overlays() -> None:
    day = _day([500.0, 500.0, 500.0, 500.0])
    cfg = SimpleNamespace(dt=0.25)
    policy = {
        "brain_eye6_running_peak_days": [np.array([410.0, 410.0, 430.0, 430.0])],
        "brain_peak_guard_target_days": [np.full(4, 600.0)],
    }

    trace = _shadow_viewer_trace(
        0,
        day,
        cfg,
        policy,
        np.full(4, 500.0),
        np.full(4, 450.0),
        np.full(4, 0.5),
    )

    assert trace["ppo_eye6_running_peak_kw"] == [410.0, 410.0, 430.0, 430.0]
    assert trace["ppo_peak_guard_target_kw"] == [600.0, 600.0, 600.0, 600.0]
    assert trace["ppo_eye6_running_peak_kw"] != trace["ppo_peak_guard_target_kw"]


def test_shadow_viewer_trace_omits_optional_ppo_overlays_for_legacy_or_ppo2_rollout() -> None:
    day = _day([200.0, 200.0])
    trace = _shadow_viewer_trace(
        0,
        day,
        SimpleNamespace(dt=0.25),
        {},
        np.full(2, 200.0),
        np.full(2, 180.0),
        np.full(2, 0.4),
    )

    assert "ppo_eye6_running_peak_kw" not in trace
    assert "ppo_peak_guard_target_kw" not in trace


def test_shadow_viewer_trace_rejects_policy_grid_resolution_mismatch() -> None:
    day = _day([100.0, 100.0, 100.0, 100.0])
    with pytest.raises(ShadowRunError, match="policy grid resolution"):
        _shadow_viewer_trace(
            0,
            day,
            SimpleNamespace(dt=0.25),
            {},
            np.full(4, 100.0),
            np.full(3, 90.0),
            np.full(4, 0.4),
        )


def test_shadow_viewer_trace_rejects_eye6_resolution_mismatch() -> None:
    day = _day([100.0, 100.0, 100.0, 100.0])
    policy = {"brain_eye6_running_peak_days": [np.full(3, 100.0)]}
    with pytest.raises(ShadowRunError, match="Eye 6 trace resolution"):
        _shadow_viewer_trace(
            0,
            day,
            SimpleNamespace(dt=0.25),
            policy,
            np.full(4, 100.0),
            np.full(4, 90.0),
            np.full(4, 0.4),
        )


def test_shadow_viewer_trace_rejects_planned_target_resolution_mismatch() -> None:
    day = _day([100.0, 100.0, 100.0, 100.0])
    policy = {"brain_peak_guard_target_days": [np.full(3, 600.0)]}
    with pytest.raises(ShadowRunError, match="planned peak target resolution"):
        _shadow_viewer_trace(
            0,
            day,
            SimpleNamespace(dt=0.25),
            policy,
            np.full(4, 100.0),
            np.full(4, 90.0),
            np.full(4, 0.4),
        )


def test_shadow_rollout_requests_diagnostic_eye6_recording_without_changing_other_dispatch_inputs() -> None:
    month = MonthData(days=[_day(np.full(96, 500.0))], source="test")
    cfg = SimpleNamespace(dt=0.25)
    agent = SimpleNamespace()
    config = {
        "parameters": {},
        "e_cap_kwh": 1250.0,
        "p_rated_kw": 450.0,
        "policy": "policy_test.pt",
    }

    with (
        patch("bess.shadow.shadow_runs.build_dispatch_config", return_value=cfg),
        patch("bess.shadow.shadow_runs.load_policy", return_value=(agent, "ppo", {"p_ref_kw": 1000.0})),
        patch("bess.shadow.shadow_runs.validate_dispatch_sampling", return_value=30.0),
        patch("bess.shadow.shadow_runs.prepare_policy_forecast") as prepare,
        patch("bess.shadow.shadow_runs.run_drl_policy", return_value={}) as rollout,
    ):
        _build_rollouts(config, month, lambda *_args: None)

    prepare.assert_called_once_with("policy_test.pt", agent, {"p_ref_kw": 1000.0}, month, 1000.0)
    rollout.assert_called_once_with(
        month,
        cfg,
        agent,
        p_ref_kw=1000.0,
        record_brain_eye6=True,
    )


def test_shadow_ui_template_parses_with_jinja() -> None:
    import main

    main.app.jinja_env.get_template("index.html")


def test_shadow_ui_uses_dispatch_grade_meter_peak_eye6_target_and_bill_components() -> None:
    template = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    block = template.split("const shadowDispatchStyles = [", 1)[1].split("];", 1)[0]

    assert 'id="shadow-bill-strip"' in template
    assert 'id="shadow-dispatch-warnings"' in template
    assert 'key: "no_bess_rolling_grid"' in block
    assert 'key: "policy_rolling_grid"' in block
    assert 'key: "no_bess_mtd_peak"' in block
    assert 'key: "policy_mtd_peak"' in block
    assert 'key: "policy_soc"' in block
    assert 'key: "ppo_eye6_running_peak_kw"' in block
    assert 'key: "ppo_peak_guard_target_kw"' in block
    assert "lineWidth: 4, glow: true" in block
    assert "dash: [10, 4], lineWidth: 3" in block
    assert 'key: "no_bess_grid"' not in block
    assert 'key: "policy_grid"' not in block
    assert "TODO(SHADOW-DISPATCH-RICH)" in template


def test_shadow_ui_keeps_old_native_trace_compatibility_but_renders_meter_only() -> None:
    template = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")

    assert "function shadowFixedMeterSeries(values)" in template
    assert "shadowFixedMeterSeries(trace.no_bess_grid)" in template
    assert "shadowFixedMeterSeries(trace.policy_grid)" in template
    assert "meter-only grid view" in template


def test_shadow_ui_reuses_shared_daily_chart_without_changing_live_default_styles() -> None:
    template = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")

    assert "styles = dailyDispatchStyles" in template
    assert 'drawDailyDispatchChart("live-dispatch-chart", selected?.trace, policyLabel, liveDailyActiveSeries, selected, tariffConfig);' in template
    assert "shadowDispatchStyles," in template
    assert "series.lineWidth || 2" in template
    assert "series.dash || []" in template
    assert "series.glow" in template


def test_shadow_ui_mtd_bill_strip_uses_same_saved_calendar_month_semantics_as_shadow_report() -> None:
    template = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")

    assert 'const month = selected.date.slice(0, 7);' in template
    assert 'const throughSelected = shadowOkDays.filter' in template
    assert 'day.date.slice(0, 7) === month && day.date <= selected.date' in template
    assert 'config.parameters?.billing_mode === "2tc"' in template
    assert "billing_peak_penalty" in template
    assert 'billCard("Policy saving"' in template
