from datetime import date, timedelta

import pytest

from bess.brain.runtime import BrainDay, BrainRuntimeError, load_csv_days, run_controllers, split_billing_periods
from bess.core.settings import DEFAULT_PARAMETERS
from bess.paths import PROJECT_ROOT


def test_canonical_dataset_becomes_complete_sequential_periods() -> None:
    days = load_csv_days(PROJECT_ROOT / "data" / "offline_tande-15.csv")
    periods = split_billing_periods(days, reject_leftover=True)
    assert len(days) == 240
    assert all(len(period.days) == 30 for period in periods)


def test_brain1_and_brain2_share_runtime_but_own_independent_worlds() -> None:
    results, warnings = run_controllers(
        ["brain1", "brain2"],
        PROJECT_ROOT / "data" / "offline_tande-15.csv",
        dict(DEFAULT_PARAMETERS),
        PROJECT_ROOT / "checkpoints",
    )
    assert set(results) <= {"brain1", "brain2"}
    assert all(result["trace"] for result in results.values())
    assert all(result["kpi"]["ending_soc_compliant"] for result in results.values())
    assert isinstance(warnings, list)


def test_runtime_trace_preserves_requested_and_executed_actions() -> None:
    results, _ = run_controllers(
        ["brain1"],
        PROJECT_ROOT / "data" / "offline_tande-15.csv",
        dict(DEFAULT_PARAMETERS),
        PROJECT_ROOT / "checkpoints",
    )
    row = results["brain1"]["trace"][0]
    assert {
        "requested_action",
        "projected_action",
        "requested_battery_kw",
        "projected_battery_kw",
        "executed_battery_kw",
        "horizon_adjusted",
    } <= row.keys()


def test_dated_periods_are_calendar_chronological_and_incomplete_months_fail() -> None:
    def day(offset: int) -> BrainDay:
        stamp = date(2026, 1, 1) + timedelta(days=offset)
        return BrainDay(offset + 1, stamp.isoformat(), "working", (1.0,) * 48, (0.0,) * 48)

    complete = [day(offset) for offset in range(31 + 28)]
    periods = split_billing_periods(complete, reject_leftover=True)
    assert [period.key for period in periods] == ["2026-01", "2026-02"]
    assert periods[0].days[0].date_iso == "2026-01-01"
    assert periods[-1].days[-1].date_iso == "2026-02-28"

    with pytest.raises(BrainRuntimeError, match="incomplete calendar"):
        split_billing_periods(complete[:-1], reject_leftover=True)
