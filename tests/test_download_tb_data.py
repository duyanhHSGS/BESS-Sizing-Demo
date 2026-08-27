from __future__ import annotations

import math
from pathlib import Path

from scripts.download_tb_data import BASE_DIR, _sensor_quality_issue

INTERVAL_MINUTES = 15
SAMPLES_PER_DAY = 24 * 60 // INTERVAL_MINUTES


def test_downloader_resolves_repo_root_without_bess_import() -> None:
    assert BASE_DIR == Path(__file__).resolve().parents[1]


def test_sensor_quality_accepts_normal_factory_day() -> None:
    load = [40.0 + (step % 20) * 3.0 for step in range(SAMPLES_PER_DAY)]
    pv = [max(0.0, 180.0 - abs(step - 48) * 8.0) for step in range(SAMPLES_PER_DAY)]

    assert _sensor_quality_issue(load, pv, INTERVAL_MINUTES) is None


def test_sensor_quality_accepts_true_zero_shutdown_day() -> None:
    load = [0.0] * SAMPLES_PER_DAY
    pv = [0.0] * SAMPLES_PER_DAY

    assert _sensor_quality_issue(load, pv, INTERVAL_MINUTES) is None


def test_sensor_quality_accepts_varying_low_load_day() -> None:
    load = [0.05 + (step % 7) * 0.01 for step in range(SAMPLES_PER_DAY)]
    pv = [0.0] * SAMPLES_PER_DAY

    assert _sensor_quality_issue(load, pv, INTERVAL_MINUTES) is None


def test_sensor_quality_rejects_nonfinite_load() -> None:
    load = [40.0] * SAMPLES_PER_DAY
    load[10] = math.nan
    pv = [0.0] * SAMPLES_PER_DAY

    assert _sensor_quality_issue(load, pv, INTERVAL_MINUTES) == "load contains non-finite values"


def test_sensor_quality_rejects_nonfinite_pv() -> None:
    load = [40.0 + step for step in range(SAMPLES_PER_DAY)]
    pv = [0.0] * SAMPLES_PER_DAY
    pv[10] = math.inf

    assert _sensor_quality_issue(load, pv, INTERVAL_MINUTES) == "PV contains non-finite values"


def test_sensor_quality_rejects_negative_load() -> None:
    load = [40.0 + step for step in range(SAMPLES_PER_DAY)]
    load[10] = -0.1
    pv = [0.0] * SAMPLES_PER_DAY

    assert _sensor_quality_issue(load, pv, INTERVAL_MINUTES) == "load contains negative values"


def test_sensor_quality_rejects_negative_pv() -> None:
    load = [40.0 + step for step in range(SAMPLES_PER_DAY)]
    pv = [0.0] * SAMPLES_PER_DAY
    pv[10] = -0.1

    assert _sensor_quality_issue(load, pv, INTERVAL_MINUTES) == "PV contains negative values"


def test_sensor_quality_allows_frozen_load_at_exact_limit() -> None:
    load = [5.0] * 16 + [20.0 + step for step in range(SAMPLES_PER_DAY - 16)]
    pv = [0.0] * SAMPLES_PER_DAY

    assert _sensor_quality_issue(load, pv, INTERVAL_MINUTES) is None


def test_sensor_quality_rejects_frozen_load_beyond_limit() -> None:
    load = [5.0] * 17 + [20.0 + step for step in range(SAMPLES_PER_DAY - 17)]
    pv = [0.0] * SAMPLES_PER_DAY

    assert _sensor_quality_issue(load, pv, INTERVAL_MINUTES) == "load sensor frozen for 255 minutes"


def test_sensor_quality_allows_load_pv_mirroring_at_exact_limit() -> None:
    load = [100.0 + step for step in range(SAMPLES_PER_DAY)]
    pv = [0.0] * SAMPLES_PER_DAY
    for step in range(4):
        load[step] = 31.98 + step
        pv[step] = 31.98 + step

    assert _sensor_quality_issue(load, pv, INTERVAL_MINUTES) is None


def test_sensor_quality_rejects_load_pv_mirroring_beyond_limit() -> None:
    load = [100.0 + step for step in range(SAMPLES_PER_DAY)]
    pv = [0.0] * SAMPLES_PER_DAY
    for step in range(5):
        load[step] = 31.98 + step
        pv[step] = 31.98 + step

    assert _sensor_quality_issue(load, pv, INTERVAL_MINUTES) == (
        "load/PV channels mirror each other for 75 minutes"
    )


def test_sensor_quality_rejects_old_point_zero_seven_failure_pattern() -> None:
    load = [0.07] * SAMPLES_PER_DAY
    pv = [0.07] * SAMPLES_PER_DAY

    issue = _sensor_quality_issue(load, pv, INTERVAL_MINUTES)

    assert issue is not None
    assert "frozen" in issue or "mirror" in issue


def test_sensor_quality_rejects_mismatched_channel_lengths() -> None:
    assert _sensor_quality_issue([1.0, 2.0], [1.0], INTERVAL_MINUTES) == (
        "load/PV sample counts differ"
    )


def test_sensor_quality_rejects_empty_day() -> None:
    assert _sensor_quality_issue([], [], INTERVAL_MINUTES) == "day contains no samples"


def test_sensor_quality_requires_positive_interval() -> None:
    # TODO(DATA-CLEAN): keep this guard if future downloaders make interval configurable.
    try:
        _sensor_quality_issue([1.0], [0.0], 0)
    except ValueError as error:
        assert str(error) == "interval_min must be positive"
    else:
        raise AssertionError("expected invalid interval to raise ValueError")
