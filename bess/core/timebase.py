"""Resolution-independent time helpers for BESS simulation and billing."""
from __future__ import annotations

HOURS_PER_DAY = 24.0
MINUTES_PER_HOUR = 60.0
MINUTES_PER_DAY = HOURS_PER_DAY * MINUTES_PER_HOUR
DEMAND_WINDOW_MINUTES = 30.0


def steps_per_day_from_dt(dt_hours: float) -> int:
    """Return integer samples/day; reject a timestep that cannot tile 24 hours."""
    dt = float(dt_hours)
    if dt <= 0.0:
        raise ValueError(f"dt_hours must be positive, got {dt_hours!r}")
    exact_steps = HOURS_PER_DAY / dt
    steps = int(round(exact_steps))
    tolerance = max(1e-9, abs(exact_steps) * 1e-9)
    if steps <= 0 or abs(exact_steps - steps) > tolerance:
        raise ValueError(f"dt_hours={dt_hours!r} does not divide a 24-hour day")
    return steps


def dt_from_steps_per_day(steps_per_day: int) -> float:
    """Return hours/sample for an integer number of samples in one day."""
    steps = int(steps_per_day)
    if steps <= 0:
        raise ValueError(f"steps_per_day must be positive, got {steps_per_day!r}")
    return HOURS_PER_DAY / steps


def steps_for_minutes(duration_minutes: float, dt_hours: float) -> int:
    """Convert a duration to an exact integer number of samples."""
    step_minutes = float(dt_hours) * MINUTES_PER_HOUR
    if step_minutes <= 0.0:
        raise ValueError(f"dt_hours must be positive, got {dt_hours!r}")
    exact_steps = float(duration_minutes) / step_minutes
    steps = int(round(exact_steps))
    tolerance = max(1e-9, abs(exact_steps) * 1e-9)
    if steps <= 0 or abs(exact_steps - steps) > tolerance:
        raise ValueError(
            f"{duration_minutes:g} minutes is not an integer number of "
            f"{step_minutes:g}-minute samples"
        )
    return steps


def demand_window_steps(dt_hours: float) -> int:
    """Number of native samples in the 30-minute demand-charge window."""
    return steps_for_minutes(DEMAND_WINDOW_MINUTES, dt_hours)


def _clock_minutes(clock_text: str) -> float:
    hour_text, minute_text = clock_text.strip().split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    if hour == 24 and minute == 0:
        return MINUTES_PER_DAY
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        raise ValueError(f"invalid clock time {clock_text!r}")
    return hour * MINUTES_PER_HOUR + minute


def _clock_step(clock_text: str, dt_hours: float) -> int:
    total_minutes = _clock_minutes(clock_text)
    if total_minutes == 0.0:
        return 0
    return steps_for_minutes(total_minutes, dt_hours)


def _range_steps(range_text: str, dt_hours: float, steps_per_day: int) -> list[int]:
    start_text, end_text = range_text.strip().split("-", 1)
    start = _clock_step(start_text, dt_hours)
    end = _clock_step(end_text, dt_hours)
    if end <= start:
        end += steps_per_day
    return [step % steps_per_day for step in range(start, end)]


def build_tariff_windows(peak_ranges: str, off_ranges: str, dt_hours: float) -> dict[str, list[int] | int]:
    """Build tariff step indices from clock windows for any exact timestep.

    Example: ``dt=1/60`` gives 1-minute samples; ``00:00-06:00`` becomes
    steps 0..359. ``dt=1/120`` gives 0.5-minute samples and the same clock
    window becomes steps 0..719. No 15-minute step numbers are stored.
    """
    steps_per_day = steps_per_day_from_dt(dt_hours)
    peaks = [
        _range_steps(item, dt_hours, steps_per_day)
        for item in peak_ranges.split(",")
        if item.strip()
    ]
    off_steps: list[int] = []
    for item in off_ranges.split(","):
        if item.strip():
            off_steps.extend(_range_steps(item, dt_hours, steps_per_day))

    peaks.sort(key=lambda window: window[0])
    if not peaks:
        first_peak: list[int] = []
        second_peak: list[int] = []
    elif len(peaks) == 1:
        first_peak, second_peak = [], peaks[0]
    else:
        first_peak, second_peak = peaks[0], peaks[-1]

    off_steps = sorted(set(off_steps))
    peak_steps = set(first_peak) | set(second_peak)
    off_set = set(off_steps)
    intermediate_steps = [
        step for step in range(steps_per_day)
        if step not in peak_steps and step not in off_set
    ]
    morning_off_steps = [step for step in off_steps if step < steps_per_day // 2]
    off_peak_end_step = morning_off_steps[-1] + 1 if morning_off_steps else 0
    first_peak_start = first_peak[0] if first_peak else (second_peak[0] if second_peak else 0)
    second_peak_start = second_peak[0] if second_peak else first_peak_start

    return {
        "W1": first_peak,
        "W2": second_peak,
        "OFF": off_steps,
        "INTER": intermediate_steps,
        "W1_START": first_peak_start,
        "W2_START": second_peak_start,
        "OFF_PEAK_END_STEP": off_peak_end_step,
    }
