"""scenario_gen.py  two data sources for the experimental protocol.

1. REAL   : the measured 30-day dataset in data/ (held out  evaluation only).
2. SIMULATOR: parameterised synthetic month generator that reproduces the
   statistical structure of data_generation/generate_{load,pv}.py (1-shift
   factory, 500 kWp PV, weather mix) but is fully seedable so the DRL agent
   trains on an unlimited stream of unseen days.

Training NEVER sees the real dataset  that is the core of the
sim-to-real evaluation design.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

import numpy as np

from common import DATA_DIR, DT_HOURS, STEPS_PER_DAY, steps_per_day_from_dt

SUNRISE_H, SUNSET_H = 6.0, 18.0

WEATHER_MIX = [
    ("sunny_normal", 0.50),
    ("partly_cloudy", 0.25),
    ("rainy_heavy", 0.15),
    ("sunny_morning_only", 0.10),
]


@dataclass
class DayData:
    load: np.ndarray            # daily kW samples
    pv: np.ndarray              # daily kW samples
    day_type: str               # working | weekend | holiday
    weather: str
    day_index: int = 0
    date_iso: str | None = None  # calendar date (API-sourced data)


@dataclass
class MonthData:
    days: list[DayData] = field(default_factory=list)
    source: str = "sim"

    def __len__(self):
        return len(self.days)


# ---------------------------------------------------------------------------
# REAL dataset loader
# ---------------------------------------------------------------------------
def load_real_month() -> MonthData:
    cal = {}
    with open(DATA_DIR / "calendar_30days.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cal[int(r["day_index"])] = (r["day_type"], r["weather"])

    def _read_series(path, col):
        out = {}
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d, t = int(r["day_index"]), int(r["step"])
                if d not in out:
                    out[d] = []
                while len(out[d]) <= t:
                    out[d].append(0.0)
                out[d][t] = float(r[col])
        return out

    loads = _read_series(DATA_DIR / "load_30days.csv", "P_load_kW")
    pvs = _read_series(DATA_DIR / "pv_30days.csv", "P_pv_kW")
    month = MonthData(source="real")
    for d in sorted(cal):
        dt_, wx = cal[d]
        month.days.append(DayData(load=np.asarray(loads[d], dtype=np.float64), pv=np.asarray(pvs[d], dtype=np.float64),
                                  day_type=dt_, weather=wx, day_index=d))
    return month


# ---------------------------------------------------------------------------
# SIMULATOR  load profiles (mirrors data_generation/generate_load.py)
# ---------------------------------------------------------------------------
def _working_load(rng: np.random.Generator, scale: float, n_steps: int, dt_hours: float) -> np.ndarray:
    h = np.arange(n_steps) * dt_hours
    base = np.select(
        [h < 5, h < 7, h < 12, h < 13, h < 17, h < 18, h < 22],
        [100.0, 200.0 + 200.0 * (h - 5.0) / 2.0, 500.0, 250.0,
         500.0, 300.0, 150.0],
        default=100.0,
    ) * scale
    noise = rng.normal(0.0, 0.05, n_steps) * base
    val = np.maximum(0.0, base + noise)
    # motor startup spikes during production hours
    spike_mask = (h >= 7) & (h < 17) & (rng.random(n_steps) < 0.06)
    val[spike_mask] += 50.0 * scale
    return val


def _flat_load(rng, level_kw, sigma, n_steps: int):
    return np.maximum(0.0, rng.normal(level_kw, sigma, n_steps))


# ---------------------------------------------------------------------------
# SIMULATOR  PV profiles (mirrors data_generation/generate_pv.py)
# ---------------------------------------------------------------------------
def _bell(h, peak_kw, peak_h=12.0, sigma_h=2.6):
    v = peak_kw * np.exp(-((h - peak_h) ** 2) / (2 * sigma_h ** 2))
    v[(h < SUNRISE_H) | (h > SUNSET_H)] = 0.0
    return v


def _pv_profile(rng: np.random.Generator, weather: str,
                pv_scale: float, n_steps: int, dt_hours: float) -> np.ndarray:
    h = np.arange(n_steps) * dt_hours
    if weather == "sunny_normal":
        v = _bell(h, rng.uniform(350, 420))
        rel = 0.10
    elif weather == "partly_cloudy":
        v = _bell(h, rng.uniform(200, 280), sigma_h=2.8)
        rel = 0.15
    elif weather == "rainy_heavy":
        v = _bell(h, rng.uniform(40, 80), sigma_h=3.5)
        rel = 0.30
    else:  # sunny_morning_only  strong morning, collapses after midday
        v = _bell(h, 300.0, peak_h=10.0, sigma_h=2.0)
        after = h >= 12.0
        v[after] = np.minimum(v[after], 50.0)
        rel = 0.20
    mask = v > 0
    v[mask] *= np.maximum(0.0, 1.0 + rng.normal(0.0, rel, mask.sum()))
    return np.maximum(0.0, v) * pv_scale


def generate_sim_month(seed: int, n_days: int = 30, load_scale: float = 1.0,
                        pv_scale: float = 1.0,
                        holiday_prob: float = 0.03,
                        dt_hours: float = DT_HOURS) -> MonthData:
    """One synthetic month. Weekly pattern starts on Monday; weather is
    sampled i.i.d. from WEATHER_MIX (matches the real month's mix)."""
    rng = np.random.default_rng(seed)
    n_steps = steps_per_day_from_dt(dt_hours)
    names = [w for w, _ in WEATHER_MIX]
    probs = np.array([p for _, p in WEATHER_MIX])
    month = MonthData(source=f"sim_seed{seed}")
    for d in range(n_days):
        dow = d % 7
        if dow >= 5:
            day_type = "weekend"
        elif rng.random() < holiday_prob:
            day_type = "holiday"
        else:
            day_type = "working"
        weather = str(rng.choice(names, p=probs))
        if day_type == "working":
            load = _working_load(rng, load_scale, n_steps, dt_hours)
        elif day_type == "weekend":
            load = _flat_load(rng, 100.0 * load_scale, 5.0 * load_scale, n_steps)
        else:
            load = _flat_load(rng, 80.0 * load_scale, 4.0 * load_scale, n_steps)
        pv = _pv_profile(rng, weather, pv_scale, n_steps, dt_hours)
        month.days.append(DayData(load=load, pv=pv, day_type=day_type,
                                  weather=weather, day_index=d + 1))
    return month
