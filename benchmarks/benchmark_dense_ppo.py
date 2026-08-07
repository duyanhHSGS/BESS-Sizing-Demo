"""Manual micro-benchmark for dense PPO hot paths.

Run only with the project environment explicitly authorized:
    python tests/benchmark_dense_ppo.py
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bess.evaluation.benchmark import _demand_windows, _rolling_30_minute_average
from bess.agents.ppo_agent import PPOAgent


def _reference_rolling(values, dt):
    values = list(values)
    return [
        sum(values[step] * weight for step, weight in window)
        for window in _demand_windows(len(values), dt)
    ]


def _measure(label, fn, repeats):
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = time.perf_counter() - started
    print(f"{label:32s} {elapsed:8.3f}s | {repeats / elapsed:12,.0f} calls/s")


def main():
    rng = np.random.default_rng(42)
    dense_day = rng.uniform(0.0, 1500.0, 1440)
    dt = 1.0 / 60.0
    agent = PPOAgent(obs_dim=13, seed=42)
    obs = rng.normal(size=13).astype(np.float32)

    expected = _reference_rolling(dense_day, dt)
    actual = _rolling_30_minute_average(dense_day, dt)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-9)

    _measure(
        "legacy rolling, 1-min day",
        lambda: _reference_rolling(dense_day, dt),
        100,
    )
    _measure(
        "optimized rolling, 1-min day",
        lambda: _rolling_30_minute_average(dense_day, dt),
        100,
    )
    _measure(
        "full deterministic PPO action",
        lambda: agent.act(obs, deterministic=True),
        10_000,
    )
    _measure(
        "actor-only PPO prediction",
        lambda: agent.predict_action(obs),
        10_000,
    )


if __name__ == "__main__":
    main()
