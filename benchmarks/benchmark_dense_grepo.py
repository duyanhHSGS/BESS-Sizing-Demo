"""Opt-in GREPO phase benchmark.

Run only from the project virtual environment:
    python tests/benchmark_dense_grepo.py
"""
from __future__ import annotations

import time

import numpy as np
import torch

from bess.core.bess_env import BESSEnv, REACTIVE_OBSERVATION_DIM
from bess.core.common import load_system_config, make_bess_config
from bess.agents.grepo_agent import GREPOAgent
from bess.core.scenario_gen import DayData, MonthData


def benchmark_resolution(minutes, group=8):
    steps = 1440 // minutes
    x = np.arange(steps, dtype=np.float64)
    base = load_system_config()
    cfg = make_bess_config(base, 1000.0, 500.0, base.P_target_user)
    cfg.dt = minutes / 60.0
    month = MonthData(
        days=[DayData(
            load=650.0 + 120.0 * np.sin(2.0 * np.pi * x / steps),
            pv=np.maximum(0.0, 350.0 * np.sin(np.pi * x / steps)),
            day_type="working",
            weather="benchmark",
            day_index=0,
            date_iso="2026-01-01",
        )],
        source="grepo_benchmark",
    )

    def make_env():
        return BESSEnv(
            cfg,
            reference_power_kw=1000.0,
            initial_running_peak_kw=300.0,
            control_interval_minutes=float(minutes),
            record_trajectory=False,
        )

    agent = GREPOAgent(
        REACTIVE_OBSERVATION_DIM, n_group=group, seed=23, device="cpu"
    )
    noise = np.random.default_rng(23).normal(
        size=(group, steps)
    ).astype(np.float32)
    started = time.perf_counter()
    scalar = agent.collect_group_scalar_reference(
        make_env, month, soc_init=0.55, noise_g=noise
    )
    scalar_seconds = time.perf_counter() - started
    started = time.perf_counter()
    optimized = agent.collect_group(
        make_env, month, soc_init=0.55, noise_g=noise
    )
    optimized_seconds = time.perf_counter() - started
    parity = max(
        float(np.max(np.abs(actual - expected)))
        for actual, expected in zip(optimized, scalar)
    )
    print(
        f"{minutes:>2}m group={group}: scalar={scalar_seconds:.3f}s "
        f"optimized={optimized_seconds:.3f}s "
        f"speedup={scalar_seconds / optimized_seconds:.2f}x "
        f"max_delta={parity:.3g}"
    )

    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        learner = GREPOAgent(
            REACTIVE_OBSERVATION_DIM, n_group=group, seed=23, device=device
        )
        started = time.perf_counter()
        learner.update(*optimized)
        elapsed = time.perf_counter() - started
        print(
            f"  {device}: update={elapsed:.3f}s "
            f"transfer={learner.last_update_stats['batch_transfer_seconds']:.3f}s"
        )


if __name__ == "__main__":
    benchmark_resolution(1)
    benchmark_resolution(15)
