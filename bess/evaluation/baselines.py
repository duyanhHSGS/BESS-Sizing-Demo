"""Reference and learned-policy rollouts used by evaluation and dispatch.

No-BESS is a neutral reference. Learned checkpoints are PPO or PPO2; PPO uses
canonical BrainEnv while PPO2 keeps its isolated senior-reference environment.
The cached Oracle LP is supplied separately by evaluation code.
"""
from __future__ import annotations

import time

import numpy as np

from bess.core.bess_env import OBSERVATION_DIM
from bess.core.brain_runtime import (
    BrainTrajectoryRecorder,
    make_brain_env,
    native_steps_per_action,
    observation_array,
    step_brain_control,
)
from bess.core.common import validate_control_interval_minutes
from bess.core.scenario_gen import MonthData


def _result(p_grid_days, soc_days, p_bess_days):
    return {
        "p_grid_days": p_grid_days,
        "soc_days": soc_days,
        "p_bess_days": p_bess_days,
    }


def run_no_bess(month: MonthData, cfg) -> dict:
    grids, socs, pbs = [], [], []
    for day in month.days:
        grid = np.maximum(0.0, day.load - day.pv)
        n_steps = len(grid)
        grids.append(grid)
        socs.append(np.full(n_steps + 1, cfg.SOC_min))
        pbs.append(np.zeros(n_steps))
    return _result(grids, socs, pbs)


def validate_dispatch_sampling(meta: dict, native_dt_minutes: float) -> float:
    """Return a compatible policy control interval for native dispatch data."""
    control_dt_minutes = float(meta.get("control_dt_minutes", native_dt_minutes))
    return validate_control_interval_minutes(native_dt_minutes, control_dt_minutes)


def run_drl_policy(
    month: MonthData,
    cfg,
    agent,
    p_ref_kw: float = 500.0,
    measure_latency: bool = False,
    deterministic: bool = True,
) -> dict:
    """Run PPO or PPO2 through the environment contract stored in checkpoint meta."""
    meta = getattr(agent, "meta", {}) or {}
    native_dt_minutes = cfg.dt * 60.0
    control_dt_minutes = validate_dispatch_sampling(meta, native_dt_minutes)

    if meta.get("reference_env") == "ppo2_senior_15m_v1":
        from bess.core.ppo2_env import PPO2Env

        env = PPO2Env(
            cfg,
            p_ref_kw=p_ref_kw,
            degradation_cost_per_kwh_discharged=float(
                meta.get("degradation_cost_per_kwh_discharged", 0.0)
            ),
        )
        observation = env.reset(month)
        done = False
        latencies: list[float] = []
        decisions = 0
        while not done:
            started = time.perf_counter()
            raw_action = agent.act(observation, deterministic=deterministic)
            action = raw_action[0] if isinstance(raw_action, tuple) else raw_action
            if measure_latency:
                latencies.append((time.perf_counter() - started) * 1e3)
            observation, _, done, _ = env.step(action)
            decisions += 1
        out = _result(env.log_grid, env.log_soc, env.log_pbess)
        out["blocked_action_count"] = 0
        out["decision_count"] = decisions
        out["blocked_action_pct"] = 0.0
        if measure_latency:
            out["latency_ms_mean"] = float(np.mean(latencies))
            out["latency_ms_max"] = float(np.max(latencies))
        return out

    checkpoint_obs_dim = meta.get("obs_dim")
    if checkpoint_obs_dim is not None and int(checkpoint_obs_dim) != OBSERVATION_DIM:
        raise ValueError(
            f"checkpoint uses legacy observation dimension {checkpoint_obs_dim}; "
            f"current BrainEnv requires {OBSERVATION_DIM}. Retrain this policy."
        )

    wear_cost = float(meta.get("battery_wear_cost", 0.0))
    env = make_brain_env(
        month,
        cfg,
        power_scale_kw=p_ref_kw,
        battery_wear_vnd_per_kwh=wear_cost,
    )
    observation = observation_array(env.reset())
    recorder = BrainTrajectoryRecorder(month, env.bess_world.state_of_charge)
    native_steps = native_steps_per_action(cfg.dt, control_dt_minutes)

    latencies: list[float] = []
    blocked_actions = 0
    decisions = 0
    done = False
    while not done:
        started = time.perf_counter()
        if deterministic and hasattr(agent, "predict_action"):
            policy_action = float(agent.predict_action(observation))
        else:
            raw_action = agent.act(observation, deterministic=deterministic)
            policy_action = float(raw_action[0] if isinstance(raw_action, tuple) else raw_action)
        action = float(np.clip(policy_action, -1.0, 1.0))
        if measure_latency:
            latencies.append((time.perf_counter() - started) * 1e3)

        transition = step_brain_control(
            env,
            action,
            native_steps=native_steps,
            recorder=recorder,
        )
        blocked_actions += int(transition.adjusted_action)
        decisions += 1
        done = transition.done
        if not done:
            if transition.next_observation is None:
                raise RuntimeError("BrainEnv omitted the next observation before episode end")
            observation = observation_array(transition.next_observation)

    out = _result(
        recorder.grid_import_days,
        recorder.state_of_charge_days,
        recorder.battery_power_days,
    )
    out["blocked_action_count"] = blocked_actions
    out["decision_count"] = decisions
    out["blocked_action_pct"] = 100.0 * blocked_actions / max(1, decisions)
    if measure_latency:
        out["latency_ms_mean"] = float(np.mean(latencies))
        out["latency_ms_max"] = float(np.max(latencies))
    return out


def rollout_activity(result: dict, dt_hours: float) -> dict:
    """Summarize battery movement without depending on any controller family."""
    rows = [np.asarray(row, dtype=np.float64) for row in result["p_bess_days"]]
    throughput = float(sum(np.sum(np.abs(row)) * dt_hours for row in rows))
    total_rows = sum(len(row) for row in rows)
    mean_abs = throughput / max(dt_hours * total_rows, 1e-12)
    soc_values = (
        np.concatenate([np.asarray(row, dtype=np.float64) for row in result["soc_days"]])
        if result["soc_days"]
        else np.asarray([0.0])
    )
    return {
        "throughput_kwh": throughput,
        "mean_abs_p_bess_kw": mean_abs,
        "soc_span_pct": float((soc_values.max() - soc_values.min()) * 100.0),
    }
