"""Stateful, day-at-a-time live replay sessions for Sizing Demo.

The selected CSV is snapshotted when a session is created.  SADRBC and the
selected DRL policy then keep their physical/controller state between day
steps, matching a rolling deployment instead of recomputing isolated days.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import numpy as np

from bess.evaluation.baselines import validate_dispatch_sampling
from bess.core.bess_env import BESSEnv
from bess.core.common import check_hard_constraints, score_month, tariff_vector_day
from bess.dispatch.dispatch_runner import (
    DispatchRunWarning,
    build_dispatch_config,
    dataset_to_month,
    load_policy,
    prepare_policy_forecast,
)
from bess.evaluation.benchmark import selected_data_path
from bess.core.settings import PPO_GAMMA
from bess.forecasting.sadrbc_forecast import (
    SADRBCForecastSpec,
    SADRBCResidualEnv,
    build_sadrbc_forecast_baseline,
)


_SESSIONS: dict[str, "LiveRunSession"] = {}
_SESSIONS_LOCK = threading.Lock()


def _number(parameters: dict[str, Any], key: str, fallback: float = 0.0) -> float:
    try:
        return float(parameters.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


class LiveRunSession:
    def __init__(self, policy_name: str, parameters: dict[str, Any]):
        self.id = uuid.uuid4().hex[:10]
        self.policy_name = policy_name
        self.parameters = dict(parameters)
        self.month = dataset_to_month(selected_data_path(parameters))
        if not self.month.days:
            raise ValueError("The selected CSV contains no complete live-run days.")

        e_cap = _number(parameters, "battery_capacity_kWh")
        p_rated = _number(parameters, "battery_power_limit_kW")
        self.sadrbc_cfg = build_dispatch_config(parameters, e_cap, p_rated)

        agent, algo, meta = load_policy(policy_name)
        policy_e = float(meta.get("e_cap_kwh") or e_cap)
        policy_p = float(meta.get("p_rated_kw") or p_rated)
        self.policy_cfg = build_dispatch_config(parameters, policy_e, policy_p)
        native_minutes = self.policy_cfg.dt * 60.0
        control_minutes = validate_dispatch_sampling(meta, native_minutes)
        p_ref = float(meta.get("p_ref_kw") or self._policy_reference_kw())
        prepare_policy_forecast(policy_name, agent, meta, self.month, p_ref)
        forecast = meta.get("sadrbc_forecast", {}) or {}
        forecast_spec = SADRBCForecastSpec(
            seed=int(forecast.get("seed", 13_0013)),
            load_sigma=float(forecast.get("load_sigma", 0.05)),
            pv_sigma=float(forecast.get("pv_sigma", 0.15)),
            rho=float(forecast.get("rho", 0.90)),
            replan_minutes=int(forecast.get("replan_minutes", 60)),
        )
        self.sadrbc_rollout = build_sadrbc_forecast_baseline(
            self.month, self.sadrbc_cfg, p_ref_kw=p_ref, spec=forecast_spec
        )
        self.agent = agent
        self.algo = algo
        self.meta = meta
        env_class = SADRBCResidualEnv if meta.get("controller") == "sadrbc_residual" else BESSEnv
        env_kwargs = {}
        if env_class is SADRBCResidualEnv:
            env_kwargs = {
                "residual_limit": float(meta.get("residual_limit", 0.20)),
                "forecast_spec": forecast_spec,
            }
        self.env = env_class(
            self.policy_cfg,
            p_ref_kw=p_ref,
            use_forecast=meta.get("obs_variant") == "fc",
            d_run_init_kw=meta.get("d_run_init_kw"),
            gamma=float(meta.get("gamma", PPO_GAMMA)),
            control_dt_minutes=control_minutes,
            **env_kwargs,
        )
        self.observation = self.env.reset(self.month)

        self.cursor = 0
        self.grids: dict[str, list[np.ndarray]] = {
            "no_bess": [],
            "sadrbc_v13": [],
            policy_name: [],
        }
        self.day_log: list[dict[str, Any]] = []
        self.error: str | None = None
        self.auto_interval: float | None = None
        self.auto_thread: threading.Thread | None = None
        self.auto_stop = threading.Event()
        self.lock = threading.Lock()
        self.created = time.time()

    def _policy_reference_kw(self) -> float:
        peak = max(
            float(np.max(np.maximum(0.0, day.load - day.pv)))
            for day in self.month.days
        )
        return max(500.0, np.ceil(peak / 500.0) * 500.0)

    def _advance_policy_day(self, day_index: int) -> tuple[np.ndarray, np.ndarray]:
        done = False
        while not done and self.env.day == day_index:
            if hasattr(self.agent, "predict_action"):
                action = self.agent.predict_action(self.observation)
            else:
                action, _, _ = self.agent.act(self.observation, deterministic=True)
            self.observation, _, done, _ = self.env.step(action)
        return (
            np.asarray(self.env.log_grid[day_index], dtype=np.float64),
            np.asarray(self.env.log_soc[day_index], dtype=np.float64),
        )

    def _method_kpi(self, method: str, cfg) -> dict[str, float]:
        return score_month(
            self.grids[method],
            cfg,
            self.month.days[: self.cursor + 1],
        )

    def _method_row(self, method: str, grid: np.ndarray, soc: np.ndarray | None, cfg) -> dict:
        tariff = tariff_vector_day(cfg, self.month.days[self.cursor])
        kpi = self._method_kpi(method, cfg)
        violations = 0
        if soc is not None:
            checks = check_hard_constraints([grid], [soc], cfg)
            violations = sum(checks.values())
        return {
            "day_energy_cost_vnd": round(float(np.sum(np.maximum(0.0, grid) * tariff) * cfg.dt)),
            "mtd_total_vnd": round(float(kpi["total_cost_vnd"])),
            "mtd_energy_vnd": round(float(kpi["energy_cost_vnd"])),
            "mtd_demand_vnd": round(float(kpi["demand_cost_vnd"])),
            "mtd_pmax_kw": round(float(kpi["pmax_month_kw"]), 2),
            "soc_end_pct": None if soc is None else round(float(soc[-1]) * 100.0, 2),
            "violation_days": int(violations),
        }

    def step_day(self) -> dict | None:
        with self.lock:
            if self.cursor >= len(self.month.days):
                self.auto_interval = None
                return None

            index = self.cursor
            day = self.month.days[index]
            no_bess_grid = np.maximum(0.0, day.load - day.pv)

            sadrbc_grid = self.sadrbc_rollout.p_grid_days[index]
            sadrbc_soc = self.sadrbc_rollout.soc_days[index]
            policy_grid, policy_soc = self._advance_policy_day(index)

            self.grids["no_bess"].append(no_bess_grid)
            self.grids["sadrbc_v13"].append(sadrbc_grid)
            self.grids[self.policy_name].append(policy_grid)
            entry = {
                "day": index + 1,
                "day_index": day.day_index,
                "date": day.date_iso,
                "day_type": day.day_type,
                "trace": {
                    "load": np.round(np.asarray(day.load, dtype=float), 3).tolist(),
                    "pv": np.round(np.asarray(day.pv, dtype=float), 3).tolist(),
                    "no_bess_grid": np.round(no_bess_grid, 3).tolist(),
                    "sadrbc_grid": np.round(sadrbc_grid, 3).tolist(),
                    "sadrbc_soc": np.round(np.asarray(sadrbc_soc) * 100.0, 3).tolist(),
                    "policy_grid": np.round(policy_grid, 3).tolist(),
                    "policy_soc": np.round(np.asarray(policy_soc) * 100.0, 3).tolist(),
                },
                "methods": {
                    "no_bess": self._method_row("no_bess", no_bess_grid, None, self.sadrbc_cfg),
                    "sadrbc_v13": self._method_row(
                        "sadrbc_v13", sadrbc_grid, sadrbc_soc, self.sadrbc_cfg
                    ),
                    self.policy_name: self._method_row(
                        self.policy_name, policy_grid, policy_soc, self.policy_cfg
                    ),
                },
            }
            self.day_log.append(entry)
            self.cursor += 1
            return entry

    def start_auto(self, interval_s: float) -> None:
        self.stop_auto()
        stop_event = threading.Event()
        self.auto_stop = stop_event
        self.auto_interval = max(1.0, float(interval_s))
        self.error = None

        def run() -> None:
            while not stop_event.is_set():
                try:
                    if self.step_day() is None:
                        break
                except Exception as exc:  # noqa: BLE001
                    self.error = str(exc)[:500]
                    break
                if stop_event.wait(self.auto_interval or 1.0):
                    break
            if self.auto_stop is stop_event:
                self.auto_interval = None

        self.auto_thread = threading.Thread(target=run, daemon=True)
        self.auto_thread.start()

    def stop_auto(self) -> None:
        self.auto_stop.set()
        self.auto_interval = None

    def status(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.parameters.get("selected_data_csv"),
            "policy": self.policy_name,
            "algo": self.algo,
            "days_done": len(self.day_log),
            "days_total": len(self.month.days),
            "auto_interval_s": self.auto_interval,
            "error": self.error,
            "created": self.created,
            "methods": ["no_bess", "sadrbc_v13", self.policy_name],
            "method_labels": {
                "no_bess": "No BESS",
                "sadrbc_v13": "SADRBC v13",
                self.policy_name: self.policy_name,
            },
            "sizing": {
                "sadrbc_v13": {
                    "e_cap_kwh": self.sadrbc_cfg.E_cap,
                    "p_rated_kw": self.sadrbc_cfg.P_rated_nominal,
                },
                self.policy_name: {
                    "e_cap_kwh": self.policy_cfg.E_cap,
                    "p_rated_kw": self.policy_cfg.P_rated_nominal,
                },
            },
            "sadrbc_forecast": self.sadrbc_rollout.spec.public(
                self.sadrbc_rollout.mode
            ),
        }


def create_session(policy_name: str, parameters: dict[str, Any]) -> LiveRunSession:
    session = LiveRunSession(policy_name, parameters)
    with _SESSIONS_LOCK:
        _SESSIONS[session.id] = session
    return session


def get_session(session_id: str) -> LiveRunSession | None:
    with _SESSIONS_LOCK:
        return _SESSIONS.get(session_id)


def list_sessions() -> list[dict[str, Any]]:
    with _SESSIONS_LOCK:
        sessions = list(_SESSIONS.values())
    return [session.status() for session in sessions]


def drop_session(session_id: str) -> bool:
    with _SESSIONS_LOCK:
        session = _SESSIONS.pop(session_id, None)
    if session is None:
        return False
    session.stop_auto()
    return True
