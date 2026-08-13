from __future__ import annotations

import csv
import math
from datetime import date, timedelta
from pathlib import Path

from bess.paths import PROJECT_ROOT
from typing import Any

import numpy as np

from bess.evaluation.benchmark import (
    DATA_PATH,
    _annotate_day_billing,
    _day_energy_cost,
    _demand_charge,
    _month_peaks,
    _month_start_day,
    _rounded_series,
    _rolling_30_minute_average,
    _to_float,
    selected_data_path,
)
from bess.training.training_checkpoints import CHECKPOINT_DIR, _load_checkpoint_meta
from bess.evaluation.baselines import run_drl_policy, run_sadrbc, validate_dispatch_sampling


BASE_DIR = PROJECT_ROOT

from bess.core.bess_env import OBSERVATION_DIM
from bess.core.common import (  # noqa: E402
    TOU_RULES,
    ensure_inside_directory,
    load_system_config,
    score_month,
)
from bess.core.timebase import dt_from_steps_per_day, steps_per_day_from_dt
from bess.agents.grepo_agent import GREPOAgent  # noqa: E402
from bess.agents.grepro_agent import GREPROAgent  # noqa: E402
from bess.agents.ppo_agent import PPOAgent  # noqa: E402
from bess.agents.ppo2_agent import PPO2InferenceAgent  # noqa: E402
from bess.agents.pro_agent import PROAgent  # noqa: E402
from bess.agents.sadrbc import SADRBCConfig  # noqa: E402
from bess.core.scenario_gen import DayData, MonthData  # noqa: E402


class DispatchRunWarning(RuntimeError):
    pass


def ensure_inside_sizing_demo(path: Path) -> Path:
    return ensure_inside_directory(path, BASE_DIR, label="Sizing_Demo")


def dataset_to_month(csv_path: Path = DATA_PATH) -> MonthData:
    rows_by_day: dict[int, dict[str, Any]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            day_index = int(row["day_index"])
            bucket = rows_by_day.setdefault(
                day_index,
                {
                    "day_type": row.get("day_type") or "working",
                    "date_iso": row.get("date_iso"),
                    "points": [],
                },
            )
            bucket["points"].append(
                (
                    int(row["step"]),
                    float(row["P_load_kW"]),
                    float(row["P_pv_kW"]),
                )
            )

    month = MonthData(source="sizing_demo")
    start = date(2026, 1, 1)
    for day_index in sorted(rows_by_day):
        raw = rows_by_day[day_index]
        points = sorted(raw["points"])
        load = np.asarray([p[1] for p in points], dtype=np.float64)
        pv = np.asarray([p[2] for p in points], dtype=np.float64)
        date_iso = raw.get("date_iso") or (start + timedelta(days=day_index - 1)).isoformat()
        month.days.append(
            DayData(
                load=load,
                pv=pv,
                day_type=raw["day_type"],
                weather="sizing_demo",
                day_index=day_index,
                date_iso=date_iso,
            )
        )
    return month


def build_dispatch_config(parameters: dict[str, Any], e_cap_kwh: float, p_rated_kw: float):
    base = load_system_config()
    dt_hours = _to_float(parameters.get("dt"), base.dt)
    dt_hours = dt_from_steps_per_day(steps_per_day_from_dt(dt_hours))
    TOU_RULES["sunday_no_peak"] = bool(parameters.get("billing_sunday"))
    return SADRBCConfig(
        {
            "E_cap_kWh": e_cap_kwh,
            "P_rated_kW": p_rated_kw,
            "eta_ch": _to_float(parameters.get("charge_efficiency"), base.eta_ch),
            "eta_dis": _to_float(parameters.get("discharge_efficiency"), base.eta_dis),
            "soc_min": _to_float(parameters.get("minimum_soc"), base.SOC_min),
            "soc_max": _to_float(parameters.get("maximum_soc"), base.SOC_max),
            "soc_safety_buffer": base.SOC_safety,
            "soc_eod": _to_float(parameters.get("required_final_soc"), base.SOC_eod),
            "soc_min_emergency": base.SOC_min_emergency,
            "dt_hours": dt_hours,
            "price_peak": _to_float(parameters.get("billing_expensive"), base.price_peak),
            "price_mid": _to_float(parameters.get("billing_normal"), base.price_mid),
            "price_off": _to_float(parameters.get("billing_cheap"), base.price_off),
            "T_cap": _to_float(parameters.get("billing_peak_penalty"), base.T_cap)
            if parameters.get("billing_mode") == "2tc"
            else 0.0,
            "FIT_PRICE": base.FIT_PRICE,
            "ENABLE_EXPORT": base.ENABLE_EXPORT,
            "P_target_user_kW": base.P_target_user,
            "V_NOMINAL": base.V_NOMINAL,
            "V_BLACKOUT_TH": base.V_BLACKOUT_TH,
            "T_DERATE": list(base.T_DERATE),
            "peak_windows": str(parameters.get("billing_windows_expensive", base.peak_windows)),
            "off_windows": str(parameters.get("billing_windows_cheap", base.off_windows)),
        }
    )


def load_policy(checkpoint_name: str, checkpoint_dir: Path = CHECKPOINT_DIR):
    checkpoint_dir = ensure_inside_sizing_demo(checkpoint_dir)
    path = ensure_inside_sizing_demo(checkpoint_dir / Path(checkpoint_name).name)
    if path.name != checkpoint_name or not path.exists() or not path.name.startswith("policy_"):
        raise DispatchRunWarning(f"Unknown local policy: {checkpoint_name}")

    algo, meta, error = _load_checkpoint_meta(path)
    if error:
        raise DispatchRunWarning(f"{checkpoint_name}: checkpoint is not loadable ({error})")
    algo = algo.lower()
    if algo == "grpo":
        raise DispatchRunWarning(f"{checkpoint_name}: GRPO is not implemented in this repo yet")
    if algo not in {"ppo", "ppo2", "grepo", "grepro", "pro"}:
        raise DispatchRunWarning(f"{checkpoint_name}: unsupported checkpoint algorithm {algo}")
    sampling_fields = {
        "native_dt_minutes",
        "control_dt_minutes",
        "native_steps_per_action",
    }
    if not sampling_fields.issubset(meta):
        raise DispatchRunWarning(
            f"{checkpoint_name}: legacy checkpoint has incomplete control "
            "sampling delta-t metadata; retrain it before Dispatch"
        )

    if algo == "ppo2":
        obs_dim = int(meta.get("obs_dim") or 17)
        agent = PPO2InferenceAgent(obs_dim=obs_dim)
    else:
        raw = None
        if algo in {"grepo", "grepro"}:
            import torch

            raw = torch.load(path, map_location="cpu")
        obs_dim = int(
            (raw or {}).get("obs_dim")
            or meta.get("obs_dim")
            or OBSERVATION_DIM
        )
        if obs_dim != OBSERVATION_DIM:
            raise DispatchRunWarning(
                f"{checkpoint_name}: legacy {obs_dim}-eye checkpoint is incompatible "
                f"with the current {OBSERVATION_DIM}-eye BrainEnv; retrain it"
            )
        if algo == "ppo":
            agent = PPOAgent(obs_dim=OBSERVATION_DIM)
        elif algo == "pro":
            agent = PROAgent(obs_dim=OBSERVATION_DIM)
        else:
            agent_class = GREPROAgent if algo == "grepro" else GREPOAgent
            agent = agent_class(
                obs_dim=OBSERVATION_DIM,
                n_group=int(meta.get("group") or meta.get("n_group") or 6),
                std=float((raw or {}).get("std") or meta.get("std") or 0.30),
                beta=float(meta.get("beta") or 0.5),
            )
    agent.load(path)
    return agent, algo, meta


def prepare_policy_forecast(
    checkpoint_name: str,
    agent,
    meta: dict,
    month: MonthData,
    p_ref_kw: float,
) -> None:
    """Reject the removed forecast-eye policy contract.

    Forecasts may still be used by external planners such as SADRBC, but they
    are never appended to the canonical seven-eye BrainEnv observation.
    """
    del agent, month, p_ref_kw
    if meta.get("obs_variant") == "fc":
        raise DispatchRunWarning(
            f"{checkpoint_name}: legacy forecast-eye checkpoints are incompatible "
            "with the fixed 7-eye BrainEnv; retrain this policy"
        )


def forecast_portability_error(meta: dict) -> str | None:
    if meta.get("reference_env") == "ppo2_senior_15m_v1":
        return None
    obs_dim = meta.get("obs_dim")
    if meta.get("obs_variant") == "fc":
        return "Legacy forecast-eye checkpoint is incompatible with the fixed 7-eye BrainEnv; retrain it."
    if obs_dim is not None and int(obs_dim) != OBSERVATION_DIM:
        return (
            f"Legacy {int(obs_dim)}-eye checkpoint is incompatible with the fixed "
            f"{OBSERVATION_DIM}-eye BrainEnv; retrain it."
        )
    return None


def run_policy_dispatch(
    checkpoint_name: str,
    parameters: dict[str, Any],
    checkpoint_dir: Path = CHECKPOINT_DIR,
    month: MonthData | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    agent, algo, meta = load_policy(checkpoint_name, checkpoint_dir)
    e_cap = meta.get("e_cap_kwh")
    p_rated = meta.get("p_rated_kw")
    if e_cap is None or p_rated is None:
        e_cap = _to_float(parameters.get("battery_capacity_kWh"), 0.0)
        p_rated = _to_float(parameters.get("battery_power_limit_kW"), 0.0)
        warnings.append(f"{checkpoint_name}: missing E/P metadata, using current UI sizing")
    requested_native_dt = _to_float(parameters.get("dt"), 0.0) * 60.0
    try:
        control_dt = validate_dispatch_sampling(meta, requested_native_dt)
    except ValueError as exc:
        raise DispatchRunWarning(f"{checkpoint_name}: {exc}") from exc
    cfg = build_dispatch_config(parameters, float(e_cap), float(p_rated))
    month = month or dataset_to_month(selected_data_path(parameters))
    expected_native_dt = float(meta["native_dt_minutes"])
    actual_native_dt = cfg.dt * 60.0
    if abs(expected_native_dt - actual_native_dt) > 1e-9:
        warnings.append(
            f"{checkpoint_name}: policy trained on {expected_native_dt:g}-minute "
            f"native data; running cross-resolution dispatch with "
            f"{control_dt:g}-minute decisions and {actual_native_dt:g}-minute physics"
        )
    p_ref = float(meta.get("p_ref_kw") or _policy_reference_kw(month))
    prepare_policy_forecast(checkpoint_name, agent, meta, month, p_ref)
    rollout = run_drl_policy(month, cfg, agent, p_ref_kw=p_ref)
    from bess.forecasting.sadrbc_forecast import rollout_activity

    days = policy_result_to_days(month, rollout, cfg, parameters)
    if algo == "ppo2" and meta.get("reference_env") == "ppo2_senior_15m_v1":
        from bess.evaluation.oracle.ppo2_oracle import score_month as score_ppo2_month

        kpi = score_ppo2_month(
            rollout["p_grid_days"],
            cfg,
            days=month.days,
            p_bess_days=rollout["p_bess_days"],
            soc_days=rollout["soc_days"],
            degradation_cost_per_kwh_discharged=float(
                meta.get("degradation_cost_per_kwh_discharged", 0.0)
            ),
        )
    else:
        kpi = score_month(rollout["p_grid_days"], cfg, month.days)
    return {
        "policy": checkpoint_name,
        "algo": algo,
        "meta": meta,
        "warnings": warnings,
        "days": days,
        "kpi": kpi,
        "activity": {
            **rollout_activity(rollout, cfg.dt),
            "blocked_action_pct": rollout.get("blocked_action_pct", 0.0),
        },
    }


def run_policies(
    policy_names: list[str],
    parameters: dict[str, Any],
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> tuple[dict[str, Any], list[str]]:
    month = dataset_to_month(selected_data_path(parameters))
    results = {}
    warnings = []
    sadrbc_forecast_spec = None
    sadrbc_p_ref = None
    for checkpoint_name in policy_names:
        if checkpoint_name == "sadrbc_v13":
            continue
        try:
            _, _, forecast_meta = load_policy(checkpoint_name, checkpoint_dir)
            contract = forecast_meta.get("sadrbc_forecast", {}) or {}
            if contract and sadrbc_forecast_spec is None:
                from bess.forecasting.sadrbc_forecast import SADRBCForecastSpec

                sadrbc_forecast_spec = SADRBCForecastSpec(
                    seed=int(contract.get("seed", 13_0013)),
                    load_sigma=float(contract.get("load_sigma", 0.05)),
                    pv_sigma=float(contract.get("pv_sigma", 0.15)),
                    rho=float(contract.get("rho", 0.90)),
                    replan_minutes=int(contract.get("replan_minutes", 60)),
                )
                sadrbc_p_ref = float(
                    forecast_meta.get("p_ref_kw") or _policy_reference_kw(month)
                )
        except DispatchRunWarning:
            continue
    for policy_name in policy_names:
        if policy_name == "sadrbc_v13":
            cfg = build_dispatch_config(
                parameters,
                _to_float(parameters.get("battery_capacity_kWh"), 0.0),
                _to_float(parameters.get("battery_power_limit_kW"), 0.0),
            )
            rollout = run_sadrbc(
                month, cfg, forecast_spec=sadrbc_forecast_spec,
                p_ref_kw=sadrbc_p_ref,
            )
            days = policy_result_to_days(month, rollout, cfg, parameters)
            results[policy_name] = {
                "policy": policy_name,
                "algo": "sadrbc",
                "meta": {
                    "e_cap_kwh": cfg.E_cap,
                    "p_rated_kw": cfg.P_rated_nominal,
                    "controller": "SADRBC v13",
                },
                "warnings": [],
                "days": days,
                "kpi": score_month(rollout["p_grid_days"], cfg, month.days),
            }
            continue
        try:
            result = run_policy_dispatch(policy_name, parameters, checkpoint_dir, month)
        except DispatchRunWarning as exc:
            warnings.append(str(exc))
            continue
        warnings.extend(result.get("warnings", []))
        results[policy_name] = result
    return results, warnings


def policy_result_to_days(
    month: MonthData,
    rollout: dict[str, Any],
    cfg,
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    days = []
    for index, day in enumerate(month.days):
        expected_steps = len(day.load)
        grid = np.maximum(0.0, np.asarray(rollout["p_grid_days"][index], dtype=np.float64))[:expected_steps]
        p_bess = np.asarray(rollout["p_bess_days"][index], dtype=np.float64)[:expected_steps]
        soc_raw = np.asarray(rollout["soc_days"][index], dtype=np.float64)
        soc = (
            soc_raw[:expected_steps]
            if len(soc_raw) >= expected_steps
            else np.pad(soc_raw, (0, expected_steps - len(soc_raw)), mode="edge")
        )
        day_row = {
            "day_index": day.day_index,
            "day_type": day.day_type,
            "date_iso": day.date_iso,
            "grid": _rounded_series(grid),
            "rolling_grid": _rounded_series(_rolling_30_minute_average(grid, cfg.dt)),
            "discharge": _rounded_series(np.maximum(0.0, p_bess)),
            "grid_charge": _rounded_series(np.maximum(0.0, -p_bess)),
            "solar_charge": [0.0 for _ in range(len(grid))],
            "soc": _rounded_series(np.clip(soc, 0.0, 1.0) * 100.0),
            "final_soc": round(float(soc_raw[-1]) * 100.0, 1) if len(soc_raw) else 0.0,
        }
        day_row["grid_kWh"] = round(float(np.sum(grid) * cfg.dt), 2)
        day_row["energy_cost_vnd"] = round(_day_energy_cost(day_row, parameters, cfg.dt))
        days.append(day_row)
    month_peaks = _month_peaks(days, cfg.dt)
    for day_row in days:
        month_peak = month_peaks.get(_month_start_day(day_row["day_index"]))
        day_row["month_peak"] = month_peak
        day_row["demand_charge_vnd"] = round(_demand_charge(parameters, month_peak["value_kW"])) if month_peak else 0
        day_row["wear_cost_note"] = "Policy bill excludes battery wear cost."
    _annotate_day_billing(days, parameters, cfg.dt)
    return days


def _policy_reference_kw(month: MonthData) -> float:
    peak = max((float(np.max(np.maximum(0.0, day.load - day.pv))) for day in month.days), default=500.0)
    return max(500.0, math.ceil(peak / 500.0) * 500.0)
