from __future__ import annotations

import csv
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from benchmark import (
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
from training_checkpoints import CHECKPOINT_DIR, _load_checkpoint_meta


BASE_DIR = Path(__file__).resolve().parent

from baselines import run_drl_policy  # noqa: E402
from common import TOU_RULES, build_tariff_windows, load_system_config, score_month  # noqa: E402
from grepo_agent import GREPOAgent  # noqa: E402
from ppo_agent import PPOAgent  # noqa: E402
from sadrbc import SADRBCConfig  # noqa: E402
from scenario_gen import DayData, MonthData  # noqa: E402


class DispatchRunWarning(RuntimeError):
    pass


def ensure_inside_sizing_demo(path: Path) -> Path:
    resolved = path.resolve()
    base = BASE_DIR.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"path escapes Sizing_Demo: {path}")
    return resolved


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
    windows = build_tariff_windows(
        str(parameters.get("billing_windows_expensive", "")),
        str(parameters.get("billing_windows_cheap", "")),
        dt_hours,
    )
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
            **windows,
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
    if algo not in {"ppo", "grepo"}:
        raise DispatchRunWarning(f"{checkpoint_name}: unsupported checkpoint algorithm {algo}")

    if algo == "ppo":
        obs_dim = int(meta.get("obs_dim") or (17 if meta.get("obs_variant") == "fc" else 13))
        agent = PPOAgent(obs_dim=obs_dim)
    else:
        import torch

        raw = torch.load(path, map_location="cpu")
        obs_dim = int(raw.get("obs_dim") or meta.get("obs_dim") or 13)
        agent = GREPOAgent(
            obs_dim=obs_dim,
            n_group=int(meta.get("group") or meta.get("n_group") or 6),
            std=float(raw.get("std") or meta.get("std") or 0.30),
            beta=float(meta.get("beta") or 0.5),
        )
    agent.load(path)
    return agent, algo, meta


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
    cfg = build_dispatch_config(parameters, float(e_cap), float(p_rated))
    month = month or dataset_to_month(selected_data_path(parameters))
    p_ref = float(meta.get("p_ref_kw") or _policy_reference_kw(month))
    rollout = run_drl_policy(month, cfg, agent, p_ref_kw=p_ref)
    days = policy_result_to_days(month, rollout, cfg, parameters)
    return {
        "policy": checkpoint_name,
        "algo": algo,
        "meta": meta,
        "warnings": warnings,
        "days": days,
        "kpi": score_month(rollout["p_grid_days"], cfg, month.days),
    }


def run_policies(
    policy_names: list[str],
    parameters: dict[str, Any],
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> tuple[dict[str, Any], list[str]]:
    month = dataset_to_month(selected_data_path(parameters))
    results = {}
    warnings = []
    for policy_name in policy_names:
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
