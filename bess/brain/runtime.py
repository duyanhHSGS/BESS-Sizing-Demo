"""Shared measured-data runtime for Dispatch, Benchmark, Live, and Shadow."""
from __future__ import annotations

import calendar
import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import torch

from bess.brain.brain1_agent import Brain1Agent
from bess.brain.brain2_agent import Brain2Agent
from bess.brain.brain3_agent import Brain3Agent
from bess.brain.brain_env import BrainEnv, BrainEnvironmentStepResult, BrainEpisode, BrainTimestepInput
from bess.core.config import BrainConfig


class BrainRuntimeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BrainDay:
    day_index: int
    date_iso: str | None
    day_type: str
    load_kw: tuple[float, ...]
    pv_kw: tuple[float, ...]

    @property
    def net_load_kw(self) -> tuple[float, ...]:
        return tuple(load - pv for load, pv in zip(self.load_kw, self.pv_kw))


@dataclass(frozen=True, slots=True)
class BrainPeriod:
    key: str
    days: tuple[BrainDay, ...]


def load_csv_days(path: Path) -> list[BrainDay]:
    buckets: dict[int, dict[str, Any]] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                day_index = int(row["day_index"])
                step_index = int(row["step"])
                load_kw = float(row["P_load_kW"])
                pv_kw = float(row.get("P_pv_kW") or 0.0)
            except (KeyError, TypeError, ValueError) as exc:
                raise BrainRuntimeError(f"invalid measured-data row: {exc}") from exc
            if not all(math.isfinite(value) for value in (load_kw, pv_kw)):
                raise BrainRuntimeError("measured load and PV values must be finite")
            bucket = buckets.setdefault(
                day_index,
                {
                    "date_iso": row.get("date_iso") or None,
                    "day_type": row.get("day_type") or "working",
                    "points": {},
                },
            )
            if step_index in bucket["points"]:
                raise BrainRuntimeError(f"duplicate day {day_index} step {step_index}")
            bucket["points"][step_index] = (load_kw, pv_kw)

    if not buckets:
        raise BrainRuntimeError("selected CSV contains no measured rows")
    days: list[BrainDay] = []
    expected_steps: int | None = None
    for day_index in sorted(buckets):
        bucket = buckets[day_index]
        indexes = sorted(bucket["points"])
        if indexes != list(range(len(indexes))):
            raise BrainRuntimeError(f"day {day_index} steps must be contiguous from zero")
        if expected_steps is None:
            expected_steps = len(indexes)
        if len(indexes) != expected_steps:
            raise BrainRuntimeError("every measured day must have the same native resolution")
        load, pv = zip(*(bucket["points"][index] for index in indexes))
        days.append(
            BrainDay(
                day_index=day_index,
                date_iso=bucket["date_iso"],
                day_type=bucket["day_type"],
                load_kw=tuple(load),
                pv_kw=tuple(pv),
            )
        )
    return days


def split_billing_periods(
    days: list[BrainDay],
    *,
    reject_leftover: bool,
    warnings: list[str] | None = None,
) -> list[BrainPeriod]:
    notices = warnings if warnings is not None else []
    dated: list[tuple[date, BrainDay]] = []
    for day in days:
        if not day.date_iso:
            dated = []
            break
        try:
            dated.append((date.fromisoformat(day.date_iso), day))
        except ValueError:
            dated = []
            break
    if dated:
        groups: dict[str, list[tuple[date, BrainDay]]] = {}
        for stamp, day in dated:
            groups.setdefault(stamp.strftime("%Y-%m"), []).append((stamp, day))
        complete = True
        for entries in groups.values():
            stamps = sorted(stamp for stamp, _ in entries)
            expected = calendar.monthrange(stamps[0].year, stamps[0].month)[1]
            complete = complete and len(stamps) == expected and stamps[0].day == 1 and stamps[-1].day == expected
        if complete:
            return [
                BrainPeriod(key, tuple(day for _, day in sorted(entries)))
                for key, entries in sorted(groups.items())
            ]
        incomplete = []
        for key, entries in sorted(groups.items()):
            stamps = sorted(stamp for stamp, _ in entries)
            expected = calendar.monthrange(stamps[0].year, stamps[0].month)[1]
            if len(stamps) != expected or stamps[0].day != 1 or stamps[-1].day != expected:
                incomplete.append(key)
        notices.append(
            "Incomplete calendar month(s) "
            + ", ".join(incomplete)
            + "; using sequential 30-day billing periods instead."
        )
        days = [day for _, day in sorted(dated)]

    full_count = len(days) // 30
    leftover = len(days) % 30
    if leftover and reject_leftover and full_count:
        notices.append(
            f"Skipped {leftover} trailing day(s) that could not form a complete 30-day billing period."
        )
    periods = [
        BrainPeriod(f"period-{index + 1:03d}", tuple(days[index * 30:(index + 1) * 30]))
        for index in range(full_count)
    ]
    if leftover and not reject_leftover:
        periods.append(BrainPeriod(f"period-{full_count + 1:03d}", tuple(days[full_count * 30:])))
    if not periods:
        if reject_leftover:
            raise BrainRuntimeError("dataset needs at least one complete billing period")
        periods.append(BrainPeriod("period-001", tuple(days)))
    return periods


def _clock_minutes(raw: str) -> int:
    pieces = raw.strip().split(":", 1)
    if len(pieces) != 2:
        raise BrainRuntimeError(f"invalid tariff time {raw!r}")
    hour, minute = (int(piece) for piece in pieces)
    if hour < 0 or hour > 24 or minute < 0 or minute > 59 or (hour == 24 and minute):
        raise BrainRuntimeError(f"invalid tariff time {raw!r}")
    return hour * 60 + minute


def parse_windows(raw: str) -> tuple[tuple[int, int], ...]:
    windows = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        pieces = item.split("-", 1)
        if len(pieces) != 2:
            raise BrainRuntimeError(f"invalid tariff window {item!r}")
        start, end = _clock_minutes(pieces[0]), _clock_minutes(pieces[1])
        if start == end:
            raise BrainRuntimeError("tariff windows cannot have zero length")
        windows.append((start, end))
    return tuple(windows)


def _inside(minute: float, windows: tuple[tuple[int, int], ...]) -> bool:
    return any(
        start <= minute < end if start < end else minute >= start or minute < end
        for start, end in windows
    )


def tariffs_for_day(day: BrainDay, config: BrainConfig) -> tuple[float, ...]:
    cheap = parse_windows(config.cheap_windows)
    expensive = parse_windows(config.expensive_windows)
    sunday = False
    if day.date_iso:
        try:
            sunday = date.fromisoformat(day.date_iso).weekday() == 6
        except ValueError:
            sunday = False
    values = []
    for index in range(len(day.load_kw)):
        minute = index * config.timestep_hours * 60.0
        if _inside(minute, cheap):
            values.append(config.cheap_tariff_vnd_per_kwh)
        elif not (config.sunday_no_peak and sunday) and _inside(minute, expensive):
            values.append(config.expensive_tariff_vnd_per_kwh)
        else:
            values.append(config.normal_tariff_vnd_per_kwh)
    return tuple(values)


def episode_for_period(period: BrainPeriod, config: BrainConfig) -> tuple[BrainEpisode, list[dict[str, Any]]]:
    steps_per_day = len(period.days[0].load_kw)
    expected = 1.0 / config.timestep_hours
    if not math.isclose(steps_per_day / 24.0, expected, rel_tol=0.0, abs_tol=1e-9):
        raise BrainRuntimeError("configured timestep does not match selected CSV")
    timesteps = []
    labels = []
    for day in period.days:
        if len(day.load_kw) != steps_per_day:
            raise BrainRuntimeError("period contains mixed native resolutions")
        tariffs = tariffs_for_day(day, config)
        is_working = day.day_type.lower() not in {"weekend", "holiday"}
        for step, (load, pv, net, tariff) in enumerate(
            zip(day.load_kw, day.pv_kw, day.net_load_kw, tariffs)
        ):
            timesteps.append(BrainTimestepInput(net, tariff, is_working))
            labels.append(
                {
                    "billing_period": period.key,
                    "day_index": day.day_index,
                    "date_iso": day.date_iso,
                    "day_type": day.day_type,
                    "step": step,
                    "time": f"{int(step * config.timestep_hours):02d}:{int((step * config.timestep_hours % 1) * 60):02d}",
                    "load_kw": load,
                    "pv_kw": pv,
                    "net_load_kw": net,
                    "tariff_vnd_per_kwh": tariff,
                }
            )
    scale = max(1.0, max(abs(step.net_load_kw) for step in timesteps), config.battery_power_kw)
    episode = BrainEpisode(
        tuple(timesteps),
        steps_per_day,
        power_scale_kw=scale,
        tariff_normalization_vnd_per_kwh=config.expensive_tariff_vnd_per_kwh,
    )
    return episode, labels


def make_env(episode: BrainEpisode, config: BrainConfig) -> BrainEnv:
    return BrainEnv(
        initial_state_of_charge=config.initial_soc,
        minimum_state_of_charge=config.minimum_soc,
        maximum_state_of_charge=config.maximum_soc,
        battery_capacity_kwh=config.battery_capacity_kwh,
        battery_power_kw=config.battery_power_kw,
        timestep_hours=config.timestep_hours,
        charge_efficiency=config.charge_efficiency,
        discharge_efficiency=config.discharge_efficiency,
        demand_charge_vnd_per_kw=config.demand_charge_vnd_per_kw,
        battery_wear_vnd_per_kwh=config.battery_wear_vnd_per_kwh,
        episode=episode,
        required_final_soc=config.required_final_soc,
    )


def _single_window(raw: str, label: str) -> tuple[int, int]:
    windows = parse_windows(raw)
    if len(windows) != 1 or windows[0][0] >= windows[0][1]:
        raise BrainRuntimeError(f"Brain 2 requires one non-wrapping {label} window")
    return windows[0]


def make_brain1(config: BrainConfig) -> Brain1Agent:
    denominator = config.expensive_tariff_vnd_per_kwh
    return Brain1Agent(
        config.cheap_tariff_vnd_per_kwh / denominator,
        config.expensive_tariff_vnd_per_kwh / denominator,
    )


def make_brain2(config: BrainConfig) -> Brain2Agent:
    cheap_start, cheap_end = _single_window(config.cheap_windows, "cheap")
    expensive_start, expensive_end = _single_window(config.expensive_windows, "expensive")
    return Brain2Agent(
        battery_capacity_kwh=config.battery_capacity_kwh,
        battery_power_kw=config.battery_power_kw,
        minimum_state_of_charge=config.minimum_soc,
        maximum_state_of_charge=config.maximum_soc,
        timestep_minutes=config.timestep_hours * 60.0,
        cheap_tariff_vnd_per_kwh=config.cheap_tariff_vnd_per_kwh,
        normal_tariff_vnd_per_kwh=config.normal_tariff_vnd_per_kwh,
        expensive_tariff_vnd_per_kwh=config.expensive_tariff_vnd_per_kwh,
        cheap_start_minute=cheap_start,
        cheap_end_minute=cheap_end,
        expensive_start_minute=expensive_start,
        expensive_end_minute=expensive_end,
    )


def load_brain3_checkpoint(path: Path, config: BrainConfig) -> tuple[Brain3Agent, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise BrainRuntimeError(f"{Path(path).name}: incompatible Brain 3 checkpoint schema")
    if payload.get("algorithm") != "brain3_dqn" or payload.get("observation_dim") != 7:
        raise BrainRuntimeError(f"{Path(path).name}: not a seven-eye Brain 3 deployment checkpoint")
    if tuple(payload.get("action_values", ())) != (-1.0, 0.0, 1.0):
        raise BrainRuntimeError(f"{Path(path).name}: incompatible Brain 3 action contract")
    meta = dict(payload.get("meta") or {})
    agent = Brain3Agent(hidden_dim=int(meta.get("hidden_dim", 128)), device="cpu")
    agent.online_network.load_state_dict(payload["online_network"])
    agent.target_network.load_state_dict(payload["online_network"])
    trained_fingerprint = meta.get("environment_fingerprint")
    if trained_fingerprint != config.fingerprint():
        raise BrainRuntimeError(f"{Path(path).name}: environment fingerprint is incompatible")
    return agent, meta


def controller_factory(controller_id: str, config: BrainConfig, checkpoint_dir: Path) -> tuple[Any, int, dict[str, Any]]:
    if controller_id == "brain1":
        return make_brain1(config), 1, {"display_name": "Brain 1", "kind": "rule"}
    if controller_id == "brain2":
        return make_brain2(config), 1, {"display_name": "Brain 2", "kind": "schedule"}
    if controller_id.startswith("brain3:"):
        name = Path(controller_id.split(":", 1)[1]).name
        if name != controller_id.split(":", 1)[1]:
            raise BrainRuntimeError("Brain 3 checkpoint name must be a basename")
        path = Path(checkpoint_dir).resolve() / name
        if path.parent != Path(checkpoint_dir).resolve() or not path.is_file():
            raise BrainRuntimeError(f"unknown Brain 3 checkpoint: {name}")
        agent, meta = load_brain3_checkpoint(path, config)
        native_minutes = config.timestep_hours * 60.0
        control_minutes = float(meta.get("control_dt_minutes", native_minutes))
        ratio = control_minutes / native_minutes
        held_steps = round(ratio)
        if held_steps <= 0 or not math.isclose(ratio, held_steps, abs_tol=1e-9):
            raise BrainRuntimeError(f"{name}: control interval is incompatible with selected data")
        return agent, held_steps, {**meta, "display_name": f"Brain 3 - {name}", "kind": "dqn"}
    raise BrainRuntimeError(f"unknown controller: {controller_id}")


def _trace_entry(label: dict[str, Any], observation: tuple[float, ...], result: BrainEnvironmentStepResult) -> dict[str, Any]:
    physics = result.bess.physics
    return {
        **label,
        "observation": list(observation),
        "requested_action": result.requested_action,
        "projected_action": result.projected_action,
        "horizon_adjusted": result.horizon_adjusted,
        "requested_battery_kw": result.requested_battery_kw,
        "projected_battery_kw": result.projected_battery_kw,
        "executed_battery_kw": physics.final_battery_kw,
        "battery_to_factory_kw": physics.battery_to_factory_kw,
        "grid_to_battery_kw": physics.grid_to_battery_kw,
        "grid_import_kw": physics.grid_import_kw,
        "raw_grid_import_kw": result.raw.grid_import_kw,
        "soc_before": physics.starting_soc,
        "soc_after": physics.next_soc,
        "meter_peak_kw": result.bess.meter.monthly_peak_kw,
        "raw_meter_peak_kw": result.raw.meter.monthly_peak_kw,
        "energy_cost_vnd": result.bess.cost.electricity_energy_cost_vnd,
        "demand_cost_vnd": result.bess.cost.demand_cost_vnd,
        "wear_cost_vnd": result.bess.cost.battery_wear_cost_vnd,
        "reward_vnd": result.reward.timestep_savings_vnd,
        "cumulative_savings_vnd": result.reward.monthly_savings_vnd,
        "done": result.done,
    }


def run_controller(
    controller_id: str,
    periods: list[BrainPeriod],
    config: BrainConfig,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    controller, held_steps, meta = controller_factory(controller_id, config, checkpoint_dir)
    trace: list[dict[str, Any]] = []
    totals = {
        "raw_cost_vnd": 0.0,
        "bess_cost_vnd": 0.0,
        "energy_cost_vnd": 0.0,
        "demand_cost_vnd": 0.0,
        "wear_cost_vnd": 0.0,
        "throughput_kwh": 0.0,
        "action_adjustments": 0,
    }
    ending_soc = config.required_final_soc
    for period in periods:
        episode, labels = episode_for_period(period, config)
        env = make_env(episode, config)
        observation = env.reset()
        action = 0.0
        for index, label in enumerate(labels):
            if index % held_steps == 0:
                action = float(controller.act(observation))
            result = env.step(action)
            if not isinstance(result, BrainEnvironmentStepResult):
                raise RuntimeError("Brain runtime requires owned-episode mode")
            trace.append(_trace_entry(label, observation, result))
            totals["throughput_kwh"] += result.bess.physics.battery_throughput_kwh
            totals["action_adjustments"] += int(
                result.horizon_adjusted
                or not math.isclose(
                    result.bess.physics.requested_battery_kw,
                    result.bess.physics.final_battery_kw,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            if result.next_observation is not None:
                observation = result.next_observation
        ending_soc = env.bess_world.state_of_charge
        totals["raw_cost_vnd"] += env.raw_world.total_operating_cost_vnd
        totals["bess_cost_vnd"] += env.bess_world.total_operating_cost_vnd
        totals["energy_cost_vnd"] += env.bess_world.total_electricity_energy_cost_vnd
        totals["demand_cost_vnd"] += env.bess_world.total_demand_cost_vnd
        totals["wear_cost_vnd"] += env.bess_world.total_battery_wear_cost_vnd
    savings = totals["raw_cost_vnd"] - totals["bess_cost_vnd"]
    kpi = {
        **{key: round(value, 6) for key, value in totals.items()},
        "savings_vnd": round(savings, 6),
        "savings_pct": round(100.0 * savings / totals["raw_cost_vnd"], 6) if totals["raw_cost_vnd"] else 0.0,
        "raw_peak_kw": round(max((row["raw_meter_peak_kw"] for row in trace), default=0.0), 6),
        "bess_peak_kw": round(max((row["meter_peak_kw"] for row in trace), default=0.0), 6),
        "ending_soc": round(ending_soc, 12),
        "ending_soc_compliant": math.isclose(ending_soc, config.required_final_soc, abs_tol=1e-9),
    }
    return {"controller": controller_id, "meta": meta, "trace": trace, "kpi": kpi}


def run_controllers(
    controller_ids: list[str],
    csv_path: Path,
    parameters: dict[str, Any],
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    config = BrainConfig.from_parameters(parameters)
    warnings: list[str] = []
    periods = split_billing_periods(
        load_csv_days(csv_path), reject_leftover=True, warnings=warnings
    )
    results: dict[str, Any] = {}
    for controller_id in dict.fromkeys(controller_ids):
        try:
            results[controller_id] = run_controller(controller_id, periods, config, checkpoint_dir)
        except (BrainRuntimeError, ValueError, RuntimeError, KeyError) as exc:
            warnings.append(f"{controller_id}: {exc}")
    return results, warnings
