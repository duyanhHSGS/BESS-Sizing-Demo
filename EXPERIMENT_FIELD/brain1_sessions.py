"""Independent, temporary web sessions where Brain 1 is the only driver."""
from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from bess.core.timebase import steps_per_day_from_dt
from bess.dispatch.dispatch_runner import dataset_to_month
from bess.evaluation.benchmark import detect_dt_hours, selected_data_filename, selected_data_path
from EXPERIMENT_FIELD.brain1_agent import Brain1Agent, Brain1Decision
from EXPERIMENT_FIELD.brain_env import (
    BrainEnvironmentStepResult,
    BrainEnv,
    BrainEpisode,
    BrainObservation,
    BrainTimestepInput,
    ElectricityMeterState,
)
from EXPERIMENT_FIELD.brain_env_sessions import (
    BrainEnvSessionComplete,
    BrainEnvSessionError,
    BrainEnvSessionNotFound,
    _clock_label,
    _finite_float,
    _settings_snapshot,
    _tariffs_for_day,
)


EYE_NAMES = (
    "time_sin",
    "time_cos",
    "normalized_net_load",
    "normalized_soc",
    "normalized_tariff",
    "normalized_monthly_peak",
    "working_day",
)


class Brain1SessionError(BrainEnvSessionError):
    """A Brain 1 spectator request cannot be completed."""


def _observation_payload(observation: BrainObservation) -> dict[str, float]:
    return {name: float(value) for name, value in zip(EYE_NAMES, observation)}


def _decision_payload(decision: Brain1Decision) -> dict[str, Any]:
    return asdict(decision)


@dataclass(slots=True)
class Brain1PlaygroundSession:
    session_id: str
    dataset_name: str
    day_index: int
    day_type: str
    date_iso: str | None
    timestep_hours: float
    load_kw: tuple[float, ...]
    pv_kw: tuple[float, ...]
    net_load_kw: tuple[float, ...]
    tariffs_vnd_per_kwh: tuple[float, ...]
    starting_peak_kw: float
    tariff_normalization_vnd_per_kwh: float
    settings: dict[str, Any]
    env: BrainEnv
    agent: Brain1Agent
    trace: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def step_index(self) -> int:
        return self.env.bess_world.timestep_index

    @property
    def total_steps(self) -> int:
        return len(self.net_load_kw)

    @property
    def complete(self) -> bool:
        return self.step_index >= self.total_steps

    def _preview_unlocked(self) -> dict[str, Any] | None:
        if self.complete:
            return None
        observation = self.env.current_observation()
        decision = self.agent.decide(observation)
        index = self.step_index
        return {
            "step_index": index,
            "time": _clock_label(index, self.timestep_hours),
            "load_kw": self.load_kw[index],
            "pv_kw": self.pv_kw[index],
            "net_load_kw": self.net_load_kw[index],
            "tariff_vnd_per_kwh": self.tariffs_vnd_per_kwh[index],
            "observation": _observation_payload(observation),
            "decision": _decision_payload(decision),
        }

    def preview(self) -> dict[str, Any] | None:
        with self.lock:
            return self._preview_unlocked()

    def _summary_unlocked(self) -> dict[str, Any]:
        latest_reward = self.trace[-1]["reward"] if self.trace else {
            "timestep_savings_vnd": 0.0,
            "monthly_savings_vnd": 0.0,
        }
        return {
            "state_of_charge": self.env.bess_world.state_of_charge,
            "bess_monthly_peak_kw": self.env.bess_world.meter_state.monthly_peak_kw,
            "raw_monthly_peak_kw": self.env.raw_world.meter_state.monthly_peak_kw,
            "bess_operating_cost_vnd": self.env.bess_world.total_operating_cost_vnd,
            "raw_operating_cost_vnd": self.env.raw_world.total_operating_cost_vnd,
            "energy_savings_vnd": self.env.electricity_energy_savings_vnd,
            "demand_savings_vnd": self.env.demand_savings_vnd,
            "battery_wear_cost_vnd": self.env.battery_wear_cost_vnd,
            "net_battery_savings_vnd": self.env.net_battery_savings_vnd,
            "latest_timestep_reward_vnd": latest_reward["timestep_savings_vnd"],
            "monthly_reward_vnd": latest_reward["monthly_savings_vnd"],
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.session_id,
                "controller": "brain1",
                "dataset_name": self.dataset_name,
                "day_index": self.day_index,
                "day_type": self.day_type,
                "date_iso": self.date_iso,
                "timestep_hours": self.timestep_hours,
                "step_index": self.step_index,
                "total_steps": self.total_steps,
                "complete": self.complete,
                "starting_peak_kw": self.starting_peak_kw,
                "tariff_normalization_vnd_per_kwh": self.tariff_normalization_vnd_per_kwh,
                "settings": dict(self.settings),
                "thresholds": {
                    "cheap_vnd_per_kwh": self.settings["cheap_tariff_vnd_per_kwh"],
                    "expensive_vnd_per_kwh": self.settings["expensive_tariff_vnd_per_kwh"],
                    "cheap_normalized": self.agent.cheap_tariff_max_normalized,
                    "expensive_normalized": self.agent.expensive_tariff_min_normalized,
                },
                "summary": self._summary_unlocked(),
                "preview": self._preview_unlocked(),
            }

    def detail(self) -> dict[str, Any]:
        with self.lock:
            return {"status": self.status(), "trace": list(self.trace)}

    def step(self) -> tuple[dict[str, Any], dict[str, Any]]:
        with self.lock:
            preview = self._preview_unlocked()
            if preview is None:
                raise BrainEnvSessionComplete("The selected Brain 1 day is already complete.")

            decision = self.agent.decide(self.env.current_observation())
            result = self.env.step(decision.action)
            if not isinstance(result, BrainEnvironmentStepResult):
                raise RuntimeError("Brain 1 must run through BrainEnv owned-episode mode")

            physics = result.bess.physics
            bess_meter = result.bess.meter
            raw_meter = result.raw.meter
            entry = {
                **preview,
                "executed_action": decision.action,
                "requested_battery_kw": physics.requested_battery_kw,
                "battery_after_police_kw": physics.battery_after_police_kw,
                "final_battery_kw": physics.final_battery_kw,
                "battery_to_factory_kw": physics.battery_to_factory_kw,
                "grid_to_battery_kw": physics.grid_to_battery_kw,
                "conversion_loss_kw": physics.conversion_loss_kw,
                "battery_throughput_kwh": physics.battery_throughput_kwh,
                "starting_soc": physics.starting_soc,
                "next_soc": physics.next_soc,
                "bess_grid_import_kw": physics.grid_import_kw,
                "raw_grid_import_kw": result.raw.grid_import_kw,
                "bess_block_completed": bess_meter.block_completed,
                "bess_completed_block_demand_kw": bess_meter.completed_block_demand_kw,
                "bess_monthly_peak_kw": bess_meter.monthly_peak_kw,
                "raw_block_completed": raw_meter.block_completed,
                "raw_completed_block_demand_kw": raw_meter.completed_block_demand_kw,
                "raw_monthly_peak_kw": raw_meter.monthly_peak_kw,
                "energy_savings_vnd": result.electricity_energy_savings_vnd,
                "demand_savings_vnd": result.demand_savings_vnd,
                "battery_wear_cost_vnd": result.battery_wear_cost_vnd,
                "net_battery_savings_vnd": result.net_battery_savings_vnd,
                "reward": {
                    "timestep_savings_vnd": result.reward.timestep_savings_vnd,
                    "monthly_savings_vnd": result.reward.monthly_savings_vnd,
                },
                "done": result.done,
                "next_observation": (
                    None
                    if result.next_observation is None
                    else _observation_payload(result.next_observation)
                ),
            }
            self.trace.append(entry)
            return entry, self.status()


_SESSIONS: dict[str, Brain1PlaygroundSession] = {}
_SESSIONS_LOCK = threading.RLock()


def _thresholds(settings: dict[str, Any]) -> tuple[float, float, float]:
    denominator = max(
        float(settings["cheap_tariff_vnd_per_kwh"]),
        float(settings["normal_tariff_vnd_per_kwh"]),
        float(settings["expensive_tariff_vnd_per_kwh"]),
    )
    if denominator <= 0.0:
        raise Brain1SessionError("Brain 1 needs at least one positive applied energy tariff.")
    cheap = float(settings["cheap_tariff_vnd_per_kwh"]) / denominator
    expensive = float(settings["expensive_tariff_vnd_per_kwh"]) / denominator
    if cheap >= expensive:
        raise Brain1SessionError(
            "Brain 1 requires the applied cheap tariff to be lower than the expensive tariff."
        )
    return cheap, expensive, denominator


def context(parameters: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(parameters)
    path = selected_data_path(snapshot)
    timestep_hours = detect_dt_hours(path)
    month = dataset_to_month(path)
    settings = _settings_snapshot(snapshot, timestep_hours)
    cheap, expensive, denominator = _thresholds(settings)
    return {
        "dataset_name": selected_data_filename(snapshot),
        "timestep_hours": timestep_hours,
        "days": [
            {
                "day_index": day.day_index,
                "day_type": day.day_type,
                "date_iso": day.date_iso,
                "step_count": min(len(day.load), len(day.pv)),
            }
            for day in month.days
        ],
        "settings": settings,
        "thresholds": {
            "cheap_vnd_per_kwh": settings["cheap_tariff_vnd_per_kwh"],
            "expensive_vnd_per_kwh": settings["expensive_tariff_vnd_per_kwh"],
            "cheap_normalized": cheap,
            "expensive_normalized": expensive,
            "normalization_vnd_per_kwh": denominator,
        },
    }


def create_session(
    parameters: dict[str, Any], *, day_index: Any, starting_peak_kw: Any = 0.0
) -> Brain1PlaygroundSession:
    snapshot = dict(parameters)
    try:
        requested_day = int(day_index)
    except (TypeError, ValueError) as exc:
        raise Brain1SessionError("Choose a valid CSV day.") from exc
    carry_in_peak = _finite_float(starting_peak_kw, "Carry-in monthly peak", minimum=0.0)

    path = selected_data_path(snapshot)
    timestep_hours = detect_dt_hours(path)
    steps_per_day = steps_per_day_from_dt(timestep_hours)
    month = dataset_to_month(path)
    day = next((candidate for candidate in month.days if candidate.day_index == requested_day), None)
    if day is None:
        raise Brain1SessionError("The selected CSV day does not exist.")
    if len(day.load) != len(day.pv) or len(day.load) != steps_per_day:
        raise Brain1SessionError(
            f"Day {requested_day} must contain exactly {steps_per_day} aligned load/PV samples."
        )

    settings = _settings_snapshot(snapshot, timestep_hours)
    cheap_threshold, expensive_threshold, denominator = _thresholds(settings)
    load_kw = tuple(float(value) for value in day.load)
    pv_kw = tuple(float(value) for value in day.pv)
    net_load_kw = tuple(load - pv for load, pv in zip(load_kw, pv_kw))
    tariffs = _tariffs_for_day(
        step_count=steps_per_day,
        timestep_hours=timestep_hours,
        date_iso=day.date_iso,
        parameters=snapshot,
    )
    is_working_day = str(day.day_type).strip().lower() == "working"
    episode = BrainEpisode(
        timesteps=tuple(
            BrainTimestepInput(
                net_load_kw=net_load,
                tariff_vnd_per_kwh=tariff,
                is_working_day=is_working_day,
            )
            for net_load, tariff in zip(net_load_kw, tariffs)
        ),
        steps_per_day=steps_per_day,
        tariff_normalization_vnd_per_kwh=denominator,
    )
    env = BrainEnv(
        initial_state_of_charge=settings["initial_soc"],
        minimum_state_of_charge=settings["minimum_soc"],
        maximum_state_of_charge=settings["maximum_soc"],
        battery_capacity_kwh=settings["battery_capacity_kwh"],
        battery_power_kw=settings["battery_power_kw"],
        timestep_hours=timestep_hours,
        charge_efficiency=settings["charge_efficiency"],
        discharge_efficiency=settings["discharge_efficiency"],
        demand_charge_vnd_per_kw=settings["demand_charge_vnd_per_kw"],
        battery_wear_vnd_per_kwh=settings["battery_wear_vnd_per_kwh"],
        episode=episode,
    )
    env.reset()
    starting_meter = ElectricityMeterState(monthly_peak_kw=carry_in_peak)
    env.bess_world.meter_state = starting_meter
    env.raw_world.meter_state = starting_meter

    session = Brain1PlaygroundSession(
        session_id=str(uuid.uuid4()),
        dataset_name=selected_data_filename(snapshot),
        day_index=requested_day,
        day_type=day.day_type,
        date_iso=day.date_iso,
        timestep_hours=timestep_hours,
        load_kw=load_kw,
        pv_kw=pv_kw,
        net_load_kw=net_load_kw,
        tariffs_vnd_per_kwh=tariffs,
        starting_peak_kw=carry_in_peak,
        tariff_normalization_vnd_per_kwh=denominator,
        settings=settings,
        env=env,
        agent=Brain1Agent(
            cheap_tariff_max_normalized=cheap_threshold,
            expensive_tariff_min_normalized=expensive_threshold,
        ),
    )
    with _SESSIONS_LOCK:
        _SESSIONS[session.session_id] = session
    return session


def get_session(session_id: str) -> Brain1PlaygroundSession:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(str(session_id))
    if session is None:
        raise BrainEnvSessionNotFound("Brain 1 session not found.")
    return session


def drop_session(session_id: str) -> bool:
    with _SESSIONS_LOCK:
        return _SESSIONS.pop(str(session_id), None) is not None
