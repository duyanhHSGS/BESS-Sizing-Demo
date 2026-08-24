"""Pure-Python DRL runtime shell.

This keeps the existing observation -> PPO -> feasible mapping -> safety projection
chain, but replaces Mongo repositories and APScheduler with explicit files and an
async loop.
"""
from __future__ import annotations

import asyncio
import calendar
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from _core import bootstrap_core

bootstrap_core()

from bess_drl.engine.drl_config import build_constraints, build_policy_config
from bess_drl.engine.feasible_action import map_feasible_action
from bess_drl.engine.obs_builder import (
    DrlTickState,
    build_observation_vector,
)
from bess_drl.engine.obs_builder import (
    TelemetrySnapshot as ObsSnapshot,
)
from bess_drl.engine.overlay_builder import build_overlay
from bess_drl.engine.policy_runner import PolicyRunner
from bess_drl.engine.safety_projection import project_action
from bess_drl.mqtt.subscriber import MqttSubscriber
from bess_drl.mqtt.telemetry_accumulator import TelemetryAccumulator
from bess_drl.state.persistent_state import PersistentState

logger = logging.getLogger("bess_drl_debloated")


@dataclass(frozen=True)
class RuntimeOptions:
    policy_path: Path
    config_path: Path | None = None
    mode: str = "shadow"
    timezone: str = "Asia/Ho_Chi_Minh"
    interval_minutes: int = 15
    tick_offset_seconds: int = 2
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_client_id: str = "bess-drl-debloated"
    state_path: Path = Path("state/runtime_state.json")
    log_path: Path = Path("logs/setpoints.jsonl")
    plan_path: Path | None = None
    controller_url: str = "http://localhost:8001"
    api_key: str = "dev-api-key-change-me"

    def validate(self) -> None:
        if self.mode not in {"shadow", "closed"}:
            raise ValueError("mode must be 'shadow' or 'closed'")
        if self.interval_minutes <= 0:
            raise ValueError("interval_minutes must be > 0")
        if self.tick_offset_seconds < 0:
            raise ValueError("tick_offset_seconds must be >= 0")
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"Policy not found: {self.policy_path}")
        if self.config_path is not None and not self.config_path.is_file():
            raise FileNotFoundError(f"Config not found: {self.config_path}")


class DebloatedRuntime:
    def __init__(self, options: RuntimeOptions) -> None:
        options.validate()
        self.options = options
        self.accumulator = TelemetryAccumulator()
        self.policy_runner = PolicyRunner()
        self.state = PersistentState.load_from_file(str(options.state_path))
        self.state.drl_mode = options.mode
        self._policy_meta = self.policy_runner.load_checkpoint(str(options.policy_path))
        if "p_ref_kw" not in self._policy_meta:
            raise ValueError("Checkpoint has no p_ref_kw; observation scale is unknown")

        embedded_config = self._policy_meta.get("effective_config")
        if embedded_config:
            self._config_snapshot = dict(embedded_config)
            if options.config_path is not None:
                logger.info(
                    "checkpoint contains effective_config; ignoring fallback --config %s",
                    options.config_path,
                )
        elif options.config_path is not None:
            self._config_snapshot = json.loads(
                options.config_path.read_text(encoding="utf-8")
            )
        else:
            raise ValueError(
                "Checkpoint has no effective_config; pass --config with the exact "
                "BessDrlConfig used for training"
            )
        self._config_snapshot["p_ref_kw"] = float(self._policy_meta["p_ref_kw"])
        self._mqtt = MqttSubscriber(
            host=options.mqtt_host,
            port=options.mqtt_port,
            accumulator=self.accumulator,
            client_id=options.mqtt_client_id,
            username=options.mqtt_username,
            password=options.mqtt_password,
        )

    async def run_forever(self) -> None:
        await self._mqtt.start()
        connected = await self._mqtt.wait_connected(timeout=10.0)
        if not connected:
            logger.warning("MQTT did not report connected within 10s; continuing")
        logger.info(
            "debloated runtime started mode=%s policy=%s interval=%dm",
            self.options.mode,
            self.options.policy_path.name,
            self.options.interval_minutes,
        )
        try:
            while True:
                delay = self._seconds_until_next_tick()
                logger.debug("next tick in %.2fs", delay)
                await asyncio.sleep(delay)
                try:
                    await self.execute_tick()
                except Exception:
                    logger.exception("tick failed")
        finally:
            self.state.save_to_file(str(self.options.state_path))
            await self._mqtt.stop()

    def _seconds_until_next_tick(self) -> float:
        period = self.options.interval_minutes * 60
        now = time.time()
        next_boundary = (int(now) // period + 1) * period
        target = next_boundary + self.options.tick_offset_seconds
        return max(0.0, target - now)

    async def execute_tick(self) -> dict | None:
        tz = ZoneInfo(self.options.timezone)
        now_local = datetime.now(tz)
        today = now_local.date()
        today_str = today.isoformat()
        current_slot = now_local.hour * 4 + now_local.minute // 15
        now_ts_ms = int(now_local.timestamp() * 1000)

        self.state.check_month_reset(self.options.timezone)

        if not self.accumulator.is_fresh(max_age_seconds=120.0):
            logger.warning("telemetry stale (>120s); skipping tick")
            return None

        completed = self.accumulator.completed_15min_snapshot(now_ts_ms)
        live = self.accumulator.latest_snapshot()
        if completed is None:
            coverage = self.accumulator.completed_15min_coverage(now_ts_ms)
            logger.warning(
                "completed 15-minute slot has no samples; coverage=%.1f%% samples=%d",
                coverage.coverage_fraction * 100.0,
                coverage.sample_count,
            )
            return None
        if live is None:
            logger.warning("no live telemetry; skipping tick")
            return None

        policy_config = build_policy_config(self._config_snapshot, today)
        constraints = build_constraints(self._config_snapshot)

        completed_demand = self.accumulator.completed_30min_grid_average(now_ts_ms)
        demand_prev, demand_for_peak = _billing_demand_transition(
            current_slot=current_slot,
            day_of_month=today.day,
            completed_demand_kw=completed_demand,
        )
        completed_effective_load = max(0.0, completed.p_load_kw - completed.p_pv_kw)

        if demand_for_peak is not None:
            nobess_demand = self.accumulator.completed_30min_nobess_average(now_ts_ms)
            if nobess_demand is None:
                logger.warning(
                    "grid demand block closed without matching no-BESS block; skipping tick"
                )
                return None
            self.state.update_running_peak(demand_for_peak, nobess_demand)
        self.state.update_net_load(completed_effective_load)

        tick_state = DrlTickState(
            d_run_kw=self.state.d_run_kw,
            d_run_nobess_kw=self.state.d_run_nobess_kw,
            current_slot=current_slot,
            day_index_in_month=today.day - 1,
            days_in_month=calendar.monthrange(today.year, today.month)[1],
            is_working_day=today.weekday() < 5,
            block_grid_sum_kw=(
                max(0.0, completed.p_grid_kw) if current_slot % 2 == 1 else 0.0
            ),
            demand_prev_kw=demand_prev or 0.0,
            net_load=self.state.net_load_history(),
        )
        observation = build_observation_vector(
            ObsSnapshot(
                p_grid_kw=completed.p_grid_kw,
                p_pv_kw=completed.p_pv_kw,
                p_load_kw=completed.p_load_kw,
                soc_fraction=live.soc_fraction,
                p_bess_kw=completed.p_bess_kw,
            ),
            policy_config,
            tick_state,
        )

        history_ready = tick_state.net_load.is_ready
        action_raw = self.policy_runner.infer(observation) if history_ready else 0.0

        mapped = map_feasible_action(
            action_raw=action_raw,
            soc_fraction=live.soc_fraction,
            load_kw=live.p_load_kw,
            pv_kw=live.p_pv_kw,
            p_rated_kw=constraints.p_rated_kw,
            e_cap_kwh=constraints.e_cap_kwh,
            soc_min=constraints.soc_min,
            soc_max=constraints.soc_max,
            eta_charge=constraints.eta_charge,
            eta_discharge=constraints.eta_discharge,
            dt_hours=0.25,
            allow_export=constraints.allow_export,
        )
        projection = project_action(
            desired_power_kw=mapped.mapped_power_kw,
            soc_fraction=live.soc_fraction,
            load_kw=live.p_load_kw,
            pv_kw=live.p_pv_kw,
            constraints=constraints,
            dt_hours=0.25,
        )
        p_drl_kw = projection.p_bess_kw
        p_requested_kw = action_raw * constraints.p_rated_kw
        tariff_zone = _tariff_zone_for_slot(current_slot, policy_config)

        base_plan = self._load_base_plan(today_str)
        base_p_plan_at_slot = 0.0
        if base_plan is not None:
            p_plan = base_plan["pPlan"]
            if current_slot < len(p_plan):
                base_p_plan_at_slot = float(p_plan[current_slot])

        record = {
            "ts": now_local.isoformat(),
            "date": today_str,
            "step": current_slot,
            "mode": self.options.mode,
            "policy_path": str(self.options.policy_path),
            "p_drl_kw": round(p_drl_kw, 3),
            "action_raw": round(action_raw, 6),
            "p_requested_kw": round(p_requested_kw, 3),
            "p_executed_kw": round(p_drl_kw, 3),
            "clip_reason": mapped.clip_reason,
            "block_phase": current_slot % 2,
            "action_held": not history_ready,
            "soc_actual": round(live.soc_fraction, 4),
            "p_load_kw": round(completed.p_load_kw, 2),
            "p_pv_kw": round(completed.p_pv_kw, 2),
            "p_grid_kw": round(completed.p_grid_kw, 2),
            "d_run_kw": round(self.state.d_run_kw, 2),
            "tariff_zone": tariff_zone,
            "p_plan_base_kw": round(base_p_plan_at_slot, 3),
            "p_ref_kw": float(self._policy_meta["p_ref_kw"]),
            "timezone": self.options.timezone,
        }
        _append_jsonl(self.options.log_path, record)

        if self.options.mode == "closed":
            overlay = self._build_overlay(
                base_plan=base_plan,
                today_str=today_str,
                current_slot=current_slot,
                p_drl_kw=p_drl_kw,
            )
            await asyncio.to_thread(self._push_plan, overlay)

        self.state.g_prev_kw = completed.p_grid_kw
        self.state.save_to_file(str(self.options.state_path))

        logger.info(
            "slot=%d zone=%s mode=%s action=%.3f req=%.1f exec=%.1f clip=%s "
            "held=%s soc=%.3f d_run=%.1f",
            current_slot,
            tariff_zone,
            self.options.mode,
            action_raw,
            p_requested_kw,
            p_drl_kw,
            mapped.clip_reason,
            not history_ready,
            live.soc_fraction,
            self.state.d_run_kw,
        )
        return record

    def _load_base_plan(self, today_str: str) -> dict | None:
        path = self.options.plan_path
        if path is None or not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        plan_date = str(raw.get("date", today_str))
        if plan_date != today_str:
            logger.warning("plan file is for %s, not %s; using standalone DRL plan", plan_date, today_str)
            return None
        p_plan = list(raw.get("pPlan", raw.get("p_plan", [])))
        dispatch = list(raw.get("dispatchSources", raw.get("dispatch_sources", [])))
        if len(dispatch) != 96:
            raise ValueError(
                f"plan file must contain 96 dispatchSources entries; got {len(dispatch)}"
            )
        return {
            "date": plan_date,
            "pPlan": p_plan,
            "dispatchSources": dispatch,
            "socPlan": list(raw.get("socPlan", raw.get("soc_plan", []))),
            "socFloor": list(raw.get("socFloor", raw.get("soc_floor", []))),
        }

    @staticmethod
    def _build_overlay(
        *,
        base_plan: dict | None,
        today_str: str,
        current_slot: int,
        p_drl_kw: float,
    ):
        if base_plan is None:
            return build_overlay(
                base_p_plan=[0.0] * 96,
                base_dispatch_sources=["standby"] * 96,
                base_soc_plan=[],
                base_soc_floor=[],
                plan_date=today_str,
                slot=current_slot,
                p_drl_kw=p_drl_kw,
            )
        return build_overlay(
            base_p_plan=base_plan["pPlan"],
            base_dispatch_sources=base_plan["dispatchSources"],
            base_soc_plan=base_plan["socPlan"],
            base_soc_floor=base_plan["socFloor"],
            plan_date=today_str,
            slot=current_slot,
            p_drl_kw=p_drl_kw,
        )

    def _push_plan(self, overlay) -> None:
        payload = {
            "date": overlay.date,
            "pPlan": overlay.p_plan,
            "dispatchSources": overlay.dispatch_sources,
        }
        if overlay.soc_plan:
            payload["socPlan"] = overlay.soc_plan
        if overlay.soc_floor:
            payload["socFloor"] = overlay.soc_floor

        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.options.controller_url.rstrip('/')}/api/plan",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self.options.api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10.0) as response:
                status = response.status
                if status >= 400:
                    raise RuntimeError(f"controller returned HTTP {status}")
        except urllib.error.HTTPError as exc:
            detail = exc.read(200).decode("utf-8", errors="replace")
            raise RuntimeError(f"controller HTTP {exc.code}: {detail}") from exc
        logger.info("plan pushed to %s", self.options.controller_url)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
        handle.write("\n")


def _tariff_zone_for_slot(slot: int, config) -> str:
    if slot in config.tariff_step_indices_peak:
        return "PEAK"
    if slot in config.tariff_step_indices_off:
        return "OFF"
    return "INTER"


def _billing_demand_transition(
    *, current_slot: int, day_of_month: int, completed_demand_kw: float | None
) -> tuple[float | None, float | None]:
    if current_slot == 0:
        return None, None if day_of_month == 1 else completed_demand_kw
    if current_slot == 1:
        return None, None
    if current_slot % 2 == 0:
        return completed_demand_kw, completed_demand_kw
    return completed_demand_kw, None
