"""In-memory synchronized day reveal for Brain controller comparisons."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from bess.brain.runtime import run_controllers
from bess.evaluation.benchmark import selected_data_path
from bess.training.brain3_checkpoints import CHECKPOINT_DIR

_SESSIONS: dict[str, "LiveSession"] = {}
_LOCK = threading.RLock()


@dataclass(slots=True)
class LiveSession:
    id: str
    controller_ids: list[str]
    results: dict[str, Any]
    warnings: list[str]
    cursor: int = 0
    days: list[dict[str, Any]] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def _day_keys(self) -> list[tuple[int, str | None]]:
        keys = []
        source = next(iter(self.results.values()), {}).get("trace", [])
        for row in source:
            key = (int(row["day_index"]), row.get("date_iso"))
            if not keys or keys[-1] != key:
                keys.append(key)
        return keys

    def step(self) -> dict[str, Any] | None:
        keys = self._day_keys()
        if self.cursor >= len(keys):
            return None
        day_index, date_iso = keys[self.cursor]
        controllers = {}
        for controller_id, result in self.results.items():
            trace = [
                row for row in result["trace"]
                if int(row["day_index"]) == day_index and row.get("date_iso") == date_iso
            ]
            controllers[controller_id] = {"trace": trace, "kpi": result["kpi"], "meta": result["meta"]}
        entry = {
            "day": self.cursor + 1,
            "day_index": day_index,
            "date": date_iso,
            "controllers": controllers,
        }
        self.days.append(entry)
        self.cursor += 1
        return entry

    def public(self) -> dict[str, Any]:
        total = len(self._day_keys())
        return {
            "id": self.id,
            "controller_ids": self.controller_ids,
            "cursor": self.cursor,
            "total_days": total,
            "complete": self.cursor >= total,
            "warnings": self.warnings,
            "days": self.days,
            "created": self.created,
        }


def create_session(payload: dict[str, Any], parameters: dict[str, Any]) -> LiveSession:
    controller_ids = payload.get("controllers") or []
    if not isinstance(controller_ids, list) or not controller_ids or not all(
        isinstance(controller_id, str) for controller_id in controller_ids
    ):
        raise ValueError("select at least one live Brain controller")
    results, warnings = run_controllers(
        controller_ids, selected_data_path(parameters), parameters, CHECKPOINT_DIR
    )
    if not results:
        raise ValueError("no selected Brain controller produced a live trace")
    session = LiveSession(uuid.uuid4().hex[:10], list(results), results, warnings)
    with _LOCK:
        _SESSIONS[session.id] = session
    return session


def list_sessions() -> list[dict[str, Any]]:
    with _LOCK:
        return [session.public() for session in sorted(_SESSIONS.values(), key=lambda item: -item.created)]


def get_session(session_id: str) -> LiveSession:
    with _LOCK:
        session = _SESSIONS.get(session_id)
    if session is None:
        raise KeyError("live session not found")
    return session


def drop_session(session_id: str) -> bool:
    with _LOCK:
        return _SESSIONS.pop(session_id, None) is not None
