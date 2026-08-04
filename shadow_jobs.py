from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ShadowJob:
    id: str
    created: float
    status: str = "queued"
    stage: str = "Waiting for Shadow HQ"
    completed: int = 0
    total: int = 0
    current_date: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created": self.created,
            "status": self.status,
            "stage": self.stage,
            "completed": self.completed,
            "total": self.total,
            "current_date": self.current_date,
            "error": self.error,
            "result": self.result,
            "elapsed_seconds": round(time.time() - self.created, 1),
        }


class ShadowJobManager:
    def __init__(self):
        self._jobs: dict[str, ShadowJob] = {}
        self._lock = threading.Lock()

    def start(self, runner: Callable) -> ShadowJob:
        job = ShadowJob(uuid.uuid4().hex, time.time())
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, runner), daemon=True).start()
        return job

    def _run(self, job: ShadowJob, runner: Callable) -> None:
        job.status = "running"

        def progress(stage: str, completed: int, total: int, current_date: str | None = None):
            job.stage = stage
            job.completed = completed
            job.total = total
            job.current_date = current_date

        try:
            job.result = runner(progress, job.cancel_event.is_set)
            job.status = "cancelled" if job.cancel_event.is_set() else "complete"
            job.stage = "Shadow catch-up cancelled" if job.cancel_event.is_set() else "Shadow catch-up complete"
            if job.status == "complete":
                job.completed = job.total
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.stage = "Shadow catch-up failed"
            job.error = str(exc)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
        return job.public() if job else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job or job.status not in {"queued", "running"}:
            return False
        job.cancel_event.set()
        return True


MANAGER = ShadowJobManager()
