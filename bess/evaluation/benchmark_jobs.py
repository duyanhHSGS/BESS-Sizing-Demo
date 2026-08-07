from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BenchmarkJob:
    id: str
    created: float
    status: str = "queued"
    stage: str = "Waiting for Brain HQ"
    completed: int = 0
    total: int = 0
    fighter: str | None = None
    error: str | None = None
    run_id: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created": self.created,
            "status": self.status,
            "stage": self.stage,
            "completed": self.completed,
            "total": self.total,
            "fighter": self.fighter,
            "error": self.error,
            "run_id": self.run_id,
            "elapsed_seconds": round(time.time() - self.created, 1),
        }


class BenchmarkJobManager:
    def __init__(self):
        self._jobs: dict[str, BenchmarkJob] = {}
        self._lock = threading.Lock()

    def start(self, runner: Callable) -> BenchmarkJob:
        job = BenchmarkJob(id=uuid.uuid4().hex, created=time.time())
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, runner), daemon=True).start()
        return job

    def _run(self, job: BenchmarkJob, runner: Callable) -> None:
        job.status = "running"

        def progress(stage: str, completed: int, total: int, fighter: str | None = None):
            job.stage = stage
            job.completed = completed
            job.total = total
            job.fighter = fighter

        try:
            result = runner(progress, job.cancel_event.is_set)
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.stage = "Tournament cancelled; no result was saved."
                return
            job.run_id = result["id"]
            job.status = "complete"
            job.stage = "Tournament complete"
            job.completed = job.total
        except BenchmarkCancelled:
            job.status = "cancelled"
            job.stage = "Tournament cancelled; no result was saved."
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.stage = "Tournament aborted; no partial leaderboard was saved."
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


class BenchmarkCancelled(RuntimeError):
    pass


MANAGER = BenchmarkJobManager()
