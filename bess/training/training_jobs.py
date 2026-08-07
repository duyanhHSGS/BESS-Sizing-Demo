from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"
    lines: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    proc: subprocess.Popen | None = None
    created: float = field(default_factory=time.time)


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _new(self, kind: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def detail(self, job_id: str) -> dict | None:
        job = self.get(job_id)
        if not job:
            return None
        return {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "lines": job.lines[-200:],
            "result": job.result,
            "error": job.error,
            "created": job.created,
        }

    def list(self) -> list[dict]:
        return [
            {
                "id": job.id,
                "kind": job.kind,
                "status": job.status,
                "created": job.created,
                "n_lines": len(job.lines),
            }
            for job in sorted(self._jobs.values(), key=lambda item: -item.created)
        ]

    def start_thread(self, kind: str, fn: Callable[[Callable[[str], None]], dict | None]) -> Job:
        job = self._new(kind)

        def _run():
            try:
                job.result = fn(lambda message: job.lines.append(str(message)))
                job.status = "done"
            except Exception as exc:  # noqa: BLE001
                import traceback

                job.status = "error"
                job.error = str(exc)
                job.lines.append(traceback.format_exc()[-2000:])

        threading.Thread(target=_run, daemon=True).start()
        return job

    def start_subprocess(
        self,
        kind: str,
        cmd: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> Job:
        job = self._new(kind)

        def _run():
            try:
                job.proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert job.proc.stdout is not None
                for line in job.proc.stdout:
                    job.lines.append(line.rstrip())
                rc = job.proc.wait()
                if job.status == "stopped":
                    return
                job.status = "done" if rc == 0 else "error"
                if rc != 0:
                    job.error = f"exit code {rc}"
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = str(exc)

        threading.Thread(target=_run, daemon=True).start()
        return job

    def stop(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job.status != "running":
            return False
        if job.proc is None:
            return False
        job.status = "stopped"
        job.proc.kill()
        return True

    def sse_events(self, job_id: str) -> Iterable[str]:
        job = self.get(job_id)
        if job is None:
            yield "event: error\ndata: job not found\n\n"
            return
        sent = 0
        while True:
            while sent < len(job.lines):
                yield f"data: {job.lines[sent]}\n\n"
                sent += 1
            if job.status != "running":
                yield f"event: end\ndata: {job.status}\n\n"
                return
            time.sleep(0.5)


MANAGER = JobManager()
