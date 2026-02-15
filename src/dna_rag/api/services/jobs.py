"""In-memory job store for async analysis.

In production this would be backed by Redis + arq / Celery.
For the PoC we run jobs in ``asyncio`` background tasks and store
results in a simple dict.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, Field


class Job(BaseModel):
    """Represents an async analysis job."""

    job_id: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    question: str = ""
    file_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None  # serialised AnalysisResponse
    error: str | None = None


class JobStore:
    """Thread-safe in-memory job store.

    Args:
        ttl_seconds: Time after which completed/failed jobs are eligible
            for eviction.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds

    def create(self, question: str, file_id: str) -> Job:
        """Create a new pending job and return it."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = Job(job_id=job_id, question=question, file_id=file_id)
        with self._lock:
            self._evict_stale()
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        """Retrieve a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def set_running(self, job_id: str) -> None:
        """Mark a job as running."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "running"
                job.started_at = datetime.now(UTC)

    def set_completed(self, job_id: str, result: dict[str, Any]) -> None:
        """Mark a job as completed with *result*."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "completed"
                job.completed_at = datetime.now(UTC)
                job.result = result

    def set_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed with an error message."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "failed"
                job.completed_at = datetime.now(UTC)
                job.error = error

    def _evict_stale(self) -> None:
        """Remove jobs that have exceeded their TTL (called under lock)."""
        stale = [
            jid
            for jid, job in self._jobs.items()
            if job.status in ("completed", "failed")
            and job.completed_at is not None
            and (datetime.now(UTC) - job.completed_at).total_seconds()
            > self._ttl
        ]
        for jid in stale:
            del self._jobs[jid]
