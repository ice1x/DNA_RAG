"""Tests for job status endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestJobStatus:
    """GET /api/v1/jobs/{job_id}."""

    def test_nonexistent_job(self, client: TestClient):
        resp = client.get("/api/v1/jobs/job_nonexistent")
        assert resp.status_code == 404

    def test_job_response_shape(self, client: TestClient, uploaded_file_id: str):
        # Submit a job first
        resp = client.post(
            "/api/v1/analyze/async",
            data={"question": "shape test", "file_id": uploaded_file_id},
        )
        job_id = resp.json()["job_id"]

        # Check shape immediately
        poll = client.get(f"/api/v1/jobs/{job_id}")
        data = poll.json()
        assert "job_id" in data
        assert "status" in data
        assert "created_at" in data
