"""Tests for the analysis endpoints."""

from __future__ import annotations

import io
import time

from fastapi.testclient import TestClient


class TestSyncAnalyze:
    """POST /api/v1/analyze (synchronous)."""

    def test_analyze_with_upload(self, client: TestClient, sample_dna_bytes: bytes):
        resp = client.post(
            "/api/v1/analyze",
            data={"question": "test trait"},
            files={"file": ("dna.txt", io.BytesIO(sample_dna_bytes), "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "test trait"
        assert data["snp_count_matched"] >= 1
        assert data["interpretation"] != ""
        assert data["id"].startswith("ana_")
        assert "processing_time_ms" in data

    def test_analyze_with_file_id(
        self, client: TestClient, uploaded_file_id: str,
    ):
        resp = client.post(
            "/api/v1/analyze",
            data={"question": "test trait", "file_id": uploaded_file_id},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["snp_count_matched"] >= 1

    def test_analyze_no_file(self, client: TestClient):
        resp = client.post(
            "/api/v1/analyze",
            data={"question": "test trait"},
        )
        assert resp.status_code == 400

    def test_analyze_bad_file_id(self, client: TestClient):
        resp = client.post(
            "/api/v1/analyze",
            data={"question": "test", "file_id": "file_nonexistent"},
        )
        assert resp.status_code == 404

    def test_analyze_llm_models_in_response(
        self, client: TestClient, uploaded_file_id: str,
    ):
        resp = client.post(
            "/api/v1/analyze",
            data={"question": "test", "file_id": uploaded_file_id},
        )
        data = resp.json()
        assert "llm_models" in data
        assert data["llm_models"]["snp_identification"] != ""


class TestAsyncAnalyze:
    """POST /api/v1/analyze/async."""

    def test_submit_job(self, client: TestClient, uploaded_file_id: str):
        resp = client.post(
            "/api/v1/analyze/async",
            data={"question": "async test", "file_id": uploaded_file_id},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"].startswith("job_")
        assert data["status"] == "pending"
        assert "/api/v1/jobs/" in data["poll_url"]

    def test_submit_job_with_upload(
        self, client: TestClient, sample_dna_bytes: bytes,
    ):
        resp = client.post(
            "/api/v1/analyze/async",
            data={"question": "async upload test"},
            files={"file": ("dna.txt", io.BytesIO(sample_dna_bytes), "text/plain")},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"].startswith("job_")

    def test_poll_job_result(self, client: TestClient, uploaded_file_id: str):
        # Submit
        resp = client.post(
            "/api/v1/analyze/async",
            data={"question": "poll test", "file_id": uploaded_file_id},
        )
        job_id = resp.json()["job_id"]

        # Poll until completed (generous timeout for slow CI runners)
        for _ in range(50):
            time.sleep(0.2)
            poll = client.get(f"/api/v1/jobs/{job_id}")
            assert poll.status_code == 200
            if poll.json()["status"] in ("completed", "failed"):
                break

        data = poll.json()
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["result"]["snp_count_matched"] >= 1

    def test_submit_no_file(self, client: TestClient):
        resp = client.post(
            "/api/v1/analyze/async",
            data={"question": "test"},
        )
        assert resp.status_code == 400
