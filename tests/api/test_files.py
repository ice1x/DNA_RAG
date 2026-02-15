"""Tests for file upload / management endpoints."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient


class TestFileUpload:
    """POST /api/v1/files."""

    def test_upload_valid_file(self, client: TestClient, sample_dna_bytes: bytes):
        resp = client.post(
            "/api/v1/files",
            files={"file": ("sample.txt", io.BytesIO(sample_dna_bytes), "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["file_id"].startswith("file_")
        assert data["row_count"] == 3
        assert data["size_bytes"] > 0

    def test_upload_deduplication(self, client: TestClient, sample_dna_bytes: bytes):
        """Uploading the same file twice returns the same file_id."""
        resp1 = client.post(
            "/api/v1/files",
            files={"file": ("a.txt", io.BytesIO(sample_dna_bytes), "text/plain")},
        )
        resp2 = client.post(
            "/api/v1/files",
            files={"file": ("b.txt", io.BytesIO(sample_dna_bytes), "text/plain")},
        )
        assert resp1.json()["file_id"] == resp2.json()["file_id"]

    def test_upload_invalid_format(self, client: TestClient):
        bad = b"this is not a dna file at all\nrandom data\n"
        resp = client.post(
            "/api/v1/files",
            files={"file": ("bad.txt", io.BytesIO(bad), "text/plain")},
        )
        assert resp.status_code == 400


class TestFileGet:
    """GET /api/v1/files/{file_id}."""

    def test_get_existing(self, client: TestClient, uploaded_file_id: str):
        resp = client.get(f"/api/v1/files/{uploaded_file_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_id"] == uploaded_file_id
        assert data["row_count"] == 3

    def test_get_nonexistent(self, client: TestClient):
        resp = client.get("/api/v1/files/file_nonexistent")
        assert resp.status_code == 404


class TestFileDelete:
    """DELETE /api/v1/files/{file_id}."""

    def test_delete_existing(self, client: TestClient, uploaded_file_id: str):
        resp = client.delete(f"/api/v1/files/{uploaded_file_id}")
        assert resp.status_code == 204

        # Confirm it's gone
        resp2 = client.get(f"/api/v1/files/{uploaded_file_id}")
        assert resp2.status_code == 404

    def test_delete_nonexistent(self, client: TestClient):
        resp = client.delete("/api/v1/files/file_nonexistent")
        assert resp.status_code == 404
