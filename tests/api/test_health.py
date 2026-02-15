"""Tests for health / readiness / formats endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dna_rag._version import __version__


class TestHealth:
    """GET /health."""

    def test_returns_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == __version__

    def test_has_request_id_header(self, client: TestClient):
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers


class TestReady:
    """GET /ready."""

    def test_returns_ok(self, client: TestClient):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")


class TestFormats:
    """GET /api/v1/formats."""

    def test_returns_all_formats(self, client: TestClient):
        resp = client.get("/api/v1/formats")
        assert resp.status_code == 200
        fmts = resp.json()["formats"]
        assert "23andme" in fmts
        assert "ancestrydna" in fmts
        assert "myheritage" in fmts
