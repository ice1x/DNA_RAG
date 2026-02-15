"""Tests for middleware (auth, rate-limit, request-id)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from dna_rag.api.config import APISettings
from dna_rag.api.main import create_app
from dna_rag.api.services.files import FileService
from dna_rag.cache.memory import InMemoryCache
from dna_rag.engine import DNAAnalysisEngine
from tests.api.conftest import INTERPRETATION, SNP_JSON
from tests.conftest import FakeLLMProvider


def _make_client(tmp_path: Path, **overrides) -> TestClient:  # noqa: ANN003
    """Helper: build a TestClient with custom settings."""
    settings = APISettings(
        llm_api_key="test-key",  # type: ignore[arg-type]
        llm_provider="deepseek",
        cache_backend="memory",
        upload_dir=str(tmp_path / "uploads"),
        log_level="WARNING",
        **overrides,
    )
    app = create_app(settings=settings)

    # Override DI
    from dna_rag.api import dependencies
    from dna_rag.api.services.analysis import AnalysisService
    from dna_rag.api.services.jobs import JobStore

    llm = FakeLLMProvider([SNP_JSON, INTERPRETATION])
    engine = DNAAnalysisEngine(snp_llm=llm, cache=InMemoryCache())
    file_svc = FileService(upload_dir=str(tmp_path / "uploads"))
    analysis_svc = AnalysisService(engine=engine, file_service=file_svc, settings=settings)
    job_store = JobStore()

    app.dependency_overrides[dependencies.get_analysis_service] = lambda: analysis_svc
    app.dependency_overrides[dependencies.get_file_service] = lambda: file_svc
    app.dependency_overrides[dependencies.get_job_store] = lambda: job_store
    app.dependency_overrides[dependencies.get_engine] = lambda: engine

    return TestClient(app)


class TestRequestId:
    """X-Request-ID middleware."""

    def test_assigns_request_id(self, client: TestClient):
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) > 0

    def test_propagates_client_id(self, client: TestClient):
        custom = "my-custom-request-id"
        resp = client.get("/health", headers={"X-Request-ID": custom})
        assert resp.headers["X-Request-ID"] == custom


class TestAuth:
    """API key auth middleware."""

    def test_auth_disabled_allows_all(self, client: TestClient):
        """Default client has auth_enabled=False."""
        resp = client.get("/api/v1/formats")
        assert resp.status_code == 200

    def test_auth_enabled_rejects_missing_token(self, tmp_path: Path):
        c = _make_client(tmp_path, auth_enabled=True, api_keys=["valid-key"])
        resp = c.get("/api/v1/formats")
        assert resp.status_code == 401

    def test_auth_enabled_rejects_invalid_token(self, tmp_path: Path):
        c = _make_client(tmp_path, auth_enabled=True, api_keys=["valid-key"])
        resp = c.get(
            "/api/v1/formats",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

    def test_auth_enabled_accepts_valid_token(self, tmp_path: Path):
        c = _make_client(tmp_path, auth_enabled=True, api_keys=["valid-key"])
        resp = c.get(
            "/api/v1/formats",
            headers={"Authorization": "Bearer valid-key"},
        )
        assert resp.status_code == 200

    def test_health_exempt_from_auth(self, tmp_path: Path):
        c = _make_client(tmp_path, auth_enabled=True, api_keys=["valid-key"])
        resp = c.get("/health")
        assert resp.status_code == 200


class TestRateLimit:
    """Rate limiting middleware."""

    def test_headers_present(self, client: TestClient):
        resp = client.get("/api/v1/formats")
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers

    def test_rate_limit_enforced(self, tmp_path: Path):
        c = _make_client(tmp_path, rate_limit_per_minute=3)
        for _ in range(3):
            resp = c.get("/api/v1/formats")
            assert resp.status_code == 200

        resp = c.get("/api/v1/formats")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_health_exempt_from_rate_limit(self, tmp_path: Path):
        c = _make_client(tmp_path, rate_limit_per_minute=1)
        # Exhaust limit
        c.get("/api/v1/formats")
        c.get("/api/v1/formats")
        # Health should still work
        resp = c.get("/health")
        assert resp.status_code == 200
