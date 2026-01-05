"""Tests for FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from dna_rag.core.chat_dna_enhanced import InterpretationResult, SNPResult
from dna_rag.models.api_models import (
    DNAQuestionRequest,
    PolygenicScoreRequest,
)


@pytest.fixture
def mock_settings() -> MagicMock:
    """Mock settings with API keys."""
    settings = MagicMock()
    settings.deepseek_api_key = "test-key"
    settings.openai_api_key = ""
    settings.openai_model = "gpt-4o-mini"
    settings.deepseek_model = "deepseek-chat"
    settings.validate_api_keys = MagicMock()
    return settings


@pytest.fixture
def mock_chat_dna() -> MagicMock:
    """Mock ChatDNA instance."""
    mock = MagicMock()
    mock.ask.return_value = "You are likely lactose tolerant based on your DNA."
    return mock


@pytest.fixture
def mock_chat_dna_enhanced() -> MagicMock:
    """Mock ChatDNAEnhanced instance."""
    mock = MagicMock()
    result = InterpretationResult(
        question="Am I lactose intolerant?",
        interpretation="You are likely lactose tolerant.",
        confidence=0.85,
        snps_found=[
            SNPResult(
                rsid="rs4988235",
                genotype="AG",
                gene="LCT",
                trait="lactose tolerance",
                validated=True,
                similarity=0.95,
            )
        ],
        sources=["dbSNP", "ClinVar"],
        caveats=["This is not medical advice"],
    )
    mock.ask.return_value = result
    return mock


@pytest.fixture
def client(
    mock_settings: MagicMock,
    mock_chat_dna: MagicMock,
    mock_chat_dna_enhanced: MagicMock,
) -> TestClient:
    """Create test client with mocked dependencies."""
    with patch("dna_rag.api.app.get_settings", return_value=mock_settings):
        with patch("dna_rag.api.app.ChatDNA", return_value=mock_chat_dna):
            with patch("dna_rag.api.app.ChatDNAEnhanced", return_value=mock_chat_dna_enhanced):
                # Set global instances
                import dna_rag.api.app as app_module
                from dna_rag.api.app import app

                app_module.chat_dna = mock_chat_dna
                app_module.chat_dna_enhanced = mock_chat_dna_enhanced
                app_module.settings = mock_settings

                return TestClient(app)


@pytest.fixture
def sample_dna_data() -> str:
    """Sample DNA data in CSV format."""
    return """rsid,chromosome,position,genotype
rs4988235,2,136608646,AG
rs1815739,11,66560624,CT
rs1042713,5,148826877,AA"""


def test_health_check(client: TestClient, mock_settings: MagicMock) -> None:
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert "providers" in data
    assert "deepseek" in data["providers"]


def test_ask_dna_question_success(
    client: TestClient, sample_dna_data: str, mock_chat_dna: MagicMock
) -> None:
    """Test successful DNA question."""
    request = DNAQuestionRequest(question="Am I lactose intolerant?", dna_data=sample_dna_data)

    response = client.post("/ask", json=request.model_dump())
    assert response.status_code == 200

    data = response.json()
    assert data["question"] == "Am I lactose intolerant?"
    assert "lactose tolerant" in data["answer"]
    assert data["provider"] == "deepseek"

    # Verify ChatDNA.ask was called
    mock_chat_dna.ask.assert_called_once()


def test_ask_dna_question_invalid_csv(client: TestClient) -> None:
    """Test DNA question with invalid CSV data."""
    request = DNAQuestionRequest(question="Am I lactose intolerant?", dna_data="invalid,csv,data")

    response = client.post("/ask", json=request.model_dump())
    assert response.status_code == 400
    assert "Failed to parse DNA data" in response.json()["detail"]


def test_ask_dna_question_missing_columns(client: TestClient) -> None:
    """Test DNA question with missing required columns."""
    dna_data = "rsid,genotype\nrs123,AA"
    request = DNAQuestionRequest(question="test", dna_data=dna_data)

    response = client.post("/ask", json=request.model_dump())
    assert response.status_code == 400
    assert "must contain columns" in response.json()["detail"]


def test_ask_enhanced_success(
    client: TestClient,
    sample_dna_data: str,
    mock_chat_dna_enhanced: MagicMock,
) -> None:
    """Test enhanced DNA question endpoint."""
    request = DNAQuestionRequest(
        question="Am I lactose intolerant?",
        dna_data=sample_dna_data,
        use_rag=True,
    )

    response = client.post("/ask-enhanced", json=request.model_dump())
    assert response.status_code == 200

    data = response.json()
    assert data["question"] == "Am I lactose intolerant?"
    assert "lactose tolerant" in data["interpretation"]
    assert data["confidence"] == 0.85
    assert len(data["snps_found"]) == 1
    assert data["snps_found"][0]["rsid"] == "rs4988235"
    assert data["snps_found"][0]["validated"] is True
    assert data["sources"] == ["dbSNP", "ClinVar"]
    assert data["caveats"] == ["This is not medical advice"]

    # Verify ChatDNAEnhanced.ask was called
    mock_chat_dna_enhanced.ask.assert_called_once()


def test_polygenic_score_success(client: TestClient, sample_dna_data: str) -> None:
    """Test polygenic score calculation."""
    with patch("dna_rag.api.app.PolygenicScoreCalculator") as mock_calculator_class:
        mock_calculator = MagicMock()
        mock_calculator.calculate_score.return_value = {
            "score": 1.25,
            "interpretation": "Elevated risk",
            "percentile": 75.5,
        }
        mock_calculator_class.return_value = mock_calculator

        request = PolygenicScoreRequest(score_name="alzheimers_risk", dna_data=sample_dna_data)

        response = client.post("/polygenic-score", json=request.model_dump())
        assert response.status_code == 200

        data = response.json()
        assert data["score_name"] == "alzheimers_risk"
        assert data["score"] == 1.25
        assert data["interpretation"] == "Elevated risk"
        assert data["percentile"] == 75.5

        # Verify calculator was called
        mock_calculator.calculate_score.assert_called_once()


def test_polygenic_score_invalid_data(client: TestClient) -> None:
    """Test polygenic score with invalid DNA data."""
    request = PolygenicScoreRequest(score_name="alzheimers_risk", dna_data="invalid,csv")

    response = client.post("/polygenic-score", json=request.model_dump())
    assert response.status_code == 400


def test_polygenic_score_unknown_score(client: TestClient, sample_dna_data: str) -> None:
    """Test polygenic score with unknown score name."""
    with patch("dna_rag.api.app.PolygenicScoreCalculator") as mock_calculator_class:
        mock_calculator = MagicMock()
        mock_calculator.calculate_score.side_effect = KeyError("unknown_score")
        mock_calculator_class.return_value = mock_calculator

        request = PolygenicScoreRequest(score_name="unknown_score", dna_data=sample_dna_data)

        response = client.post("/polygenic-score", json=request.model_dump())
        assert response.status_code == 400
        assert "Unknown polygenic score" in response.json()["detail"]


def test_cors_headers(client: TestClient) -> None:
    """Test CORS headers are present."""
    response = client.get("/health")
    assert response.status_code == 200
    # CORS middleware adds these headers
    assert "access-control-allow-origin" in response.headers


def test_openapi_schema(client: TestClient) -> None:
    """Test OpenAPI schema is accessible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert schema["info"]["title"] == "DNA RAG API"
    assert schema["info"]["version"] == "0.1.0"
    assert "/health" in schema["paths"]
    assert "/ask" in schema["paths"]
    assert "/ask-enhanced" in schema["paths"]
    assert "/polygenic-score" in schema["paths"]
