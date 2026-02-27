"""Tests for the RAG-augmented and NCBI-validated engine pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dna_rag.cache.memory import InMemoryCache
from dna_rag.engine import DNAAnalysisEngine
from dna_rag.exceptions import NoSNPsFoundError
from dna_rag.snp_database import SNPValidationResult
from tests.conftest import FakeLLMProvider

# ---------------------------------------------------------------------------
# Helpers & Mocks
# ---------------------------------------------------------------------------

_SNP_JSON = json.dumps(
    {
        "rs1": {
            "gene": "LCT",
            "chromosome": "1",
            "position": 111,
            "trait": "Lactose tolerance",
        }
    }
)

_SNP_JSON_TWO = json.dumps(
    {
        "rs1": {
            "gene": "LCT",
            "chromosome": "1",
            "position": 111,
            "trait": "Lactose tolerance",
        },
        "rs_fake": {
            "gene": "FAKE",
            "chromosome": "1",
            "position": 999,
            "trait": "Fake trait",
        },
    }
)

_INTERPRETATION = "Based on your genotype AA at rs1, you are likely lactose tolerant."


class MockVectorStore:
    """Returns pre-configured search results."""

    def __init__(self, results: dict[str, dict[str, Any]]) -> None:
        self._results = results
        self.calls: list[str] = []

    def search(
        self,
        query: str,
        n_results: int = 10,
        min_similarity: float = 0.3,
    ) -> dict[str, dict[str, Any]]:
        self.calls.append(query)
        return self._results


class ErrorVectorStore:
    """Always raises on search."""

    def search(self, query: str, **kwargs: Any) -> dict:
        raise RuntimeError("Vector store unavailable")


class MockSNPDatabase:
    """Validates RSIDs against a known set."""

    def __init__(self, valid_rsids: set[str]) -> None:
        self._valid = valid_rsids

    def validate_batch(self, rsids: list[str]) -> dict[str, SNPValidationResult]:
        return {
            rsid: SNPValidationResult(
                rsid=rsid,
                exists=(rsid in self._valid),
                validated=True,
                source="mock",
            )
            for rsid in rsids
        }


class ErrorSNPDatabase:
    """Always raises on validate_batch."""

    def validate_batch(self, rsids: list[str]) -> dict:
        raise RuntimeError("NCBI unavailable")


# ---------------------------------------------------------------------------
# RAG context injection
# ---------------------------------------------------------------------------


class TestRAGContext:
    """Vector store context injection into the LLM prompt."""

    def test_rag_context_injected_into_prompt(self, sample_23andme_file: Path):
        """When vector store returns results, the LLM prompt contains them."""
        mock_vs = MockVectorStore({
            "rs1": {"trait": "Lactose tolerance", "gene": "LCT", "similarity": 0.9},
        })
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, vector_store=mock_vs)

        engine.analyze("lactose", sample_23andme_file)

        # First LLM call (SNP identification) should contain RAG hints
        assert "Known SNPs" in llm.calls[0]
        assert "rs1" in llm.calls[0]
        assert "LCT" in llm.calls[0]

    def test_no_vector_store_works(self, sample_23andme_file: Path):
        """Engine works without a vector store (backward compat)."""
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, vector_store=None)

        result = engine.analyze("lactose", sample_23andme_file)
        assert result.snp_count_matched == 1
        assert result.rag_context_used is False

    def test_vector_store_error_graceful_degradation(self, sample_23andme_file: Path):
        """If vector store throws, the engine continues without RAG context."""
        mock_vs = ErrorVectorStore()
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, vector_store=mock_vs)

        result = engine.analyze("lactose", sample_23andme_file)
        assert result.snp_count_matched == 1
        assert result.rag_context_used is False

    def test_vector_store_empty_results(self, sample_23andme_file: Path):
        """Empty vector store results don't modify the prompt."""
        mock_vs = MockVectorStore({})
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, vector_store=mock_vs)

        engine.analyze("lactose", sample_23andme_file)
        # Prompt should NOT contain RAG hints section
        assert "Known SNPs" not in llm.calls[0]

    def test_rag_context_used_flag(self, sample_23andme_file: Path):
        """AnalysisResult.rag_context_used is True when RAG provides context."""
        mock_vs = MockVectorStore({
            "rs1": {"trait": "Lactose tolerance", "gene": "LCT", "similarity": 0.9},
        })
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, vector_store=mock_vs)

        result = engine.analyze("lactose", sample_23andme_file)
        assert result.rag_context_used is True

    def test_vector_store_search_params_forwarded(self, sample_23andme_file: Path):
        """Custom rag_search_results and rag_min_similarity are forwarded."""
        call_args: list[dict[str, Any]] = []

        class SpyVectorStore:
            def search(self, query: str, n_results: int = 10, min_similarity: float = 0.3):
                call_args.append({"n_results": n_results, "min_similarity": min_similarity})
                return {"rs1": {"trait": "T", "gene": "G", "similarity": 0.8}}

        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(
            snp_llm=llm,
            vector_store=SpyVectorStore(),
            rag_search_results=5,
            rag_min_similarity=0.5,
        )
        engine.analyze("test", sample_23andme_file)

        assert call_args[0]["n_results"] == 5
        assert call_args[0]["min_similarity"] == 0.5


# ---------------------------------------------------------------------------
# NCBI validation
# ---------------------------------------------------------------------------


class TestNCBIValidation:
    """NCBI SNP validation integration."""

    def test_invalid_snps_filtered_out(self, sample_23andme_file: Path):
        """SNPs that NCBI says don't exist are removed."""
        mock_db = MockSNPDatabase(valid_rsids={"rs1"})
        llm = FakeLLMProvider([_SNP_JSON_TWO, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, snp_database=mock_db)

        result = engine.analyze("lactose", sample_23andme_file)
        assert result.snp_count_requested == 2  # LLM said 2
        assert result.snp_count_matched == 1    # only rs1 survived validation + file match
        assert result.validation_used is True

    def test_all_snps_invalid_raises(self, sample_23andme_file: Path):
        """If all SNPs are invalid, NoSNPsFoundError is raised."""
        mock_db = MockSNPDatabase(valid_rsids=set())
        llm = FakeLLMProvider([_SNP_JSON])
        engine = DNAAnalysisEngine(snp_llm=llm, snp_database=mock_db)

        with pytest.raises(NoSNPsFoundError, match="No valid SNPs"):
            engine.analyze("lactose", sample_23andme_file)

    def test_no_database_works(self, sample_23andme_file: Path):
        """Engine works without SNP database (backward compat)."""
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, snp_database=None)

        result = engine.analyze("lactose", sample_23andme_file)
        assert result.validation_used is False

    def test_database_error_graceful_degradation(self, sample_23andme_file: Path):
        """If NCBI validation throws, engine continues with all SNPs."""
        mock_db = ErrorSNPDatabase()
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, snp_database=mock_db)

        result = engine.analyze("lactose", sample_23andme_file)
        assert result.snp_count_matched == 1
        # validation_used is True because db was configured, even if it errored
        assert result.validation_used is True


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


class TestRAGAndValidationCombined:
    """Both RAG and NCBI validation enabled together."""

    def test_both_enabled(self, sample_23andme_file: Path):
        """RAG context + NCBI validation work together."""
        mock_vs = MockVectorStore({
            "rs1": {"trait": "Lactose tolerance", "gene": "LCT", "similarity": 0.9},
        })
        mock_db = MockSNPDatabase(valid_rsids={"rs1"})
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(
            snp_llm=llm,
            vector_store=mock_vs,
            snp_database=mock_db,
        )

        result = engine.analyze("lactose", sample_23andme_file)
        assert result.rag_context_used is True
        assert result.validation_used is True
        assert result.snp_count_matched == 1

    def test_rag_with_cache(self, sample_23andme_file: Path):
        """RAG context works alongside the existing cache."""
        mock_vs = MockVectorStore({
            "rs1": {"trait": "Lactose tolerance", "gene": "LCT", "similarity": 0.9},
        })
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        cache = InMemoryCache()
        engine = DNAAnalysisEngine(
            snp_llm=llm,
            vector_store=mock_vs,
            cache=cache,
        )

        r1 = engine.analyze("lactose", sample_23andme_file)
        r2 = engine.analyze("lactose", sample_23andme_file)

        assert r1.cached is False
        assert r2.cached is True
        assert r1.rag_context_used is True
        # Only 2 LLM calls (first run): SNP + interpretation
        assert len(llm.calls) == 2
