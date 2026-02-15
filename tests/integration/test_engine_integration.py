"""Integration tests for the full analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from dna_rag.cache.memory import InMemoryCache
from dna_rag.engine import DNAAnalysisEngine
from tests.conftest import FakeLLMProvider

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
_INTERPRETATION = "Analysis complete."


class TestMultiFormat:
    """The engine works with all supported DNA formats."""

    def test_23andme(self, sample_23andme_file: Path):
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, cache=InMemoryCache())
        result = engine.analyze("test", sample_23andme_file)
        assert result.snp_count_matched >= 1

    def test_myheritage(self, sample_myheritage_file: Path):
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, cache=InMemoryCache())
        result = engine.analyze("test", sample_myheritage_file)
        assert result.snp_count_matched >= 1

    def test_ancestrydna(self, sample_ancestrydna_file: Path):
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, cache=InMemoryCache())
        result = engine.analyze("test", sample_ancestrydna_file)
        assert result.snp_count_matched >= 1


class TestReusesCache:
    """Multiple questions on the same file reuse caches."""

    def test_multiple_questions_same_file(self, sample_23andme_file: Path):
        snp_json_2 = json.dumps(
            {
                "rs3": {
                    "gene": "MCM6",
                    "chromosome": "2",
                    "position": 200,
                    "trait": "Persistence",
                }
            }
        )
        llm = FakeLLMProvider([
            _SNP_JSON, "interp1",
            snp_json_2, "interp2",
        ])
        cache = InMemoryCache()
        engine = DNAAnalysisEngine(snp_llm=llm, cache=cache)

        r1 = engine.analyze("lactose", sample_23andme_file)
        r2 = engine.analyze("persistence", sample_23andme_file)

        assert r1.question == "lactose"
        assert r2.question == "persistence"
        # DNA file should be loaded only once (cached)
        # 4 LLM calls: snp1, interp1, snp2, interp2
        assert len(llm.calls) == 4


class TestResultSerialization:
    """AnalysisResult can be serialized to/from JSON."""

    def test_json_roundtrip(self, sample_23andme_file: Path):
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm)

        result = engine.analyze("test", sample_23andme_file)
        serialized = result.model_dump_json()
        data = json.loads(serialized)

        assert data["question"] == "test"
        assert data["interpretation"] == _INTERPRETATION
        assert isinstance(data["matched_snps"], list)
        assert data["snp_count_matched"] >= 1
