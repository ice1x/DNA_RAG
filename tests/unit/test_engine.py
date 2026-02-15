"""Tests for the core DNA analysis engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dna_rag.cache.memory import InMemoryCache
from dna_rag.engine import DNAAnalysisEngine
from dna_rag.exceptions import (
    LLMResponseError,
    NoMatchingVariantsError,
    NoSNPsFoundError,
)
from dna_rag.models import SNPMetadata
from tests.conftest import FakeLLMProvider

# ---------------------------------------------------------------------------
# Helpers
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

_SNP_JSON_MULTI = json.dumps(
    {
        "rs1": {
            "gene": "LCT",
            "chromosome": "1",
            "position": 111,
            "trait": "Lactose tolerance",
        },
        "rs3": {
            "gene": "MCM6",
            "chromosome": "2",
            "position": 200,
            "trait": "Lactase persistence",
        },
    }
)

_INTERPRETATION = "Based on your genotype AA at rs1, you are likely lactose tolerant."


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    """End-to-end pipeline with FakeLLMProvider."""

    def test_full_pipeline(self, sample_23andme_file: Path):
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, cache=InMemoryCache())

        result = engine.analyze("lactose tolerance", sample_23andme_file)

        assert result.question == "lactose tolerance"
        assert result.interpretation == _INTERPRETATION
        assert result.snp_count_requested == 1
        assert result.snp_count_matched == 1
        assert result.cached is False
        assert len(result.matched_snps) == 1
        assert result.matched_snps[0].rsid == "rs1"
        assert result.matched_snps[0].genotype == "AA"
        assert result.matched_snps[0].gene == "LCT"

    def test_multiple_matches(self, sample_23andme_file: Path):
        llm = FakeLLMProvider([_SNP_JSON_MULTI, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, cache=InMemoryCache())

        result = engine.analyze("lactose", sample_23andme_file)
        assert result.snp_count_requested == 2
        assert result.snp_count_matched == 2


class TestSeparateLLMs:
    """Engine uses different LLMs for SNP identification and interpretation."""

    def test_two_separate_llms(self, sample_23andme_file: Path):
        snp_llm = FakeLLMProvider([_SNP_JSON])
        interp_llm = FakeLLMProvider(["Interpretation by second LLM"])
        engine = DNAAnalysisEngine(
            snp_llm=snp_llm,
            interpretation_llm=interp_llm,
            cache=InMemoryCache(),
        )

        result = engine.analyze("test", sample_23andme_file)

        assert len(snp_llm.calls) == 1  # SNP step called snp_llm
        assert len(interp_llm.calls) == 1  # Interp step called interp_llm
        assert result.interpretation == "Interpretation by second LLM"

    def test_single_llm_used_for_both(self, sample_23andme_file: Path):
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm)

        engine.analyze("test", sample_23andme_file)
        assert len(llm.calls) == 2  # Both steps used same LLM


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """Pipeline errors raise appropriate exceptions."""

    def test_no_snps_found(self, sample_23andme_file: Path):
        llm = FakeLLMProvider(["{}"])
        engine = DNAAnalysisEngine(snp_llm=llm)
        with pytest.raises(NoSNPsFoundError, match="No relevant SNPs"):
            engine.analyze("nonsense question", sample_23andme_file)

    def test_no_matching_variants(self, sample_23andme_file: Path):
        snp_json = json.dumps(
            {
                "rs999": {
                    "gene": "X",
                    "chromosome": "1",
                    "position": 999,
                    "trait": "T",
                }
            }
        )
        llm = FakeLLMProvider([snp_json])
        engine = DNAAnalysisEngine(snp_llm=llm)
        with pytest.raises(NoMatchingVariantsError, match="No matching"):
            engine.analyze("test", sample_23andme_file)

    def test_invalid_json_response(self, sample_23andme_file: Path):
        llm = FakeLLMProvider(["not valid json at all"])
        engine = DNAAnalysisEngine(snp_llm=llm)
        with pytest.raises(LLMResponseError, match="invalid JSON"):
            engine.analyze("test", sample_23andme_file)

    def test_non_dict_json_response(self, sample_23andme_file: Path):
        llm = FakeLLMProvider(['["rs1", "rs2"]'])
        engine = DNAAnalysisEngine(snp_llm=llm)
        with pytest.raises(LLMResponseError, match="Expected JSON object"):
            engine.analyze("test", sample_23andme_file)

    def test_file_not_found(self):
        llm = FakeLLMProvider([])
        engine = DNAAnalysisEngine(snp_llm=llm)
        with pytest.raises(FileNotFoundError):
            engine.analyze("test", Path("/nonexistent/file.txt"))


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    """Cache behaviour."""

    def test_answer_cache_hit(self, sample_23andme_file: Path):
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        cache = InMemoryCache()
        engine = DNAAnalysisEngine(snp_llm=llm, cache=cache)

        r1 = engine.analyze("lactose", sample_23andme_file)
        r2 = engine.analyze("lactose", sample_23andme_file)

        assert r1.cached is False
        assert r2.cached is True
        # LLM should only be called twice (first run): SNP + interpret
        assert len(llm.calls) == 2

    def test_snp_cache_reuse_different_files(self, tmp_path: Path):
        """Same question, different DNA files -> SNP cache reused."""
        f1 = tmp_path / "file1.txt"
        f1.write_text("# 23andMe\nrs1\t1\t111\tAA\n")
        f2 = tmp_path / "file2.txt"
        f2.write_text("# 23andMe\nrs1\t1\t111\tGG\n")

        llm = FakeLLMProvider([_SNP_JSON, "interp1", "interp2"])
        cache = InMemoryCache()
        engine = DNAAnalysisEngine(snp_llm=llm, cache=cache)

        engine.analyze("lactose", f1)
        engine.analyze("lactose", f2)

        # SNP lookup should only happen once (cached),
        # interpretation twice (different files)
        assert len(llm.calls) == 3

    def test_no_cache_works(self, sample_23andme_file: Path):
        llm = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])
        engine = DNAAnalysisEngine(snp_llm=llm, cache=None)

        result = engine.analyze("lactose", sample_23andme_file)
        assert result.cached is False

    def test_cache_invalidation_on_file_change(self, tmp_path: Path):
        f = tmp_path / "dna.txt"
        f.write_text("# 23andMe\nrs1\t1\t111\tAA\n")

        llm = FakeLLMProvider([_SNP_JSON, "interp1", _SNP_JSON, "interp2"])
        cache = InMemoryCache()
        engine = DNAAnalysisEngine(snp_llm=llm, cache=cache)

        r1 = engine.analyze("lactose", f)

        # Modify file -> hash changes -> cache miss
        f.write_text("# 23andMe\nrs1\t1\t111\tGG\n")
        r2 = engine.analyze("lactose", f)

        assert r1.cached is False
        assert r2.cached is False


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


class TestExtractJson:
    """The extract_json static method."""

    def test_plain_json(self):
        assert DNAAnalysisEngine.extract_json('{"key": "value"}') == '{"key": "value"}'

    def test_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        assert DNAAnalysisEngine.extract_json(text) == '{"key": "value"}'

    def test_markdown_without_lang(self):
        text = '```\n{"key": "value"}\n```'
        assert DNAAnalysisEngine.extract_json(text) == '{"key": "value"}'

    def test_leading_whitespace(self):
        assert DNAAnalysisEngine.extract_json('  {"key": "value"}  ') == '{"key": "value"}'

    def test_text_around_code_block(self):
        text = 'Here is the JSON:\n```json\n{"rs1": {}}\n```\nDone.'
        assert DNAAnalysisEngine.extract_json(text) == '{"rs1": {}}'


# ---------------------------------------------------------------------------
# Case-insensitive RSID matching
# ---------------------------------------------------------------------------


class TestCaseInsensitiveRSID:
    """RSID matching should be case-insensitive."""

    def test_uppercase_rsid_in_llm_response(self, tmp_path: Path):
        f = tmp_path / "dna.txt"
        f.write_text("# 23andMe\nrs1\t1\t111\tAA\n")

        # LLM returns uppercase RS1
        snp_json = json.dumps(
            {"RS1": {"gene": "G", "chromosome": "1", "position": 111, "trait": "T"}}
        )
        # But SNPDictResponse normalises to lowercase via validator
        llm = FakeLLMProvider([snp_json, "interpretation"])
        engine = DNAAnalysisEngine(snp_llm=llm, cache=None)

        result = engine.analyze("test", f)
        assert result.snp_count_matched == 1


# ---------------------------------------------------------------------------
# Filter and hash helpers
# ---------------------------------------------------------------------------


class TestFilterSNPs:
    """The _filter_snps static method."""

    def test_empty_snp_dict(self):
        df = pd.DataFrame({
            "RSID": ["rs1"], "CHROMOSOME": ["1"],
            "POSITION": [111], "GENOTYPE": ["AA"],
        })
        result = DNAAnalysisEngine._filter_snps(df, {})
        assert result.empty

    def test_no_matches(self):
        df = pd.DataFrame({
            "RSID": ["rs1"], "CHROMOSOME": ["1"],
            "POSITION": [111], "GENOTYPE": ["AA"],
        })
        snps = {"rs999": SNPMetadata(gene="G", chromosome="1", position=999, trait="T")}
        result = DNAAnalysisEngine._filter_snps(df, snps)
        assert result.empty


class TestHashFile:
    """The _hash_file static method."""

    def test_consistent_hash(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        h1 = DNAAnalysisEngine._hash_file(f)
        h2 = DNAAnalysisEngine._hash_file(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa")
        f2.write_text("bbb")
        assert DNAAnalysisEngine._hash_file(f1) != DNAAnalysisEngine._hash_file(f2)
