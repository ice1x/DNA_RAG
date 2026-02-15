"""Tests for Pydantic data models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dna_rag.models import (
    AnalysisResult,
    SNPDictResponse,
    SNPMetadata,
    SNPResult,
)


class TestSNPMetadata:
    """Tests for SNPMetadata model."""

    def test_valid_construction(self):
        m = SNPMetadata(gene="LCT", chromosome="2", position=123, trait="Lactose")
        assert m.gene == "LCT"
        assert m.chromosome == "2"
        assert m.position == 123

    def test_chromosome_normalisation_numeric(self):
        m = SNPMetadata(gene="G", chromosome=" 1 ", position=0, trait="T")
        assert m.chromosome == "1"

    def test_chromosome_normalisation_m_to_mt(self):
        m = SNPMetadata(gene="G", chromosome="M", position=0, trait="T")
        assert m.chromosome == "MT"

    def test_chromosome_x_y_mt(self):
        for chrom in ("X", "Y", "MT"):
            m = SNPMetadata(gene="G", chromosome=chrom, position=0, trait="T")
            assert m.chromosome == chrom

    def test_invalid_chromosome(self):
        with pytest.raises(ValidationError):
            SNPMetadata(gene="G", chromosome="Z", position=0, trait="T")

    def test_invalid_chromosome_23(self):
        with pytest.raises(ValidationError):
            SNPMetadata(gene="G", chromosome="23", position=0, trait="T")

    def test_negative_position(self):
        with pytest.raises(ValidationError):
            SNPMetadata(gene="G", chromosome="1", position=-1, trait="T")

    def test_position_zero_valid(self):
        m = SNPMetadata(gene="G", chromosome="1", position=0, trait="T")
        assert m.position == 0

    def test_unicode_trait(self):
        m = SNPMetadata(
            gene="MT-ND1",
            chromosome="MT",
            position=3307,
            trait="Ashkenazi Jewish ancestry",
        )
        assert "Ashkenazi" in m.trait


class TestSNPDictResponse:
    """Tests for SNPDictResponse model."""

    def test_valid_snps(self):
        resp = SNPDictResponse(
            snps={
                "rs1": SNPMetadata(gene="G", chromosome="1", position=111, trait="T"),
            }
        )
        assert "rs1" in resp.snps

    def test_filters_non_rs_keys(self):
        resp = SNPDictResponse(
            snps={
                "rs1": SNPMetadata(gene="G", chromosome="1", position=111, trait="T"),
                "bad_key": SNPMetadata(gene="G", chromosome="1", position=112, trait="T"),
            }
        )
        assert "rs1" in resp.snps
        assert "bad_key" not in resp.snps

    def test_normalises_rsid_to_lowercase(self):
        resp = SNPDictResponse(
            snps={
                "RS123": SNPMetadata(gene="G", chromosome="1", position=111, trait="T"),
            }
        )
        assert "rs123" in resp.snps
        assert "RS123" not in resp.snps

    def test_empty_dict(self):
        resp = SNPDictResponse(snps={})
        assert len(resp.snps) == 0

    def test_default_empty(self):
        resp = SNPDictResponse()
        assert len(resp.snps) == 0


class TestSNPResult:
    """Tests for SNPResult model."""

    def test_valid_construction(self):
        r = SNPResult(
            rsid="rs1", chromosome="1", position=111, genotype="AA",
            gene="LCT", trait="Lactose",
        )
        assert r.rsid == "rs1"
        assert r.genotype == "AA"

    def test_defaults(self):
        r = SNPResult(rsid="rs1", chromosome="1", position=111, genotype="AA")
        assert r.gene == "unknown"
        assert r.trait == "unknown"


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_serialisation_roundtrip(self):
        result = AnalysisResult(
            question="lactose",
            matched_snps=[
                SNPResult(rsid="rs1", chromosome="1", position=111, genotype="AA"),
            ],
            interpretation="You are lactose tolerant.",
            snp_count_requested=5,
            snp_count_matched=1,
        )
        data = json.loads(result.model_dump_json())
        assert data["question"] == "lactose"
        assert data["cached"] is False
        assert len(data["matched_snps"]) == 1

    def test_cached_flag(self):
        result = AnalysisResult(
            question="q",
            matched_snps=[],
            interpretation="i",
            snp_count_requested=0,
            snp_count_matched=0,
            cached=True,
        )
        assert result.cached is True
