"""Tests for AncestryDNA parser."""

from __future__ import annotations

from pathlib import Path

from dna_rag.parsers.ancestrydna import AncestryDNAParser


class TestCanParse:
    """Format detection."""

    def test_detects_ancestrydna_by_header(self, sample_ancestrydna_file: Path):
        assert AncestryDNAParser.can_parse(sample_ancestrydna_file)

    def test_rejects_23andme(self, sample_23andme_file: Path):
        assert not AncestryDNAParser.can_parse(sample_23andme_file)

    def test_rejects_nonexistent(self, tmp_path: Path):
        assert not AncestryDNAParser.can_parse(tmp_path / "nope.txt")


class TestParse:
    """Parsing valid files."""

    def test_parses_valid_file(self, sample_ancestrydna_file: Path):
        df = AncestryDNAParser.parse(sample_ancestrydna_file)
        assert len(df) == 3
        assert list(df.columns) == ["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"]

    def test_merges_alleles(self, sample_ancestrydna_file: Path):
        df = AncestryDNAParser.parse(sample_ancestrydna_file)
        assert df.iloc[0]["GENOTYPE"] == "AA"
        assert df.iloc[2]["GENOTYPE"] == "CT"

    def test_rsid_normalised(self, sample_ancestrydna_file: Path):
        df = AncestryDNAParser.parse(sample_ancestrydna_file)
        assert df.iloc[0]["RSID"] == "rs1"

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("#AncestryDNA raw data download\n")
        df = AncestryDNAParser.parse(f)
        assert df.empty
