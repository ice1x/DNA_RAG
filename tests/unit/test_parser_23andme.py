"""Tests for 23andMe parser."""

from __future__ import annotations

from pathlib import Path

from dna_rag.parsers.twentythreeandme import TwentyThreeAndMeParser


class TestCanParse:
    """Format detection."""

    def test_detects_23andme_by_header(self, sample_23andme_file: Path):
        assert TwentyThreeAndMeParser.can_parse(sample_23andme_file)

    def test_rejects_myheritage(self, sample_myheritage_file: Path):
        assert not TwentyThreeAndMeParser.can_parse(sample_myheritage_file)

    def test_rejects_nonexistent(self, tmp_path: Path):
        assert not TwentyThreeAndMeParser.can_parse(tmp_path / "nope.txt")

    def test_detects_by_tab_format(self, tmp_path: Path):
        """File without 23andMe header but tab-separated rs-data."""
        f = tmp_path / "plain.txt"
        f.write_text("rs100\t1\t500\tAA\n")
        assert TwentyThreeAndMeParser.can_parse(f)


class TestParse:
    """Parsing valid and invalid files."""

    def test_parses_valid_file(self, sample_23andme_file: Path):
        df = TwentyThreeAndMeParser.parse(sample_23andme_file)
        assert len(df) == 4
        assert list(df.columns) == ["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"]
        # RSID should be lowercase
        assert df.iloc[0]["RSID"] == "rs1"
        # CHROMOSOME should be uppercase
        assert df.iloc[3]["CHROMOSOME"] == "X"
        # POSITION should be int
        assert df.iloc[0]["POSITION"] == 111

    def test_empty_file_after_comments(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("# only comments\n# nothing else\n")
        df = TwentyThreeAndMeParser.parse(f)
        assert df.empty

    def test_rsid_normalised_lowercase(self, tmp_path: Path):
        f = tmp_path / "upper.txt"
        f.write_text("RS100\t1\t500\tAA\n")
        df = TwentyThreeAndMeParser.parse(f)
        assert df.iloc[0]["RSID"] == "rs100"

    def test_genotype_normalised_uppercase(self, tmp_path: Path):
        f = tmp_path / "lower.txt"
        f.write_text("rs100\t1\t500\tag\n")
        df = TwentyThreeAndMeParser.parse(f)
        assert df.iloc[0]["GENOTYPE"] == "AG"
