"""Tests for MyHeritage parser."""

from __future__ import annotations

from pathlib import Path

from dna_rag.parsers.myheritage import MyHeritageParser


class TestCanParse:
    """Format detection."""

    def test_detects_myheritage_by_header(self, sample_myheritage_file: Path):
        assert MyHeritageParser.can_parse(sample_myheritage_file)

    def test_rejects_23andme(self, sample_23andme_file: Path):
        assert not MyHeritageParser.can_parse(sample_23andme_file)

    def test_rejects_nonexistent(self, tmp_path: Path):
        assert not MyHeritageParser.can_parse(tmp_path / "nope.csv")


class TestParse:
    """Parsing valid files."""

    def test_parses_valid_file(self, sample_myheritage_file: Path):
        df = MyHeritageParser.parse(sample_myheritage_file)
        assert len(df) == 3
        assert list(df.columns) == ["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"]

    def test_result_renamed_to_genotype(self, sample_myheritage_file: Path):
        df = MyHeritageParser.parse(sample_myheritage_file)
        assert "GENOTYPE" in df.columns
        assert "RESULT" not in df.columns

    def test_rsid_normalised(self, sample_myheritage_file: Path):
        df = MyHeritageParser.parse(sample_myheritage_file)
        assert df.iloc[0]["RSID"] == "rs1"

    def test_headerless_csv(self, tmp_path: Path):
        """MyHeritage CSV without explicit header row."""
        f = tmp_path / "raw.csv"
        f.write_text("rs10,1,500,AG\nrs20,2,600,CC\n")
        df = MyHeritageParser.parse(f)
        assert len(df) == 2

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.csv"
        f.write_text("# MyHeritage DNA raw data\nRSID,CHROMOSOME,POSITION,RESULT\n")
        df = MyHeritageParser.parse(f)
        assert df.empty
