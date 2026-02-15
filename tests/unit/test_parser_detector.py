"""Tests for format auto-detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from dna_rag.exceptions import UnsupportedFormatError
from dna_rag.parsers.detector import detect_and_parse


class TestAutoDetect:
    """Auto-detect and parse DNA files."""

    def test_detects_23andme(self, sample_23andme_file: Path):
        df = detect_and_parse(sample_23andme_file)
        assert len(df) == 4
        assert list(df.columns) == ["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"]

    def test_detects_myheritage(self, sample_myheritage_file: Path):
        df = detect_and_parse(sample_myheritage_file)
        assert len(df) == 3

    def test_detects_ancestrydna(self, sample_ancestrydna_file: Path):
        df = detect_and_parse(sample_ancestrydna_file)
        assert len(df) == 3

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            detect_and_parse(tmp_path / "nonexistent.txt")

    def test_unsupported_format_raises(self, tmp_path: Path):
        f = tmp_path / "garbage.txt"
        f.write_text("this is not a DNA file\nwith random content\n")
        with pytest.raises(UnsupportedFormatError):
            detect_and_parse(f)
