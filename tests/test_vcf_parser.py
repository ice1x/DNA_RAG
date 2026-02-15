"""Tests for the VCF parser."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from dna_rag.parsers.vcf import VCFParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VCF_HEADER = (
    "##fileformat=VCFv4.1\n"
    "##INFO=<ID=DP,Number=1,Type=Integer>\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
)

VCF_LINE_HET = "1\t12345\trs12345\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
VCF_LINE_HOM = "2\t67890\trs67890\tC\tT\t.\tPASS\t.\tGT\t1/1\n"
VCF_LINE_REF = "3\t11111\trs11111\tG\tA\t.\tPASS\t.\tGT\t0/0\n"
VCF_LINE_NO_RS = "4\t22222\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
VCF_LINE_INFO_RS = "5\t33333\t.\tT\tC\t.\tPASS\tRS=55555\tGT\t0/1\n"


@pytest.fixture()
def vcf_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.vcf"
    p.write_text(VCF_HEADER + VCF_LINE_HET + VCF_LINE_HOM + VCF_LINE_REF)
    return p


@pytest.fixture()
def vcf_gz_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.vcf.gz"
    content = (VCF_HEADER + VCF_LINE_HET + VCF_LINE_HOM).encode()
    with gzip.open(p, "wb") as f:
        f.write(content)
    return p


@pytest.fixture()
def vcf_with_info_rs(tmp_path: Path) -> Path:
    p = tmp_path / "info_rs.vcf"
    p.write_text(VCF_HEADER + VCF_LINE_INFO_RS)
    return p


@pytest.fixture()
def vcf_no_rs(tmp_path: Path) -> Path:
    p = tmp_path / "no_rs.vcf"
    p.write_text(VCF_HEADER + VCF_LINE_NO_RS)
    return p


@pytest.fixture()
def vcf_empty(tmp_path: Path) -> Path:
    p = tmp_path / "empty.vcf"
    p.write_text(VCF_HEADER)
    return p


# ---------------------------------------------------------------------------
# Tests: can_parse
# ---------------------------------------------------------------------------


def test_can_parse_vcf(vcf_file: Path) -> None:
    assert VCFParser.can_parse(vcf_file) is True


def test_can_parse_vcf_gz(vcf_gz_file: Path) -> None:
    assert VCFParser.can_parse(vcf_gz_file) is True


def test_can_parse_rejects_tsv(tmp_path: Path) -> None:
    p = tmp_path / "not_vcf.tsv"
    p.write_text("rs12345\t1\t12345\tAG\n")
    assert VCFParser.can_parse(p) is False


def test_can_parse_missing_file(tmp_path: Path) -> None:
    assert VCFParser.can_parse(tmp_path / "nope.vcf") is False


# ---------------------------------------------------------------------------
# Tests: parse
# ---------------------------------------------------------------------------


def test_parse_basic(vcf_file: Path) -> None:
    df = VCFParser.parse(vcf_file)
    assert list(df.columns) == ["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"]
    assert len(df) == 3
    row0 = df.iloc[0]
    assert row0["RSID"] == "rs12345"
    assert row0["CHROMOSOME"] == "1"
    assert row0["POSITION"] == 12345
    assert row0["GENOTYPE"] == "AG"


def test_parse_homozygous_alt(vcf_file: Path) -> None:
    df = VCFParser.parse(vcf_file)
    hom = df[df["RSID"] == "rs67890"].iloc[0]
    assert hom["GENOTYPE"] == "TT"


def test_parse_homozygous_ref(vcf_file: Path) -> None:
    df = VCFParser.parse(vcf_file)
    ref = df[df["RSID"] == "rs11111"].iloc[0]
    assert ref["GENOTYPE"] == "GG"


def test_parse_gzip(vcf_gz_file: Path) -> None:
    df = VCFParser.parse(vcf_gz_file)
    assert len(df) == 2


def test_parse_info_rsid(vcf_with_info_rs: Path) -> None:
    df = VCFParser.parse(vcf_with_info_rs)
    assert len(df) == 1
    assert df.iloc[0]["RSID"] == "rs55555"


def test_parse_no_rsid_skipped(vcf_no_rs: Path) -> None:
    df = VCFParser.parse(vcf_no_rs)
    assert len(df) == 0


def test_parse_empty_vcf(vcf_empty: Path) -> None:
    df = VCFParser.parse(vcf_empty)
    assert len(df) == 0
    assert list(df.columns) == ["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"]


def test_parse_phased_genotype(tmp_path: Path) -> None:
    """Phased VCFs use ``|`` instead of ``/``."""
    p = tmp_path / "phased.vcf"
    line = "1\t100\trs999\tA\tG\t.\tPASS\t.\tGT\t0|1\n"
    p.write_text(VCF_HEADER + line)
    df = VCFParser.parse(p)
    assert len(df) == 1
    assert df.iloc[0]["GENOTYPE"] == "AG"


def test_parse_multi_alt(tmp_path: Path) -> None:
    """Multi-allelic sites: ALT = G,T  and GT = 1/2."""
    p = tmp_path / "multi.vcf"
    line = "1\t200\trs888\tA\tG,T\t.\tPASS\t.\tGT\t1/2\n"
    p.write_text(VCF_HEADER + line)
    df = VCFParser.parse(p)
    assert len(df) == 1
    assert df.iloc[0]["GENOTYPE"] == "GT"


def test_detector_finds_vcf(vcf_file: Path) -> None:
    """VCFParser is registered in the auto-detector."""
    from dna_rag.parsers.detector import detect_and_parse

    df = detect_and_parse(vcf_file)
    assert len(df) == 3
