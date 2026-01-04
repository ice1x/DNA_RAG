"""Tests for VCF parser module."""

import gzip
from pathlib import Path

import pytest

from vcf_parser import VCFParser, convert_vcf_to_csv


@pytest.fixture
def sample_vcf_content():
    """Sample VCF content."""
    return """##fileformat=VCFv4.2
##contig=<ID=1>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1
1\t123456\trs123\tA\tG\t.\tPASS\t.\tGT\t0/1
2\t234567\trs456\tC\tT\t.\tPASS\t.\tGT\t1/1
3\t345678\t.\tG\tA\t.\tPASS\tRS=rs789\tGT\t0/0
"""


@pytest.fixture
def vcf_file(tmp_path, sample_vcf_content):
    """Create temporary VCF file."""
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text(sample_vcf_content)
    return vcf_path


@pytest.fixture
def gzipped_vcf_file(tmp_path, sample_vcf_content):
    """Create temporary gzipped VCF file."""
    vcf_gz_path = tmp_path / "test.vcf.gz"
    with gzip.open(vcf_gz_path, "wt", encoding="utf-8") as f:
        f.write(sample_vcf_content)
    return vcf_gz_path


def test_vcf_parser_init(vcf_file):
    """Test VCF parser initialization."""
    parser = VCFParser(vcf_file)
    assert parser.vcf_file == vcf_file


def test_vcf_parser_file_not_found(tmp_path):
    """Test parser with non-existent file."""
    with pytest.raises(FileNotFoundError):
        VCFParser(tmp_path / "nonexistent.vcf")


def test_parse_vcf(vcf_file):
    """Test parsing VCF file."""
    parser = VCFParser(vcf_file)
    df = parser.parse()

    assert len(df) >= 2  # At least rs123 and rs456
    assert "RSID" in df.columns
    assert "CHROMOSOME" in df.columns
    assert "POSITION" in df.columns
    assert "GENOTYPE" in df.columns


def test_parse_gzipped_vcf(gzipped_vcf_file):
    """Test parsing gzipped VCF file."""
    parser = VCFParser(gzipped_vcf_file)
    df = parser.parse()

    assert len(df) >= 2
    assert "rs123" in df["RSID"].values


def test_parse_variant_line():
    """Test parsing a single variant line."""
    parser = VCFParser.__new__(VCFParser)
    line = "1\t123456\trs123\tA\tG\t.\tPASS\t.\tGT\t0/1\n"

    record = parser._parse_variant_line(line)

    assert record is not None
    assert record["RSID"] == "rs123"
    assert record["CHROMOSOME"] == "1"
    assert record["POSITION"] == 123456
    assert record["GENOTYPE"] == "AG"


def test_parse_variant_without_rsid():
    """Test parsing variant without RS ID."""
    parser = VCFParser.__new__(VCFParser)
    line = "1\t123456\t.\tA\tG\t.\tPASS\tRS=rs999\tGT\t0/1\n"

    record = parser._parse_variant_line(line)

    assert record is not None
    assert record["RSID"] == "rs999"


def test_extract_genotype():
    """Test genotype extraction."""
    # Test phased genotype
    genotype = VCFParser._extract_genotype("GT", "0|1", "A", "G")
    assert genotype == "AG"

    # Test unphased genotype
    genotype = VCFParser._extract_genotype("GT", "0/1", "C", "T")
    assert genotype == "CT"

    # Test homozygous
    genotype = VCFParser._extract_genotype("GT", "1/1", "A", "G")
    assert genotype == "GG"


def test_extract_genotype_complex_format():
    """Test genotype extraction with complex FORMAT."""
    genotype = VCFParser._extract_genotype("GT:DP:GQ", "0/1:20:99", "A", "G")
    assert genotype == "AG"


def test_extract_rsid_from_info():
    """Test RS ID extraction from INFO field."""
    # Test RS= format
    rsid = VCFParser._extract_rsid_from_info("RS=rs123;AF=0.5")
    assert rsid == "rs123"

    # Test dbSNP= format
    rsid = VCFParser._extract_rsid_from_info("dbSNP=rs456;AF=0.3")
    assert rsid == "rs456"

    # Test no RS ID
    rsid = VCFParser._extract_rsid_from_info("AF=0.5")
    assert rsid is None


def test_convert_vcf_to_csv(vcf_file, tmp_path):
    """Test converting VCF to CSV."""
    output_csv = tmp_path / "output.csv"
    convert_vcf_to_csv(vcf_file, output_csv)

    assert output_csv.exists()
    content = output_csv.read_text()
    assert "rs123" in content or "rs456" in content


def test_parse_empty_vcf(tmp_path):
    """Test parsing empty VCF."""
    empty_vcf = tmp_path / "empty.vcf"
    empty_vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n")

    parser = VCFParser(empty_vcf)
    df = parser.parse()

    assert len(df) == 0
    assert "RSID" in df.columns


def test_parse_malformed_line(vcf_file):
    """Test parsing with malformed lines."""
    # Add malformed line
    with open(vcf_file, "a") as f:
        f.write("malformed line\n")

    parser = VCFParser(vcf_file)
    df = parser.parse()  # Should not crash

    assert len(df) >= 0
