"""Tests for SNP database module."""

from unittest.mock import Mock, patch

import pytest
import requests

from dna_rag.utils.snp_database import SNPDatabase, SNPValidationResult


@pytest.fixture
def snp_db():
    """Create SNP database instance."""
    return SNPDatabase(cache_ttl=60, max_cache_size=100)


@pytest.fixture
def mock_dbsnp_response():
    """Mock dbSNP API response."""
    return {
        "result": {
            "123": {
                "chr": "1",
                "chrpos": "123456",
                "genes": [{"name": "GENE1"}],
                "allele_origin": "A/G",
            }
        }
    }


def test_snp_database_init(snp_db):
    """Test SNP database initialization."""
    assert snp_db._timeout == 10.0
    assert len(snp_db._cache) == 0


def test_validate_rsid_cache_hit(snp_db):
    """Test validation with cache hit."""
    # Populate cache
    cached_result = SNPValidationResult(rsid="rs123", exists=True, validated=True)
    snp_db._cache["rs123"] = cached_result

    result = snp_db.validate_rsid("rs123")
    assert result == cached_result
    assert result.rsid == "rs123"


@patch("snp_database.requests.Session.get")
def test_validate_rsid_success(mock_get, snp_db, mock_dbsnp_response):
    """Test successful RS ID validation."""
    mock_response = Mock()
    mock_response.json.return_value = mock_dbsnp_response
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    result = snp_db.validate_rsid("rs123")

    assert result.rsid == "rs123"
    assert result.exists is True
    assert result.validated is True
    assert result.chromosome == "1"
    assert result.position == 123456
    assert result.gene == "GENE1"


@patch("snp_database.requests.Session.get")
def test_validate_rsid_not_found(mock_get, snp_db):
    """Test validation for non-existent RS ID."""
    mock_response = Mock()
    mock_response.json.return_value = {"result": {}}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    result = snp_db.validate_rsid("rs999")

    assert result.rsid == "rs999"
    assert result.exists is False
    assert result.validated is False


@patch("snp_database.requests.Session.get")
def test_validate_rsid_network_error(mock_get, snp_db):
    """Test validation with network error."""
    mock_get.side_effect = requests.RequestException("Network error")

    result = snp_db.validate_rsid("rs123")

    assert result.rsid == "rs123"
    assert result.exists is False
    assert result.validated is False


def test_validate_batch(snp_db):
    """Test batch validation."""
    # Mock individual validation
    with patch.object(snp_db, "validate_rsid") as mock_validate:
        mock_validate.side_effect = [
            SNPValidationResult(rsid="rs1", exists=True),
            SNPValidationResult(rsid="rs2", exists=False),
        ]

        results = snp_db.validate_batch(["rs1", "rs2"])

        assert len(results) == 2
        assert results["rs1"].exists is True
        assert results["rs2"].exists is False


def test_extract_chromosome():
    """Test chromosome extraction."""
    snp_data = {"chr": "1"}
    chrom = SNPDatabase._extract_chromosome(snp_data)
    assert chrom == "1"

    snp_data = {}
    chrom = SNPDatabase._extract_chromosome(snp_data)
    assert chrom is None


def test_extract_position():
    """Test position extraction."""
    snp_data = {"chrpos": "123456"}
    pos = SNPDatabase._extract_position(snp_data)
    assert pos == 123456

    snp_data = {}
    pos = SNPDatabase._extract_position(snp_data)
    assert pos is None


def test_extract_gene():
    """Test gene extraction."""
    snp_data = {"genes": [{"name": "GENE1"}]}
    gene = SNPDatabase._extract_gene(snp_data)
    assert gene == "GENE1"

    snp_data = {"genes": []}
    gene = SNPDatabase._extract_gene(snp_data)
    assert gene is None


def test_extract_alleles():
    """Test allele extraction."""
    snp_data = {"allele_origin": "A/G"}
    alleles = SNPDatabase._extract_alleles(snp_data)
    assert alleles == ["A", "G"]

    snp_data = {}
    alleles = SNPDatabase._extract_alleles(snp_data)
    assert alleles == []
