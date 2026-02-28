"""Tests for the SNP database client.

These tests mock HTTP calls to avoid hitting the real NCBI API.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from dna_rag.cache.memory import InMemoryCache
from dna_rag.snp_database import SNPDatabase, SNPValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_dbsnp_response(snp_id: str, *, maf: bool = False) -> dict[str, Any]:
    """Build a minimal dbSNP-like JSON response."""
    entry: dict[str, Any] = {
        "chr": "19",
        "chrpos": "44908684",
        "genes": [{"name": "APOE"}],
        "allele_origin": "C/T",
    }
    if maf:
        entry["global_mafs"] = [
            {"study": "GnomAD_genomes", "freq": "C=0.150000/12000"},
            {"study": "1000Genomes", "freq": "C=0.140000/700"},
        ]
    return {"result": {snp_id: entry}}


def _mock_clinvar_search_response(ids: list[str] | None = None) -> dict[str, Any]:
    """Build a minimal ClinVar esearch response."""
    if ids is None:
        ids = []
    return {"esearchresult": {"idlist": ids}}


def _mock_clinvar_summary_response(
    uid: str,
    significance: str = "Pathogenic",
    trait: str = "Alzheimer disease",
) -> dict[str, Any]:
    """Build a minimal ClinVar esummary response."""
    return {
        "result": {
            uid: {
                "germline_classification": {
                    "description": significance,
                    "trait_set": [{"trait_name": trait}] if trait else [],
                },
            }
        }
    }


def _urlopen_factory(data: dict[str, Any]):
    """Create a mock for urllib.request.urlopen."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = json.dumps(data).encode()
    return cm


# ---------------------------------------------------------------------------
# Tests: validate_rsid
# ---------------------------------------------------------------------------


def test_validate_rsid_found() -> None:
    db = SNPDatabase()
    dbsnp_data = _mock_dbsnp_response("429358")
    clinvar_search = _mock_clinvar_search_response()  # no ClinVar results

    responses = [
        _urlopen_factory(dbsnp_data),
        _urlopen_factory(clinvar_search),
    ]
    with patch("dna_rag.snp_database.urllib.request.urlopen", side_effect=responses):
        result = db.validate_rsid("rs429358")

    assert isinstance(result, SNPValidationResult)
    assert result.exists is True
    assert result.validated is True
    assert result.chromosome == "19"
    assert result.gene == "APOE"
    assert result.alleles == ["C", "T"]


def test_validate_rsid_not_found() -> None:
    db = SNPDatabase()
    data: dict[str, Any] = {"result": {}}
    with patch("dna_rag.snp_database.urllib.request.urlopen", return_value=_urlopen_factory(data)):
        result = db.validate_rsid("rs000000")

    assert result.exists is False
    assert result.validated is False


def test_validate_rsid_network_error() -> None:
    db = SNPDatabase()
    with patch(
        "dna_rag.snp_database.urllib.request.urlopen",
        side_effect=OSError("network error"),
    ):
        result = db.validate_rsid("rs429358")

    assert result.exists is False
    assert result.source == "dbsnp_error"


# ---------------------------------------------------------------------------
# Tests: MAF parsing
# ---------------------------------------------------------------------------


def test_maf_parsed_from_dbsnp() -> None:
    db = SNPDatabase()
    dbsnp_data = _mock_dbsnp_response("429358", maf=True)
    clinvar_search = _mock_clinvar_search_response()

    responses = [
        _urlopen_factory(dbsnp_data),
        _urlopen_factory(clinvar_search),
    ]
    with patch("dna_rag.snp_database.urllib.request.urlopen", side_effect=responses):
        result = db.validate_rsid("rs429358")

    assert result.maf == 0.15
    assert result.maf_allele == "C"
    assert result.maf_study == "GnomAD_genomes"


def test_maf_none_when_not_present() -> None:
    db = SNPDatabase()
    dbsnp_data = _mock_dbsnp_response("429358", maf=False)
    clinvar_search = _mock_clinvar_search_response()

    responses = [
        _urlopen_factory(dbsnp_data),
        _urlopen_factory(clinvar_search),
    ]
    with patch("dna_rag.snp_database.urllib.request.urlopen", side_effect=responses):
        result = db.validate_rsid("rs429358")

    assert result.maf is None
    assert result.maf_allele is None


# ---------------------------------------------------------------------------
# Tests: ClinVar integration
# ---------------------------------------------------------------------------


def test_clinvar_significance_fetched() -> None:
    db = SNPDatabase()
    dbsnp_data = _mock_dbsnp_response("429358")
    clinvar_search = _mock_clinvar_search_response(["17863"])
    clinvar_summary = _mock_clinvar_summary_response(
        "17863", significance="Pathogenic", trait="Alzheimer disease",
    )

    responses = [
        _urlopen_factory(dbsnp_data),
        _urlopen_factory(clinvar_search),
        _urlopen_factory(clinvar_summary),
    ]
    with patch("dna_rag.snp_database.urllib.request.urlopen", side_effect=responses):
        result = db.validate_rsid("rs429358")

    assert result.clinical_significance == "Pathogenic"
    assert result.clinvar_trait == "Alzheimer disease"


def test_clinvar_no_results() -> None:
    db = SNPDatabase()
    dbsnp_data = _mock_dbsnp_response("429358")
    clinvar_search = _mock_clinvar_search_response([])

    responses = [
        _urlopen_factory(dbsnp_data),
        _urlopen_factory(clinvar_search),
    ]
    with patch("dna_rag.snp_database.urllib.request.urlopen", side_effect=responses):
        result = db.validate_rsid("rs429358")

    assert result.clinical_significance is None
    assert result.clinvar_trait is None


def test_clinvar_failure_degrades_gracefully() -> None:
    """ClinVar failure should not prevent dbSNP validation."""
    db = SNPDatabase()
    dbsnp_data = _mock_dbsnp_response("429358")
    dbsnp_mock = _urlopen_factory(dbsnp_data)

    def side_effect(*args, **kwargs):
        # First call succeeds (dbSNP), second raises (ClinVar)
        side_effect.call_count += 1
        if side_effect.call_count == 1:
            return dbsnp_mock
        raise OSError("ClinVar unreachable")
    side_effect.call_count = 0

    with patch("dna_rag.snp_database.urllib.request.urlopen", side_effect=side_effect):
        result = db.validate_rsid("rs429358")

    assert result.exists is True
    assert result.gene == "APOE"
    assert result.clinical_significance is None


# ---------------------------------------------------------------------------
# Tests: caching
# ---------------------------------------------------------------------------


def test_caching_avoids_second_call() -> None:
    cache = InMemoryCache()
    db = SNPDatabase(cache=cache)
    dbsnp_data = _mock_dbsnp_response("429358")
    clinvar_search = _mock_clinvar_search_response()

    responses = [
        _urlopen_factory(dbsnp_data),
        _urlopen_factory(clinvar_search),
    ]

    target = "dna_rag.snp_database.urllib.request.urlopen"
    with patch(target, side_effect=responses) as mock_open:
        r1 = db.validate_rsid("rs429358")
        r2 = db.validate_rsid("rs429358")

    # urlopen called twice for first request (dbSNP + ClinVar), then cached
    assert mock_open.call_count == 2
    assert r1.exists is True
    assert r2.exists is True


# ---------------------------------------------------------------------------
# Tests: batch
# ---------------------------------------------------------------------------


def test_validate_batch() -> None:
    db = SNPDatabase(rate_limit_delay=0.0)  # no delay in tests

    responses = [
        _urlopen_factory(_mock_dbsnp_response("429358")),
        _urlopen_factory(_mock_clinvar_search_response()),
        _urlopen_factory(_mock_dbsnp_response("7412")),
        _urlopen_factory(_mock_clinvar_search_response()),
    ]

    target = "dna_rag.snp_database.urllib.request.urlopen"
    with patch(target, side_effect=responses):
        results = db.validate_batch(["rs429358", "rs7412"])

    assert len(results) == 2
    assert all(r.exists for r in results.values())
