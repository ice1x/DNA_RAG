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

def _mock_dbsnp_response(snp_id: str) -> dict[str, Any]:
    """Build a minimal dbSNP-like JSON response."""
    return {
        "result": {
            snp_id: {
                "chr": "19",
                "chrpos": "44908684",
                "genes": [{"name": "APOE"}],
                "allele_origin": "C/T",
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
    data = _mock_dbsnp_response("429358")
    with patch("dna_rag.snp_database.urllib.request.urlopen", return_value=_urlopen_factory(data)):
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
# Tests: caching
# ---------------------------------------------------------------------------


def test_caching_avoids_second_call() -> None:
    cache = InMemoryCache()
    db = SNPDatabase(cache=cache)
    data = _mock_dbsnp_response("429358")

    target = "dna_rag.snp_database.urllib.request.urlopen"
    with patch(target, return_value=_urlopen_factory(data)) as mock_open:
        r1 = db.validate_rsid("rs429358")
        r2 = db.validate_rsid("rs429358")

    # urlopen should only be called once
    assert mock_open.call_count == 1
    assert r1.exists is True
    assert r2.exists is True


# ---------------------------------------------------------------------------
# Tests: batch
# ---------------------------------------------------------------------------


def test_validate_batch() -> None:
    db = SNPDatabase(rate_limit_delay=0.0)  # no delay in tests

    responses = [
        _urlopen_factory(_mock_dbsnp_response("429358")),
        _urlopen_factory(_mock_dbsnp_response("7412")),
    ]

    target = "dna_rag.snp_database.urllib.request.urlopen"
    with patch(target, side_effect=responses):
        results = db.validate_batch(["rs429358", "rs7412"])

    assert len(results) == 2
    assert all(r.exists for r in results.values())
