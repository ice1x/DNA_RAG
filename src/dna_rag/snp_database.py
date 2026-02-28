"""SNP database client for validating RS identifiers against NCBI dbSNP.

This module provides an :class:`SNPDatabase` that fetches metadata from the
`NCBI E-utilities <https://www.ncbi.nlm.nih.gov/books/NBK25501/>`_ API and
caches results in-memory using :class:`~dna_rag.cache.base.Cache`.

The module has **no heavy dependencies** -- it uses the stdlib
:mod:`urllib.request` for HTTP so it works without ``requests``.

Example::

    from dna_rag.snp_database import SNPDatabase
    db = SNPDatabase()
    result = db.validate_rsid("rs429358")
    assert result.exists
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel, Field

from dna_rag.cache.base import Cache

logger = logging.getLogger(__name__)

_DBSNP_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_NS_SNP_DB = "snp_db"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SNPValidationResult(BaseModel):
    """Result of validating a single RSID against dbSNP and optionally ClinVar."""

    rsid: str
    exists: bool
    chromosome: str | None = None
    position: int | None = None
    gene: str | None = None
    alleles: list[str] = Field(default_factory=list)
    maf: float | None = None
    maf_allele: str | None = None
    maf_study: str | None = None
    clinical_significance: str | None = None
    clinvar_trait: str | None = None
    validated: bool = False
    source: str = "unknown"


# ---------------------------------------------------------------------------
# Database client
# ---------------------------------------------------------------------------


class SNPDatabase:
    """Lightweight client for NCBI dbSNP validation.

    Args:
        cache: Optional cache backend.  Validated results are stored under
            the ``snp_db`` namespace.
        request_timeout: HTTP timeout in seconds.
        rate_limit_delay: Seconds to wait between batch requests
            (NCBI recommends max 3 req/s).
    """

    def __init__(
        self,
        cache: Cache | None = None,
        request_timeout: float = 10.0,
        rate_limit_delay: float = 0.34,
    ) -> None:
        self._cache = cache
        self._timeout = request_timeout
        self._rate_limit_delay = rate_limit_delay

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def validate_rsid(self, rsid: str) -> SNPValidationResult:
        """Validate *rsid* and return metadata from dbSNP + ClinVar.

        Results are cached if a :class:`~dna_rag.cache.base.Cache` was provided.
        """
        if self._cache is not None:
            cached: SNPValidationResult | None = self._cache.get(
                _NS_SNP_DB, rsid,
            )
            if cached is not None:
                return cached

        result = self._fetch_from_dbsnp(rsid)

        # Enrich with ClinVar data if the SNP exists
        if result.exists:
            clinvar = self._fetch_clinvar(rsid)
            if clinvar:
                result.clinical_significance = clinvar.get("significance")
                result.clinvar_trait = clinvar.get("trait")

        if self._cache is not None:
            self._cache.set(_NS_SNP_DB, rsid, result)

        return result

    def validate_batch(
        self, rsids: list[str],
    ) -> dict[str, SNPValidationResult]:
        """Validate multiple RSIDs, rate-limiting between calls."""
        results: dict[str, SNPValidationResult] = {}
        for rsid in rsids:
            results[rsid] = self.validate_rsid(rsid)
            time.sleep(self._rate_limit_delay)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_from_dbsnp(self, rsid: str) -> SNPValidationResult:
        """Fetch SNP info from NCBI E-utilities (stdlib only)."""
        snp_id = rsid.lower().replace("rs", "")
        url = f"{_DBSNP_URL}?db=snp&id={snp_id}&retmode=json"

        try:
            req = urllib.request.Request(  # noqa: S310
                url,
                headers={"User-Agent": "DNA_RAG/1.0 (Educational research tool)"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                data: dict[str, Any] = json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning("dbSNP fetch failed for %s: %s", rsid, exc)
            return SNPValidationResult(
                rsid=rsid, exists=False, validated=False, source="dbsnp_error",
            )

        if "result" not in data or snp_id not in data["result"]:
            return SNPValidationResult(
                rsid=rsid, exists=False, validated=False, source="dbsnp",
            )

        snp_data = data["result"][snp_id]
        maf_allele, maf_freq, maf_study = _extract_maf(snp_data)
        return SNPValidationResult(
            rsid=rsid,
            exists=True,
            chromosome=_extract_str(snp_data, "chr"),
            position=_extract_int(snp_data, "chrpos"),
            gene=_extract_gene(snp_data),
            alleles=_extract_alleles(snp_data),
            maf=maf_freq,
            maf_allele=maf_allele,
            maf_study=maf_study,
            validated=True,
            source="dbsnp",
        )


    def _fetch_clinvar(self, rsid: str) -> dict[str, str] | None:
        """Fetch clinical significance from NCBI ClinVar for *rsid*.

        Uses a two-step approach: esearch to find ClinVar IDs, then
        esummary to get clinical classification.  Returns ``None`` on
        any failure (graceful degradation).
        """
        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=clinvar&term={rsid}&retmode=json"
        )
        try:
            req = urllib.request.Request(
                search_url,
                headers={"User-Agent": "DNA_RAG/1.0 (Educational research tool)"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                search_data: dict[str, Any] = json.loads(resp.read().decode())

            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return None

            # Fetch summary for the first result
            summary_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=clinvar&id={id_list[0]}&retmode=json"
            )
            req2 = urllib.request.Request(
                summary_url,
                headers={"User-Agent": "DNA_RAG/1.0 (Educational research tool)"},
            )
            with urllib.request.urlopen(req2, timeout=self._timeout) as resp2:  # noqa: S310
                summary_data: dict[str, Any] = json.loads(resp2.read().decode())

            entry = summary_data.get("result", {}).get(id_list[0], {})

            # Extract germline classification (most common for consumer SNPs)
            germline = entry.get("germline_classification", {})
            significance = germline.get("description", "")
            trait_set = germline.get("trait_set", [])
            trait = trait_set[0].get("trait_name", "") if trait_set else ""

            if significance:
                return {"significance": significance, "trait": trait}
        except (urllib.error.URLError, json.JSONDecodeError, OSError, KeyError, IndexError) as exc:
            logger.warning("ClinVar fetch failed for %s: %s", rsid, exc)

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_str(data: dict[str, Any], key: str) -> str | None:
    val = data.get(key, "")
    return str(val) if val else None


def _extract_int(data: dict[str, Any], key: str) -> int | None:
    val = data.get(key, "")
    try:
        return int(val) if val else None
    except (ValueError, TypeError):
        return None


def _extract_gene(data: dict[str, Any]) -> str | None:
    genes = data.get("genes", [])
    if genes and isinstance(genes, list):
        first = genes[0]
        if isinstance(first, dict):
            return first.get("name")
        return str(first)
    return None


def _extract_alleles(data: dict[str, Any]) -> list[str]:
    for key in ("allele_origin", "docsum"):
        val = str(data.get(key, ""))
        if "/" in val:
            return val.split("/")
    return []


def _extract_maf(
    data: dict[str, Any],
) -> tuple[str | None, float | None, str | None]:
    """Extract minor allele frequency from dbSNP ``global_mafs``.

    Prefers GnomAD, then 1000Genomes, then falls back to the first entry.
    Returns ``(allele, frequency, study)`` or ``(None, None, None)``.
    """
    mafs = data.get("global_mafs", [])
    if not mafs or not isinstance(mafs, list):
        return None, None, None

    # Preferred studies in priority order
    preferred = ("GnomAD_genomes", "GnomAD", "1000Genomes", "1000Genomes_30X")
    chosen = None
    for study_name in preferred:
        for entry in mafs:
            if entry.get("study") == study_name:
                chosen = entry
                break
        if chosen:
            break
    if chosen is None:
        chosen = mafs[0]

    freq_str = chosen.get("freq", "")
    study = chosen.get("study")
    # Format: "T=0.082867/415" -> allele="T", freq=0.082867
    try:
        allele_part, _ = freq_str.split("/", 1)
        allele, freq_val = allele_part.split("=", 1)
        return allele, float(freq_val), study
    except (ValueError, AttributeError):
        return None, None, None
