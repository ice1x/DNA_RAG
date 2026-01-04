"""SNP database module for fetching and validating SNP information.

This module provides interfaces to various SNP databases including:
- NCBI dbSNP for SNP validation
- SNPedia-like cached data
- ClinVar for clinical significance
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from cachetools import TTLCache
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SNPValidationResult(BaseModel):
    """Result of SNP validation."""

    rsid: str
    exists: bool
    chromosome: str | None = None
    position: int | None = None
    gene: str | None = None
    clinical_significance: str | None = None
    alleles: list[str] = Field(default_factory=list)
    validated: bool = False
    source: str = "unknown"


class SNPDatabase:
    """Interface to SNP databases for validation and enrichment.

    This class provides methods to validate SNP identifiers and retrieve
    metadata from public databases. Results are cached to minimize API calls.
    """

    def __init__(
        self,
        cache_ttl: int = 3600,
        max_cache_size: int = 1000,
        request_timeout: float = 10.0,
    ) -> None:
        """Initialize SNP database interface.

        Parameters
        ----------
        cache_ttl : int
            Time-to-live for cached results in seconds.
        max_cache_size : int
            Maximum number of SNPs to cache.
        request_timeout : float
            Timeout for HTTP requests in seconds.
        """
        self._cache: TTLCache = TTLCache(maxsize=max_cache_size, ttl=cache_ttl)
        self._timeout = request_timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "DNA_RAG/1.0 (Educational research tool)",
            }
        )

    def validate_rsid(self, rsid: str) -> SNPValidationResult:
        """Validate an RSID and retrieve metadata.

        Parameters
        ----------
        rsid : str
            The RS identifier (e.g., "rs123").

        Returns
        -------
        SNPValidationResult
            Validation result with metadata.
        """
        if rsid in self._cache:
            logger.debug(f"Cache hit for {rsid}")
            return self._cache[rsid]

        # Try to fetch from NCBI dbSNP
        result = self._fetch_from_dbsnp(rsid)

        # Cache the result
        self._cache[rsid] = result
        return result

    def validate_batch(self, rsids: list[str]) -> dict[str, SNPValidationResult]:
        """Validate multiple RSIDs in batch.

        Parameters
        ----------
        rsids : List[str]
            List of RS identifiers.

        Returns
        -------
        Dict[str, SNPValidationResult]
            Mapping of RSID to validation result.
        """
        results = {}
        for rsid in rsids:
            results[rsid] = self.validate_rsid(rsid)
            # Rate limiting: 3 requests per second max
            time.sleep(0.34)
        return results

    def _fetch_from_dbsnp(self, rsid: str) -> SNPValidationResult:
        """Fetch SNP information from NCBI dbSNP.

        Uses the NCBI E-utilities API to retrieve SNP information.

        Parameters
        ----------
        rsid : str
            The RS identifier.

        Returns
        -------
        SNPValidationResult
            Validation result.
        """
        # Remove 'rs' prefix if present
        snp_id = rsid.replace("rs", "")

        # NCBI E-utilities API endpoint
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {"db": "snp", "id": snp_id, "retmode": "json"}

        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()

            # Check if SNP exists
            if "result" not in data or snp_id not in data["result"]:
                return SNPValidationResult(
                    rsid=rsid,
                    exists=False,
                    validated=False,
                    source="dbsnp",
                )

            snp_data = data["result"][snp_id]

            # Extract relevant fields
            chromosome = self._extract_chromosome(snp_data)
            position = self._extract_position(snp_data)
            gene = self._extract_gene(snp_data)
            alleles = self._extract_alleles(snp_data)

            return SNPValidationResult(
                rsid=rsid,
                exists=True,
                chromosome=chromosome,
                position=position,
                gene=gene,
                alleles=alleles,
                validated=True,
                source="dbsnp",
            )

        except requests.RequestException as exc:
            logger.warning(f"Failed to fetch {rsid} from dbSNP: {exc}")
            return SNPValidationResult(
                rsid=rsid,
                exists=False,
                validated=False,
                source="dbsnp_error",
            )
        except (KeyError, ValueError) as exc:
            logger.warning(f"Failed to parse dbSNP response for {rsid}: {exc}")
            return SNPValidationResult(
                rsid=rsid,
                exists=False,
                validated=False,
                source="dbsnp_parse_error",
            )

    @staticmethod
    def _extract_chromosome(snp_data: dict[str, Any]) -> str | None:
        """Extract chromosome from dbSNP data."""
        try:
            chr_data = snp_data.get("chr", "")
            return str(chr_data) if chr_data else None
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _extract_position(snp_data: dict[str, Any]) -> int | None:
        """Extract genomic position from dbSNP data."""
        try:
            # Try to get position from chrpos field
            chrpos = snp_data.get("chrpos", "")
            if chrpos:
                return int(chrpos)
            return None
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _extract_gene(snp_data: dict[str, Any]) -> str | None:
        """Extract gene name from dbSNP data."""
        try:
            genes = snp_data.get("genes", [])
            if genes and isinstance(genes, list) and len(genes) > 0:
                gene_info = genes[0]
                if isinstance(gene_info, dict):
                    return gene_info.get("name", None)
                return str(gene_info)
            return None
        except (KeyError, ValueError, IndexError):
            return None

    @staticmethod
    def _extract_alleles(snp_data: dict[str, Any]) -> list[str]:
        """Extract alleles from dbSNP data."""
        try:
            # Get allele string (e.g., "A/G")
            allele_str = snp_data.get("allele_origin", "")
            if not allele_str:
                # Try alternative field
                allele_str = snp_data.get("docsum", "")
            # Parse allele string
            if "/" in str(allele_str):
                return str(allele_str).split("/")
            return []
        except (KeyError, ValueError):
            return []

    def get_clinical_significance(self, rsid: str) -> str | None:
        """Get clinical significance from ClinVar.

        Parameters
        ----------
        rsid : str
            The RS identifier.

        Returns
        -------
        Optional[str]
            Clinical significance if available.
        """
        # This is a placeholder for ClinVar integration
        # Full implementation would query ClinVar API
        logger.debug(f"ClinVar lookup for {rsid} not yet implemented")
        return None
