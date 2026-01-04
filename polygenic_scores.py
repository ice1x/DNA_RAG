"""Polygenic risk score calculation module.

This module provides functionality to calculate polygenic risk scores (PRS)
from SNP data. PRS aggregate the effects of multiple genetic variants to
estimate risk for complex traits.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SNPWeight(BaseModel):
    """Weight and metadata for a single SNP in a polygenic score."""

    rsid: str
    risk_allele: str
    weight: float
    effect_size: float = Field(default=0.0, description="Effect size (e.g., odds ratio or beta)")


class PolygenicScore(BaseModel):
    """Definition of a polygenic score."""

    name: str
    description: str
    snp_weights: List[SNPWeight]
    baseline_risk: float = Field(
        default=0.0,
        description="Population baseline risk or mean score",
    )


class PRSResult(BaseModel):
    """Result of polygenic risk score calculation."""

    score_name: str
    raw_score: float
    normalized_score: float
    percentile: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentile relative to population",
    )
    snps_matched: int
    snps_total: int
    interpretation: str = ""


class PolygenicScoreCalculator:
    """Calculator for polygenic risk scores.

    This class computes PRS by summing weighted effects of SNPs found
    in a user's genotype data.
    """

    def __init__(self) -> None:
        """Initialize the PRS calculator."""
        self._scores: Dict[str, PolygenicScore] = {}
        self._load_example_scores()

    def _load_example_scores(self) -> None:
        """Load example polygenic scores for demonstration.

        In production, these would be loaded from validated PRS databases
        like PGS Catalog (https://www.pgscatalog.org).
        """
        # Example: Alzheimer's disease risk (simplified)
        alzheimers_prs = PolygenicScore(
            name="alzheimers_risk",
            description="Polygenic risk score for Alzheimer's disease",
            snp_weights=[
                SNPWeight(
                    rsid="rs429358",
                    risk_allele="C",
                    weight=2.5,
                    effect_size=3.7,
                ),
                SNPWeight(
                    rsid="rs7412",
                    risk_allele="T",
                    weight=-0.5,
                    effect_size=0.6,
                ),
            ],
            baseline_risk=0.1,
        )

        # Example: Type 2 diabetes risk (simplified)
        diabetes_prs = PolygenicScore(
            name="type2_diabetes_risk",
            description="Polygenic risk score for Type 2 diabetes",
            snp_weights=[
                SNPWeight(
                    rsid="rs7903146",
                    risk_allele="T",
                    weight=1.2,
                    effect_size=1.37,
                ),
                SNPWeight(
                    rsid="rs1801282",
                    risk_allele="G",
                    weight=0.8,
                    effect_size=1.14,
                ),
            ],
            baseline_risk=0.08,
        )

        self._scores["alzheimers_risk"] = alzheimers_prs
        self._scores["type2_diabetes_risk"] = diabetes_prs

    def calculate(self, score_name: str, genotype_data: pd.DataFrame) -> PRSResult:
        """Calculate a polygenic risk score.

        Parameters
        ----------
        score_name : str
            Name of the score to calculate.
        genotype_data : pd.DataFrame
            User's genotype data with columns: RSID, GENOTYPE.

        Returns
        -------
        PRSResult
            Calculated polygenic risk score result.
        """
        if score_name not in self._scores:
            raise ValueError(f"Unknown score: {score_name}")

        prs_def = self._scores[score_name]

        # Initialize score
        raw_score = 0.0
        snps_matched = 0

        # Calculate weighted sum
        for snp_weight in prs_def.snp_weights:
            rsid = snp_weight.rsid
            risk_allele = snp_weight.risk_allele
            weight = snp_weight.weight

            # Find SNP in user's data
            matches = genotype_data[genotype_data["RSID"] == rsid]

            if matches.empty:
                continue

            genotype = str(matches.iloc[0]["GENOTYPE"])

            # Count risk alleles (0, 1, or 2)
            risk_allele_count = genotype.count(risk_allele)

            # Add to score
            raw_score += risk_allele_count * weight
            snps_matched += 1

        # Normalize score (mean=0, std=1 in population)
        # This is a simplified normalization
        normalized_score = (raw_score - prs_def.baseline_risk) / max(1.0, len(prs_def.snp_weights))

        # Estimate percentile (simplified - assumes normal distribution)
        percentile = self._score_to_percentile(normalized_score)

        # Generate interpretation
        interpretation = self._interpret_score(score_name, normalized_score, percentile)

        return PRSResult(
            score_name=score_name,
            raw_score=raw_score,
            normalized_score=normalized_score,
            percentile=percentile,
            snps_matched=snps_matched,
            snps_total=len(prs_def.snp_weights),
            interpretation=interpretation,
        )

    @staticmethod
    def _score_to_percentile(normalized_score: float) -> float:
        """Convert normalized score to percentile.

        Uses a simplified normal distribution assumption.

        Parameters
        ----------
        normalized_score : float
            Normalized score (z-score).

        Returns
        -------
        float
            Percentile (0-100).
        """
        # Simplified: assume mean=0, std=1
        # This is a rough approximation
        if normalized_score < -2:
            return 2.5
        elif normalized_score < -1:
            return 16.0
        elif normalized_score < 0:
            return 50.0 - abs(normalized_score) * 34.0
        elif normalized_score < 1:
            return 50.0 + normalized_score * 34.0
        elif normalized_score < 2:
            return 84.0
        else:
            return 97.5

    def _interpret_score(self, score_name: str, normalized_score: float, percentile: float) -> str:
        """Generate interpretation for a score.

        Parameters
        ----------
        score_name : str
            Name of the score.
        normalized_score : float
            Normalized score.
        percentile : float
            Percentile.

        Returns
        -------
        str
            Human-readable interpretation.
        """
        if percentile < 25:
            risk_level = "lower than average"
        elif percentile < 75:
            risk_level = "average"
        else:
            risk_level = "higher than average"

        trait = score_name.replace("_", " ").title()

        return (
            f"Your {trait} polygenic score is {risk_level} "
            f"(percentile: {percentile:.1f}). This score is based on "
            "a limited set of genetic variants and should not be used "
            "for medical decisions. Many factors beyond genetics influence "
            "disease risk."
        )

    def list_available_scores(self) -> List[str]:
        """List available polygenic scores.

        Returns
        -------
        List[str]
            List of score names.
        """
        return list(self._scores.keys())

    def get_score_info(self, score_name: str) -> PolygenicScore:
        """Get information about a score.

        Parameters
        ----------
        score_name : str
            Name of the score.

        Returns
        -------
        PolygenicScore
            Score definition.
        """
        if score_name not in self._scores:
            raise ValueError(f"Unknown score: {score_name}")
        return self._scores[score_name]
