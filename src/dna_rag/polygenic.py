"""Polygenic risk score (PRS) calculation.

This module calculates PRS by aggregating weighted effects of multiple SNPs
from a user's genotype data.  In production, score definitions would be
loaded from validated PRS databases like `PGS Catalog <https://www.pgscatalog.org>`_.

The calculator operates on :class:`~pandas.DataFrame` objects produced by
the parser layer (columns: ``RSID``, ``CHROMOSOME``, ``POSITION``, ``GENOTYPE``).
"""

from __future__ import annotations

import logging

from pandas import DataFrame
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SNPWeight(BaseModel):
    """Weight and metadata for a single SNP in a polygenic score."""

    rsid: str
    risk_allele: str
    weight: float
    effect_size: float = Field(
        default=0.0, description="Effect size (e.g. odds ratio or beta)",
    )


class PolygenicScore(BaseModel):
    """Definition of a polygenic risk score."""

    name: str
    description: str
    snp_weights: list[SNPWeight]
    baseline_risk: float = Field(
        default=0.0,
        description="Population baseline risk or mean score",
    )


class PRSResult(BaseModel):
    """Result of a polygenic risk score calculation."""

    score_name: str
    raw_score: float
    normalized_score: float
    percentile: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentile relative to population",
    )
    snps_matched: int
    snps_total: int
    interpretation: str = ""


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class PolygenicScoreCalculator:
    """Calculator for polygenic risk scores.

    Scores can be registered via :meth:`register` or by passing them to
    the constructor.  The built-in example scores (Alzheimer's, T2 Diabetes)
    are loaded automatically if *load_examples* is ``True`` (default).

    Example::

        calc = PolygenicScoreCalculator()
        result = calc.calculate("alzheimers_risk", dna_df)
    """

    def __init__(
        self,
        scores: dict[str, PolygenicScore] | None = None,
        *,
        load_examples: bool = True,
    ) -> None:
        self._scores: dict[str, PolygenicScore] = scores or {}
        if load_examples and not self._scores:
            self._load_example_scores()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, score: PolygenicScore) -> None:
        """Register a new score definition."""
        self._scores[score.name] = score

    def list_available_scores(self) -> list[str]:
        """Return names of all registered scores."""
        return list(self._scores.keys())

    def get_score_info(self, score_name: str) -> PolygenicScore:
        """Return the definition for *score_name*.

        Raises:
            KeyError: Unknown score name.
        """
        if score_name not in self._scores:
            raise KeyError(f"Unknown score: {score_name}")
        return self._scores[score_name]

    def calculate(
        self,
        score_name: str,
        genotype_data: DataFrame,
    ) -> PRSResult:
        """Calculate a polygenic risk score.

        Args:
            score_name: Registered score name.
            genotype_data: DataFrame with ``RSID`` and ``GENOTYPE`` columns.

        Returns:
            Calculated :class:`PRSResult`.

        Raises:
            KeyError: Unknown *score_name*.
        """
        if score_name not in self._scores:
            raise KeyError(f"Unknown score: {score_name}")

        prs_def = self._scores[score_name]
        raw_score = 0.0
        snps_matched = 0

        for snp_weight in prs_def.snp_weights:
            matches = genotype_data[
                genotype_data["RSID"].str.lower() == snp_weight.rsid.lower()
            ]
            if matches.empty:
                continue

            genotype = str(matches.iloc[0]["GENOTYPE"])
            risk_allele_count = genotype.upper().count(snp_weight.risk_allele.upper())
            raw_score += risk_allele_count * snp_weight.weight
            snps_matched += 1

        total = len(prs_def.snp_weights)
        normalized_score = (
            (raw_score - prs_def.baseline_risk) / max(1.0, total)
        )
        percentile = _score_to_percentile(normalized_score)
        interpretation = _interpret_score(score_name, percentile)

        return PRSResult(
            score_name=score_name,
            raw_score=raw_score,
            normalized_score=normalized_score,
            percentile=percentile,
            snps_matched=snps_matched,
            snps_total=total,
            interpretation=interpretation,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_example_scores(self) -> None:
        """Seed a few well-known example scores for demos / tests."""
        self._scores["alzheimers_risk"] = PolygenicScore(
            name="alzheimers_risk",
            description="Polygenic risk score for Alzheimer's disease",
            snp_weights=[
                SNPWeight(rsid="rs429358", risk_allele="C", weight=2.5, effect_size=3.7),
                SNPWeight(rsid="rs7412", risk_allele="T", weight=-0.5, effect_size=0.6),
            ],
            baseline_risk=0.1,
        )
        self._scores["type2_diabetes_risk"] = PolygenicScore(
            name="type2_diabetes_risk",
            description="Polygenic risk score for Type 2 diabetes",
            snp_weights=[
                SNPWeight(rsid="rs7903146", risk_allele="T", weight=1.2, effect_size=1.37),
                SNPWeight(rsid="rs1801282", risk_allele="G", weight=0.8, effect_size=1.14),
            ],
            baseline_risk=0.08,
        )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _score_to_percentile(z: float) -> float:
    """Approximate percentile from a z-score (simplified)."""
    if z < -2:
        return 2.5
    if z < -1:
        return 16.0
    if z < 0:
        return 50.0 - abs(z) * 34.0
    if z < 1:
        return 50.0 + z * 34.0
    if z < 2:
        return 84.0
    return 97.5


def _interpret_score(score_name: str, percentile: float) -> str:
    """Build a human-readable disclaimer-style interpretation."""
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
