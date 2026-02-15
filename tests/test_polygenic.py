"""Tests for the polygenic risk score calculator."""

from __future__ import annotations

import pandas as pd
import pytest

from dna_rag.polygenic import (
    PolygenicScore,
    PolygenicScoreCalculator,
    PRSResult,
    SNPWeight,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def calculator() -> PolygenicScoreCalculator:
    return PolygenicScoreCalculator(load_examples=True)


@pytest.fixture()
def empty_calculator() -> PolygenicScoreCalculator:
    return PolygenicScoreCalculator(load_examples=False)


@pytest.fixture()
def genotype_df() -> pd.DataFrame:
    """DataFrame containing known Alzheimer's-related SNPs."""
    return pd.DataFrame(
        {
            "RSID": ["rs429358", "rs7412", "rs9999"],
            "CHROMOSOME": ["19", "19", "1"],
            "POSITION": [44908684, 44908822, 100],
            "GENOTYPE": ["CC", "CT", "AA"],
        }
    )


# ---------------------------------------------------------------------------
# Tests: init & registration
# ---------------------------------------------------------------------------


def test_example_scores_loaded(calculator: PolygenicScoreCalculator) -> None:
    available = calculator.list_available_scores()
    assert "alzheimers_risk" in available
    assert "type2_diabetes_risk" in available


def test_empty_calculator_has_no_scores(
    empty_calculator: PolygenicScoreCalculator,
) -> None:
    assert empty_calculator.list_available_scores() == []


def test_register_custom_score(
    empty_calculator: PolygenicScoreCalculator,
) -> None:
    score = PolygenicScore(
        name="test_score",
        description="A test score",
        snp_weights=[
            SNPWeight(rsid="rs111", risk_allele="A", weight=1.0),
        ],
    )
    empty_calculator.register(score)
    assert "test_score" in empty_calculator.list_available_scores()


def test_get_score_info(calculator: PolygenicScoreCalculator) -> None:
    info = calculator.get_score_info("alzheimers_risk")
    assert info.name == "alzheimers_risk"
    assert len(info.snp_weights) == 2


def test_get_score_info_unknown_raises(
    calculator: PolygenicScoreCalculator,
) -> None:
    with pytest.raises(KeyError, match="Unknown score"):
        calculator.get_score_info("no_such_score")


# ---------------------------------------------------------------------------
# Tests: calculate
# ---------------------------------------------------------------------------


def test_calculate_alzheimers(
    calculator: PolygenicScoreCalculator, genotype_df: pd.DataFrame,
) -> None:
    result = calculator.calculate("alzheimers_risk", genotype_df)
    assert isinstance(result, PRSResult)
    assert result.score_name == "alzheimers_risk"
    assert result.snps_matched == 2
    assert result.snps_total == 2
    # rs429358 CC = 2 risk alleles * 2.5 = 5.0
    # rs7412 CT = 1 risk allele * -0.5 = -0.5
    assert result.raw_score == pytest.approx(4.5)


def test_calculate_no_matching_snps(
    calculator: PolygenicScoreCalculator,
) -> None:
    empty_df = pd.DataFrame(
        {"RSID": ["rs99999"], "CHROMOSOME": ["1"], "POSITION": [1], "GENOTYPE": ["AA"]}
    )
    result = calculator.calculate("alzheimers_risk", empty_df)
    assert result.snps_matched == 0
    assert result.raw_score == 0.0


def test_calculate_unknown_score_raises(
    calculator: PolygenicScoreCalculator, genotype_df: pd.DataFrame,
) -> None:
    with pytest.raises(KeyError, match="Unknown score"):
        calculator.calculate("nonexistent", genotype_df)


def test_result_has_interpretation(
    calculator: PolygenicScoreCalculator, genotype_df: pd.DataFrame,
) -> None:
    result = calculator.calculate("alzheimers_risk", genotype_df)
    assert "polygenic score" in result.interpretation.lower()
    assert "medical decisions" in result.interpretation.lower()


def test_percentile_in_range(
    calculator: PolygenicScoreCalculator, genotype_df: pd.DataFrame,
) -> None:
    result = calculator.calculate("alzheimers_risk", genotype_df)
    assert 0.0 <= result.percentile <= 100.0


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


def test_case_insensitive_rsid_matching(
    calculator: PolygenicScoreCalculator,
) -> None:
    """RSIDs in the genotype data may be lowercase."""
    df = pd.DataFrame(
        {
            "RSID": ["RS429358"],  # uppercase
            "CHROMOSOME": ["19"],
            "POSITION": [44908684],
            "GENOTYPE": ["CC"],
        }
    )
    # Our calculator lowercases both sides, so this should match
    result = calculator.calculate("alzheimers_risk", df)
    assert result.snps_matched == 1
