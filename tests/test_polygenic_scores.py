"""Tests for polygenic scores module."""

import pandas as pd
import pytest

from polygenic_scores import PolygenicScore, PolygenicScoreCalculator, SNPWeight


@pytest.fixture
def calculator():
    """Create polygenic score calculator."""
    return PolygenicScoreCalculator()


@pytest.fixture
def sample_genotype_data():
    """Create sample genotype data."""
    return pd.DataFrame(
        {
            "RSID": ["rs429358", "rs7412", "rs7903146", "rs1801282"],
            "CHROMOSOME": ["19", "19", "10", "3"],
            "POSITION": [44908684, 44908822, 114758349, 12368125],
            "GENOTYPE": ["CC", "TT", "CT", "CG"],
        }
    )


def test_calculator_init(calculator):
    """Test calculator initialization."""
    assert len(calculator._scores) > 0
    scores = calculator.list_available_scores()
    assert "alzheimers_risk" in scores
    assert "type2_diabetes_risk" in scores


def test_calculate_alzheimers(calculator, sample_genotype_data):
    """Test Alzheimer's risk calculation."""
    result = calculator.calculate("alzheimers_risk", sample_genotype_data)

    assert result.score_name == "alzheimers_risk"
    assert result.snps_matched >= 0
    assert result.snps_total == 2
    assert result.raw_score >= 0.0  # Raw score should be non-negative
    assert 0.0 <= result.percentile <= 100.0


def test_calculate_diabetes(calculator, sample_genotype_data):
    """Test Type 2 diabetes risk calculation."""
    result = calculator.calculate("type2_diabetes_risk", sample_genotype_data)

    assert result.score_name == "type2_diabetes_risk"
    assert result.snps_matched >= 0
    assert result.snps_total == 2
    assert result.interpretation != ""


def test_calculate_unknown_score(calculator, sample_genotype_data):
    """Test calculation with unknown score."""
    with pytest.raises(ValueError, match="Unknown score"):
        calculator.calculate("unknown_score", sample_genotype_data)


def test_calculate_empty_genotype(calculator):
    """Test calculation with empty genotype data."""
    empty_df = pd.DataFrame(columns=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])
    result = calculator.calculate("alzheimers_risk", empty_df)

    assert result.snps_matched == 0
    assert result.raw_score == 0.0


def test_score_to_percentile():
    """Test score to percentile conversion."""
    calc = PolygenicScoreCalculator()

    # Test various z-scores
    assert calc._score_to_percentile(-3.0) == 2.5
    assert calc._score_to_percentile(-1.0) == 16.0
    assert calc._score_to_percentile(0.0) == 50.0  # Median percentile
    assert calc._score_to_percentile(1.0) == 84.0
    assert calc._score_to_percentile(3.0) == 97.5


def test_interpret_score(calculator):
    """Test score interpretation."""
    interp = calculator._interpret_score("alzheimers_risk", 0.0, 50.0)
    assert "average" in interp.lower()

    interp = calculator._interpret_score("alzheimers_risk", 2.0, 85.0)
    assert "higher" in interp.lower()

    interp = calculator._interpret_score("alzheimers_risk", -2.0, 15.0)
    assert "lower" in interp.lower()


def test_list_available_scores(calculator):
    """Test listing available scores."""
    scores = calculator.list_available_scores()
    assert isinstance(scores, list)
    assert len(scores) >= 2


def test_get_score_info(calculator):
    """Test getting score info."""
    info = calculator.get_score_info("alzheimers_risk")
    assert isinstance(info, PolygenicScore)
    assert info.name == "alzheimers_risk"
    assert len(info.snp_weights) > 0


def test_get_score_info_unknown(calculator):
    """Test getting info for unknown score."""
    with pytest.raises(ValueError, match="Unknown score"):
        calculator.get_score_info("unknown_score")


def test_snp_weight_model():
    """Test SNP weight model."""
    weight = SNPWeight(
        rsid="rs123",
        risk_allele="A",
        weight=1.5,
        effect_size=1.3,
    )
    assert weight.rsid == "rs123"
    assert weight.risk_allele == "A"
    assert weight.weight == 1.5


def test_polygenic_score_model():
    """Test polygenic score model."""
    score = PolygenicScore(
        name="test_score",
        description="Test score",
        snp_weights=[
            SNPWeight(rsid="rs1", risk_allele="A", weight=1.0),
        ],
        baseline_risk=0.1,
    )
    assert score.name == "test_score"
    assert len(score.snp_weights) == 1
