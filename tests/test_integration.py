"""Integration tests for DNA_RAG enhanced features."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from chat_dna_enhanced import ChatDNAEnhanced, InterpretationResult
from polygenic_scores import PolygenicScoreCalculator


@pytest.fixture
def enhanced_chat(tmp_path):
    """Create enhanced ChatDNA instance."""
    return ChatDNAEnhanced(
        api_key="test",
        use_vector_store=True,
        use_validation=False,  # Disable to avoid real API calls
        vector_store_path=tmp_path / "vector_store",
    )


@pytest.fixture
def sample_dna_file_enhanced(tmp_path):
    """Create enhanced sample DNA file."""
    dna_path = tmp_path / "sample_enhanced.csv"
    dna_path.write_text(
        "rs4988235,2,136608646,CT\n"
        "rs1815739,11,66560624,CC\n"
        "rs7412,19,44908822,TT\n"
        "rs429358,19,44908684,CC\n"
    )
    return dna_path


def test_enhanced_chat_init(enhanced_chat):
    """Test enhanced chat initialization."""
    assert enhanced_chat._use_vector_store is True
    assert enhanced_chat._vector_store is not None
    assert enhanced_chat._vector_store.count() > 0


def test_ask_with_vector_store(enhanced_chat, sample_dna_file_enhanced):
    """Test asking question with vector store."""
    with patch.object(enhanced_chat, "llm") as mock_llm:
        mock_response = Mock()
        mock_response.content = (
            "INTERPRETATION: You have the lactose tolerance variant.\n"
            "CONFIDENCE: 0.8\n"
            "CAVEATS: Individual variation exists"
        )
        mock_llm.return_value = mock_response

        result = enhanced_chat.ask(
            "lactose tolerance", sample_dna_file_enhanced, return_structured=True
        )

        assert isinstance(result, InterpretationResult)
        assert result.question == "lactose tolerance"
        assert 0.0 <= result.confidence <= 1.0


def test_conversation_history(enhanced_chat, sample_dna_file_enhanced):
    """Test conversation history tracking."""
    with patch.object(enhanced_chat, "llm") as mock_llm:
        mock_response = Mock()
        mock_response.content = (
            "INTERPRETATION: Test response\nCONFIDENCE: 0.5\nCAVEATS: None"
        )
        mock_llm.return_value = mock_response

        enhanced_chat.ask("question 1", sample_dna_file_enhanced)
        enhanced_chat.ask("question 2", sample_dna_file_enhanced)

        history = enhanced_chat.get_conversation_history(sample_dna_file_enhanced)
        assert len(history) >= 4  # 2 questions + 2 answers


def test_structured_output(enhanced_chat, sample_dna_file_enhanced):
    """Test structured output format."""
    with patch.object(enhanced_chat, "llm") as mock_llm:
        mock_response = Mock()
        mock_response.content = (
            "INTERPRETATION: Structured test\n"
            "CONFIDENCE: 0.9\n"
            "CAVEATS: Caveat 1; Caveat 2"
        )
        mock_llm.return_value = mock_response

        result = enhanced_chat.ask(
            "test", sample_dna_file_enhanced, return_structured=True
        )

        assert isinstance(result, InterpretationResult)
        assert result.interpretation != ""
        assert len(result.caveats) > 0
        assert len(result.sources) > 0


def test_snp_validation_disabled(enhanced_chat, sample_dna_file_enhanced):
    """Test with SNP validation disabled."""
    with patch.object(enhanced_chat, "llm") as mock_llm:
        mock_response = Mock()
        mock_response.content = "INTERPRETATION: Test\nCONFIDENCE: 0.5\nCAVEATS: None"
        mock_llm.return_value = mock_response

        result = enhanced_chat.ask(
            "test", sample_dna_file_enhanced, return_structured=True
        )

        # Should work without validation
        assert isinstance(result, InterpretationResult)


def test_polygenic_score_integration(sample_dna_file_enhanced):
    """Test polygenic score calculation integration."""
    # Load DNA data
    df = pd.read_csv(
        sample_dna_file_enhanced,
        names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"],
    )

    # Calculate score
    calculator = PolygenicScoreCalculator()
    result = calculator.calculate("alzheimers_risk", df)

    assert result.score_name == "alzheimers_risk"
    assert result.snps_matched >= 0
    assert result.interpretation != ""


def test_caching(enhanced_chat, sample_dna_file_enhanced):
    """Test result caching."""
    with patch.object(enhanced_chat, "llm") as mock_llm:
        mock_response = Mock()
        mock_response.content = "INTERPRETATION: Cached\nCONFIDENCE: 0.7\nCAVEATS: Test"
        mock_llm.return_value = mock_response

        # First call
        result1 = enhanced_chat.ask("cache test", sample_dna_file_enhanced)
        call_count_1 = mock_llm.call_count

        # Second call (should be cached)
        result2 = enhanced_chat.ask("cache test", sample_dna_file_enhanced)
        call_count_2 = mock_llm.call_count

        # LLM should not be called again
        assert call_count_2 == call_count_1


def test_no_snps_found(enhanced_chat, sample_dna_file_enhanced):
    """Test behavior when no SNPs found."""
    with patch.object(enhanced_chat._vector_store, "search") as mock_search:
        mock_search.return_value = {}

        result = enhanced_chat.ask(
            "unknown trait", sample_dna_file_enhanced, return_structured=True
        )

        assert result.confidence == 0.0
        assert "No relevant SNPs found" in result.interpretation


def test_parse_interpretation():
    """Test interpretation parsing."""
    response = "INTERPRETATION: This is a test\nCONFIDENCE: 0.8\nCAVEATS: None"
    interp = ChatDNAEnhanced._parse_interpretation(response)
    assert interp == "This is a test"


def test_parse_confidence():
    """Test confidence parsing."""
    response = "INTERPRETATION: Test\nCONFIDENCE: 0.75\nCAVEATS: None"
    conf = ChatDNAEnhanced._parse_confidence(response)
    assert conf == 0.75

    # Test default value
    response_no_conf = "INTERPRETATION: Test"
    conf = ChatDNAEnhanced._parse_confidence(response_no_conf)
    assert conf == 0.5


def test_parse_caveats():
    """Test caveats parsing."""
    response = "INTERPRETATION: Test\nCONFIDENCE: 0.5\nCAVEATS: Caveat 1; Caveat 2; Caveat 3"
    caveats = ChatDNAEnhanced._parse_caveats(response)
    assert len(caveats) == 3
    assert "Caveat 1" in caveats
