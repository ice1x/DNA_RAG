"""Tests for the exception hierarchy."""

from dna_rag.exceptions import (
    AnalysisError,
    CacheError,
    ConfigurationError,
    DNARagError,
    InvalidDNAFileError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    NoMatchingVariantsError,
    NoSNPsFoundError,
    ParsingError,
    UnsupportedFormatError,
)


class TestExceptionHierarchy:
    """All custom exceptions inherit from DNARagError."""

    def test_llm_errors_inherit_from_llm_error(self):
        for exc_cls in (LLMConnectionError, LLMResponseError, LLMRateLimitError):
            assert issubclass(exc_cls, LLMError)
            assert issubclass(exc_cls, DNARagError)

    def test_parsing_errors_inherit_from_parsing_error(self):
        for exc_cls in (UnsupportedFormatError, InvalidDNAFileError):
            assert issubclass(exc_cls, ParsingError)
            assert issubclass(exc_cls, DNARagError)

    def test_analysis_errors_inherit_from_analysis_error(self):
        for exc_cls in (NoSNPsFoundError, NoMatchingVariantsError):
            assert issubclass(exc_cls, AnalysisError)
            assert issubclass(exc_cls, DNARagError)

    def test_config_and_cache_errors(self):
        assert issubclass(ConfigurationError, DNARagError)
        assert issubclass(CacheError, DNARagError)

    def test_exception_message_preserved(self):
        msg = "something went wrong"
        exc = LLMResponseError(msg)
        assert str(exc) == msg

    def test_exception_chaining(self):
        original = ValueError("bad json")
        try:
            raise LLMResponseError("parse failed") from original
        except LLMResponseError as chained:
            assert chained.__cause__ is original

    def test_can_catch_broadly(self):
        try:
            raise NoSNPsFoundError("no snps")
        except DNARagError as exc:
            assert "no snps" in str(exc)

    def test_can_catch_narrowly(self):
        try:
            raise LLMRateLimitError("429")
        except LLMRateLimitError:
            pass  # expected
