"""Custom exception hierarchy for dna-rag.

All exceptions inherit from :class:`DNARagError` so callers can catch
broadly (``except DNARagError``) or narrowly (``except LLMResponseError``).
"""

from __future__ import annotations


class DNARagError(Exception):
    """Base exception for all dna-rag errors."""


# --- LLM errors -----------------------------------------------------------


class LLMError(DNARagError):
    """Base for all LLM-related errors."""


class LLMConnectionError(LLMError):
    """Cannot reach the LLM API."""


class LLMResponseError(LLMError):
    """LLM returned an invalid or unparseable response."""


class LLMRateLimitError(LLMError):
    """LLM API rate limit exceeded."""


# --- Parsing errors --------------------------------------------------------


class ParsingError(DNARagError):
    """Base for DNA file parsing errors."""


class UnsupportedFormatError(ParsingError):
    """DNA file format not recognised."""


class InvalidDNAFileError(ParsingError):
    """DNA file is malformed or contains invalid data."""


# --- Analysis errors -------------------------------------------------------


class AnalysisError(DNARagError):
    """Error during the analysis pipeline."""


class NoSNPsFoundError(AnalysisError):
    """LLM could not identify relevant SNPs for the question."""


class NoMatchingVariantsError(AnalysisError):
    """No matching variants found in the user's DNA file."""


# --- Configuration errors --------------------------------------------------


class ConfigurationError(DNARagError):
    """Invalid or missing configuration."""


# --- Cache errors ----------------------------------------------------------


class CacheError(DNARagError):
    """Cache operation failed."""
