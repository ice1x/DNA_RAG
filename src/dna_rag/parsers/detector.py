"""Auto-detection of DNA file formats.

Tries each registered parser's :meth:`can_parse` method in order.
The first parser that claims the file wins.
"""

from __future__ import annotations

from pathlib import Path

from pandas import DataFrame

from dna_rag.exceptions import UnsupportedFormatError
from dna_rag.logging import get_logger
from dna_rag.parsers.ancestrydna import AncestryDNAParser
from dna_rag.parsers.myheritage import MyHeritageParser
from dna_rag.parsers.twentythreeandme import TwentyThreeAndMeParser
from dna_rag.parsers.vcf import VCFParser

logger = get_logger(__name__)

# Each parser class has static can_parse(Path)->bool and parse(Path)->DataFrame.
# Order matters: most distinctive format first.
_PARSERS: list[type] = [
    VCFParser,
    AncestryDNAParser,
    TwentyThreeAndMeParser,
    MyHeritageParser,
]


def detect_and_parse(file_path: Path) -> DataFrame:
    """Auto-detect the DNA file format and parse it.

    Args:
        file_path: Path to the raw DNA data file.

    Returns:
        Standardised :class:`~pandas.DataFrame` with columns
        ``RSID``, ``CHROMOSOME``, ``POSITION``, ``GENOTYPE``.

    Raises:
        FileNotFoundError: *file_path* does not exist.
        UnsupportedFormatError: No parser recognises the file.
        InvalidDNAFileError: The file is recognised but malformed.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"DNA file not found: {file_path}")

    for parser_cls in _PARSERS:
        if parser_cls.can_parse(file_path):  # type: ignore[attr-defined]
            logger.info(
                "format_detected",
                parser=parser_cls.__name__,
                file=str(file_path),
            )
            return parser_cls.parse(file_path)  # type: ignore[attr-defined, no-any-return]

    raise UnsupportedFormatError(
        f"Could not detect DNA file format for: {file_path}. "
        "Supported formats: VCF, 23andMe, AncestryDNA, MyHeritage."
    )
