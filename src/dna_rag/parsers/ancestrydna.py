"""Parser for AncestryDNA raw data exports.

AncestryDNA files are tab-separated with two allele columns::

    #AncestryDNA raw data download
    rsid  chromosome  position  allele1  allele2
    rs12345  1  12345  A  G
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame

from dna_rag.exceptions import InvalidDNAFileError
from dna_rag.logging import get_logger

logger = get_logger(__name__)

_MARKER = "ancestrydna"


class AncestryDNAParser:
    """Parser for AncestryDNA raw DNA data files."""

    @staticmethod
    def can_parse(file_path: Path) -> bool:
        """Detect by checking for ``AncestryDNA`` in header comments."""
        try:
            with file_path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.startswith("#"):
                        # Check for 5-column tab-separated layout (allele1/allele2)
                        parts = line.strip().split("\t")
                        if len(parts) == 5 and parts[0].lower().startswith("rs"):
                            return True
                        # Also accept the header row
                        return len(parts) == 5 and "allele" in line.lower()
                    if _MARKER in line.lower():
                        return True
        except OSError:
            return False
        return False

    @staticmethod
    def parse(file_path: Path) -> DataFrame:
        """Parse an AncestryDNA raw data file.

        Merges ``allele1`` + ``allele2`` into a single ``GENOTYPE`` column.

        Returns:
            Standardised DataFrame.

        Raises:
            InvalidDNAFileError: File cannot be parsed.
        """
        try:
            df = pd.read_csv(
                file_path,
                comment="#",
                delimiter="\t",
                header=None,
                names=["RSID", "CHROMOSOME", "POSITION", "ALLELE1", "ALLELE2"],
                dtype={"RSID": str, "CHROMOSOME": str, "ALLELE1": str, "ALLELE2": str},
            )
        except Exception as exc:
            raise InvalidDNAFileError(
                f"Failed to parse AncestryDNA file {file_path}: {exc}"
            ) from exc

        if df.empty:
            logger.warning("empty_dna_file", parser="AncestryDNA", file=str(file_path))
            return _empty_frame()

        # Skip header-like rows where RSID doesn't start with "rs"
        df = df[df["RSID"].str.strip().str.lower().str.startswith("rs")].copy()

        df["RSID"] = df["RSID"].str.strip().str.lower()
        df["CHROMOSOME"] = df["CHROMOSOME"].str.strip().str.upper()
        df["ALLELE1"] = df["ALLELE1"].fillna("0").str.strip().str.upper()
        df["ALLELE2"] = df["ALLELE2"].fillna("0").str.strip().str.upper()
        df["GENOTYPE"] = df["ALLELE1"] + df["ALLELE2"]
        df["POSITION"] = pd.to_numeric(df["POSITION"], errors="coerce").fillna(0).astype(int)

        df = df[["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"]]
        logger.info("parsed", parser="AncestryDNA", rows=len(df))
        return df


def _empty_frame() -> DataFrame:
    return DataFrame(columns=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])
