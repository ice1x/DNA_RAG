"""Parser for MyHeritage raw DNA data exports.

MyHeritage files are comma-separated CSV with comment lines::

    # MyHeritage DNA raw data
    RSID,CHROMOSOME,POSITION,RESULT
    "rs12345","1","12345","AG"

The ``RESULT`` column is renamed to ``GENOTYPE`` for consistency.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame

from dna_rag.exceptions import InvalidDNAFileError
from dna_rag.logging import get_logger

logger = get_logger(__name__)

_MARKER = "myheritage"


class MyHeritageParser:
    """Parser for MyHeritage raw DNA data files."""

    @staticmethod
    def can_parse(file_path: Path) -> bool:
        """Detect by header comments or CSV with RESULT column."""
        try:
            with file_path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        if _MARKER in stripped.lower():
                            return True
                        continue
                    # First non-comment line: check for comma-separated layout
                    parts = stripped.split(",")
                    if len(parts) >= 4:
                        # Could be the header row or a data row
                        if "result" in stripped.lower():
                            return True
                        # 4-column comma-separated with rs-prefixed first field
                        first = parts[0].strip().strip('"').lower()
                        if first.startswith("rs"):
                            return True
                    return False
        except OSError:
            return False
        return False

    @staticmethod
    def parse(file_path: Path) -> DataFrame:
        """Parse a MyHeritage raw data file.

        Returns:
            Standardised DataFrame.

        Raises:
            InvalidDNAFileError: File cannot be parsed.
        """
        try:
            # Try reading with header first (RSID,CHROMOSOME,POSITION,RESULT)
            df = pd.read_csv(
                file_path,
                comment="#",
                delimiter=",",
                dtype=str,
            )
            # Normalise column names
            df.columns = [c.strip().upper() for c in df.columns]

            if "RESULT" in df.columns:
                df = df.rename(columns={"RESULT": "GENOTYPE"})

            expected = {"RSID", "CHROMOSOME", "POSITION", "GENOTYPE"}
            if not expected.issubset(set(df.columns)):
                # Fall back to headerless parsing
                df = pd.read_csv(
                    file_path,
                    comment="#",
                    delimiter=",",
                    header=None,
                    names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"],
                    dtype=str,
                )
        except Exception as exc:
            raise InvalidDNAFileError(
                f"Failed to parse MyHeritage file {file_path}: {exc}"
            ) from exc

        if df.empty:
            logger.warning("empty_dna_file", parser="MyHeritage", file=str(file_path))
            return _empty_frame()

        df = df[["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"]].copy()
        df["RSID"] = df["RSID"].str.strip().str.strip('"').str.lower()
        df["CHROMOSOME"] = df["CHROMOSOME"].str.strip().str.strip('"').str.upper()
        df["GENOTYPE"] = df["GENOTYPE"].str.strip().str.strip('"').str.upper()
        pos_col = df["POSITION"].str.strip().str.strip('"')
        df["POSITION"] = pd.to_numeric(pos_col, errors="coerce").fillna(0).astype(int)

        # Filter out non-rs rows (e.g. header rows that slipped through)
        df = df[df["RSID"].str.startswith("rs")].copy()

        logger.info("parsed", parser="MyHeritage", rows=len(df))
        return df


def _empty_frame() -> DataFrame:
    return DataFrame(columns=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])
