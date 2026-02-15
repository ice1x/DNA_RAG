"""Parser for VCF (Variant Call Format) files.

VCF is a standard bioinformatics format for genetic variants::

    ##fileformat=VCFv4.1
    #CHROM  POS  ID  REF  ALT  QUAL  FILTER  INFO  FORMAT  SAMPLE
    1  12345  rs12345  A  G  .  PASS  .  GT  0/1

Supports plain-text (``.vcf``) and gzip-compressed (``.vcf.gz``) files.
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from pandas import DataFrame

from dna_rag.exceptions import InvalidDNAFileError

logger = logging.getLogger(__name__)

_VCF_HEADER = "#CHROM"
_VCF_MAGIC = "##fileformat=VCF"


class VCFParser:
    """Parser for VCF files conforming to :class:`~dna_rag.parsers.base.DNAParser`."""

    @staticmethod
    def can_parse(file_path: Path) -> bool:
        """Detect VCF by checking for ``##fileformat=VCF`` or ``#CHROM`` header."""
        try:
            opener = gzip.open if file_path.suffix == ".gz" else open
            with opener(file_path, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[call-overload]
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith(_VCF_MAGIC):
                        return True
                    if stripped.startswith(_VCF_HEADER):
                        return True
                    # If first non-empty line is neither, not a VCF
                    if not stripped.startswith("##"):
                        return False
        except OSError:
            return False
        return False

    @staticmethod
    def parse(file_path: Path) -> DataFrame:
        """Parse a VCF file into the standardised 4-column DataFrame.

        Returns:
            DataFrame with columns ``RSID``, ``CHROMOSOME``, ``POSITION``, ``GENOTYPE``.

        Raises:
            InvalidDNAFileError: The file cannot be parsed.
        """
        try:
            variants = list(_parse_vcf(file_path))
        except Exception as exc:
            raise InvalidDNAFileError(
                f"Failed to parse VCF file {file_path}: {exc}"
            ) from exc

        if not variants:
            logger.warning("empty_vcf_file", extra={"file": str(file_path)})
            return DataFrame(columns=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])

        df = DataFrame(variants)
        df["RSID"] = df["RSID"].str.strip().str.lower()
        df["CHROMOSOME"] = df["CHROMOSOME"].str.strip().str.upper()
        df["GENOTYPE"] = df["GENOTYPE"].str.strip().str.upper()
        logger.info("parsed VCF: %d variants from %s", len(df), file_path)
        return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_vcf(file_path: Path) -> TextIO:
    """Open a VCF file, transparently handling gzip."""
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8")  # type: ignore[return-value]
    return open(file_path, encoding="utf-8")  # noqa: SIM115


def _parse_vcf(file_path: Path) -> Iterator[dict[str, str | int]]:
    """Yield parsed variant records from a VCF file."""
    with _open_vcf(file_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            record = _parse_variant_line(line)
            if record is not None:
                yield record


def _parse_variant_line(line: str) -> dict[str, str | int] | None:
    """Parse a single VCF data line into a dict."""
    fields = line.strip().split("\t")
    if len(fields) < 10:
        return None

    chrom = fields[0]
    pos = fields[1]
    rs_id = fields[2]
    ref = fields[3]
    alt = fields[4]
    format_field = fields[8]
    sample = fields[9]

    # Skip variants without an RS ID
    if not rs_id or rs_id == ".":
        rs_id = _extract_rsid_from_info(fields[7]) or ""
    if not rs_id or rs_id == ".":
        return None

    genotype = _extract_genotype(format_field, sample, ref, alt)
    if not genotype:
        return None

    try:
        pos_int = int(pos)
    except ValueError:
        return None

    return {
        "RSID": rs_id,
        "CHROMOSOME": chrom.replace("chr", ""),
        "POSITION": pos_int,
        "GENOTYPE": genotype,
    }


def _extract_genotype(
    format_field: str, sample: str, ref: str, alt: str,
) -> str | None:
    """Convert VCF GT field into a two-character genotype string (e.g. ``AG``)."""
    format_keys = format_field.split(":")
    sample_values = sample.split(":")

    if "GT" not in format_keys:
        return None

    gt_index = format_keys.index("GT")
    if gt_index >= len(sample_values):
        return None

    gt = sample_values[gt_index]

    separator = "/" if "/" in gt else ("|" if "|" in gt else None)
    if separator is None:
        return None

    alleles = gt.split(separator)

    allele_map: dict[str, str] = {
        "0": ref[0] if ref else "N",
    }
    for i, a in enumerate(alt.split(","), start=1):
        allele_map[str(i)] = a[0] if a else "N"

    bases = [allele_map.get(a, "N") for a in alleles if a in allele_map]
    if len(bases) != 2:
        return None

    return "".join(bases)


def _extract_rsid_from_info(info: str) -> str | None:
    """Try to extract an RS ID from the VCF INFO field."""
    for item in info.split(";"):
        if item.startswith(("RS=", "rsID=", "dbSNP=")):
            value = item.split("=", 1)[1]
            if not value.startswith("rs"):
                value = f"rs{value}"
            return value
    return None
