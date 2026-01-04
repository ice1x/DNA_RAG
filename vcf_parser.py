"""VCF file parser for genomic variant data.

This module provides functionality to parse VCF (Variant Call Format) files
and convert them to a standard format compatible with DNA_RAG.
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class VCFParser:
    """Parser for VCF files.

    Supports both plain-text and gzipped VCF files. Extracts SNP information
    including RS IDs, chromosomes, positions, and genotypes.
    """

    def __init__(self, vcf_file: Path) -> None:
        """Initialize VCF parser.

        Parameters
        ----------
        vcf_file : Path
            Path to VCF file (.vcf or .vcf.gz).
        """
        self.vcf_file = vcf_file

        if not vcf_file.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_file}")

    def parse(self) -> DataFrame:
        """Parse VCF file and return standardized DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with columns: RSID, CHROMOSOME, POSITION, GENOTYPE.
        """
        variants = []

        with self._open_vcf() as vcf:
            for record in self._parse_records(vcf):
                if record:
                    variants.append(record)

        if not variants:
            logger.warning(f"No variants found in {self.vcf_file}")
            return pd.DataFrame(columns=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])

        df = pd.DataFrame(variants)
        logger.info(f"Parsed {len(df)} variants from {self.vcf_file}")
        return df

    def _open_vcf(self) -> TextIO:
        """Open VCF file, handling gzip compression.

        Returns
        -------
        TextIO
            File handle.
        """
        if self.vcf_file.suffix == ".gz":
            return gzip.open(self.vcf_file, "rt", encoding="utf-8")  # type: ignore
        return open(self.vcf_file, encoding="utf-8")

    def _parse_records(self, vcf: TextIO) -> Iterator[dict[str, str | int] | None]:
        """Parse VCF records.

        Parameters
        ----------
        vcf : TextIO
            VCF file handle.

        Yields
        ------
        Optional[Dict[str, Any]]
            Parsed record or None.
        """
        for line in vcf:
            # Skip header lines
            if line.startswith("#"):
                continue

            # Parse variant line
            record = self._parse_variant_line(line)
            if record:
                yield record

    def _parse_variant_line(self, line: str) -> dict[str, str | int] | None:
        """Parse a single VCF variant line.

        Parameters
        ----------
        line : str
            VCF line.

        Returns
        -------
        Optional[Dict[str, Any]]
            Parsed variant record or None.
        """
        try:
            fields = line.strip().split("\t")

            if len(fields) < 10:
                return None

            chrom = fields[0]
            pos = int(fields[1])
            rs_id = fields[2]
            ref = fields[3]
            alt = fields[4]
            info = fields[7]
            format_field = fields[8]
            sample = fields[9]

            # Extract genotype
            genotype = self._extract_genotype(format_field, sample, ref, alt)

            if not genotype:
                return None

            # Handle RS ID
            if rs_id == ".":
                # No RS ID, try to extract from INFO field
                rs_id = self._extract_rsid_from_info(info)

            # Skip variants without RS ID
            if not rs_id or rs_id == ".":
                return None

            return {
                "RSID": rs_id,
                "CHROMOSOME": chrom.replace("chr", ""),
                "POSITION": pos,
                "GENOTYPE": genotype,
            }

        except (ValueError, IndexError) as exc:
            logger.debug(f"Failed to parse VCF line: {exc}")
            return None

    @staticmethod
    def _extract_genotype(format_field: str, sample: str, ref: str, alt: str) -> str | None:
        """Extract genotype from FORMAT and sample fields.

        Parameters
        ----------
        format_field : str
            VCF FORMAT field.
        sample : str
            VCF sample field.
        ref : str
            Reference allele.
        alt : str
            Alternate allele(s).

        Returns
        -------
        Optional[str]
            Genotype (e.g., "AA", "AG", "GG") or None.
        """
        format_keys = format_field.split(":")
        sample_values = sample.split(":")

        if "GT" not in format_keys:
            return None

        gt_index = format_keys.index("GT")

        if gt_index >= len(sample_values):
            return None

        gt = sample_values[gt_index]

        # Parse genotype
        if "/" in gt:
            alleles = gt.split("/")
        elif "|" in gt:
            alleles = gt.split("|")
        else:
            return None

        # Convert numeric genotype to bases
        try:
            allele_map = {
                "0": ref[0] if ref else "N",
                "1": alt.split(",")[0][0] if alt else "N",
            }

            genotype_bases = "".join([allele_map.get(a, "N") for a in alleles if a in allele_map])

            if len(genotype_bases) != 2:
                return None

            return genotype_bases

        except (IndexError, KeyError):
            return None

    @staticmethod
    def _extract_rsid_from_info(info: str) -> str | None:
        """Extract RS ID from INFO field.

        Parameters
        ----------
        info : str
            VCF INFO field.

        Returns
        -------
        Optional[str]
            RS ID or None.
        """
        for item in info.split(";"):
            if item.startswith("RS=") or item.startswith("rsID="):
                return item.split("=")[1]
            if item.startswith("dbSNP="):
                return item.split("=")[1]
        return None


def convert_vcf_to_csv(vcf_file: Path, output_file: Path) -> None:
    """Convert VCF file to CSV format.

    Parameters
    ----------
    vcf_file : Path
        Input VCF file.
    output_file : Path
        Output CSV file.
    """
    parser = VCFParser(vcf_file)
    df = parser.parse()

    df.to_csv(output_file, index=False, header=False)
    logger.info(f"Converted {vcf_file} to {output_file}")
