"""DNA file parser protocol.

All parsers produce a :class:`~pandas.DataFrame` with the same schema:

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Example
   * - RSID
     - str (lowercase)
     - ``rs12345``
   * - CHROMOSOME
     - str
     - ``1``, ``X``, ``MT``
   * - POSITION
     - int
     - ``12345``
   * - GENOTYPE
     - str
     - ``AA``, ``AG``, ``--``
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pandas import DataFrame


@runtime_checkable
class DNAParser(Protocol):
    """Protocol for DNA file format parsers."""

    @staticmethod
    def can_parse(file_path: Path) -> bool:
        """Return ``True`` if this parser can handle *file_path*.

        Implementations should inspect only the first few lines of the
        file (header / comments) to make a fast decision.
        """
        ...

    @staticmethod
    def parse(file_path: Path) -> DataFrame:
        """Parse *file_path* into the standardised 4-column DataFrame.

        Raises:
            InvalidDNAFileError: The file is malformed.
        """
        ...
