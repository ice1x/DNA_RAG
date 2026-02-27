"""Fixtures for UI (Streamlit) tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SNP_JSON = json.dumps(
    {
        "rs1": {
            "gene": "LCT",
            "chromosome": "1",
            "position": 111,
            "trait": "Lactose tolerance",
        }
    }
)

INTERPRETATION = "Based on your genotype AA at rs1, you are likely lactose tolerant."


@pytest.fixture
def dna_file(tmp_path: Path) -> Path:
    """Minimal 23andMe DNA file for UI tests."""
    f = tmp_path / "sample.txt"
    f.write_text("# 23andMe data\nrs1\t1\t111\tAA\nrs2\t1\t112\tGG\n")
    return f
