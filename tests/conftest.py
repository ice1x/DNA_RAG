import sys
from pathlib import Path

import pytest

# Ensure the project root and tests directory are on the import path
ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.extend([str(ROOT), str(TESTS)])


@pytest.fixture(autouse=True)
def reset_chromadb():
    """Reset ChromaDB singleton between tests to avoid settings conflicts."""
    yield
    # Clean up ChromaDB singleton after each test
    try:
        import chromadb.api.client

        chromadb.api.client.SharedSystemClient.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def sample_dna_file(tmp_path: Path) -> Path:
    """Create a small DNA CSV file for testing."""
    data = "rs1,1,111,AA\nrs2,1,112,GG\n"
    file = tmp_path / "dna.csv"
    file.write_text(data)
    return file
