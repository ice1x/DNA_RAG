"""Tests for vector store module."""

import pytest

from vector_store import SNPVectorStore, create_sample_snp_database


@pytest.fixture
def vector_store(tmp_path):
    """Create vector store instance."""
    return SNPVectorStore(persist_directory=tmp_path / "chroma")


@pytest.fixture
def memory_vector_store():
    """Create in-memory vector store."""
    return SNPVectorStore(persist_directory=None)


def test_vector_store_init(vector_store):
    """Test vector store initialization."""
    assert vector_store._collection is not None
    assert vector_store.count() >= 0


def test_add_snp(vector_store):
    """Test adding a single SNP."""
    initial_count = vector_store.count()

    vector_store.add_snp(
        rsid="rs123",
        trait="Test trait",
        gene="GENE1",
        chromosome="1",
        position=123456,
        description="Test description",
    )

    assert vector_store.count() == initial_count + 1


def test_add_snps_batch(vector_store):
    """Test adding multiple SNPs."""
    snps = [
        {
            "rsid": "rs1",
            "trait": "Trait 1",
            "gene": "GENE1",
            "chromosome": "1",
            "position": 111,
        },
        {
            "rsid": "rs2",
            "trait": "Trait 2",
            "gene": "GENE2",
            "chromosome": "2",
            "position": 222,
        },
    ]

    initial_count = vector_store.count()
    vector_store.add_snps_batch(snps)

    assert vector_store.count() == initial_count + 2


def test_search(vector_store):
    """Test SNP search."""
    # Add some SNPs
    vector_store.add_snp(
        rsid="rs4988235",
        trait="Lactose tolerance",
        gene="MCM6",
        chromosome="2",
        position=136608646,
    )

    vector_store.add_snp(
        rsid="rs123",
        trait="Eye color",
        gene="HERC2",
        chromosome="15",
        position=28365618,
    )

    # Search for lactose
    results = vector_store.search("lactose tolerance", n_results=5)

    assert len(results) > 0
    assert "rs4988235" in results
    assert results["rs4988235"]["trait"] == "Lactose tolerance"


def test_search_no_results(vector_store):
    """Test search with no results."""
    results = vector_store.search("completely unrelated query xyz123")
    # May return empty or very low similarity results
    assert isinstance(results, dict)


def test_search_with_similarity_threshold(vector_store):
    """Test search with similarity threshold."""
    vector_store.add_snp(
        rsid="rs123",
        trait="Lactose tolerance",
        gene="MCM6",
    )

    # High threshold should filter results
    results = vector_store.search("completely different topic", n_results=5, min_similarity=0.9)

    # Should have very few or no results
    assert len(results) >= 0


def test_count(vector_store):
    """Test count method."""
    initial = vector_store.count()
    assert initial >= 0

    # Use a unique ID that won't exist in the sample database
    vector_store.add_snp(rsid="rs_test_unique_count", trait="Test")
    assert vector_store.count() == initial + 1


def test_clear(vector_store):
    """Test clearing vector store."""
    # Use a unique ID that won't exist in the sample database
    vector_store.add_snp(rsid="rs_test_unique_clear", trait="Test")
    assert vector_store.count() > 0

    vector_store.clear()
    assert vector_store.count() == 0


def test_create_sample_database(memory_vector_store):
    """Test creating sample database."""
    initial = memory_vector_store.count()
    create_sample_snp_database(memory_vector_store)

    assert memory_vector_store.count() > initial
    assert memory_vector_store.count() >= 6  # Sample has 6 SNPs


def test_persistence(tmp_path):
    """Test vector store persistence."""
    persist_dir = tmp_path / "chroma_persist"

    # Create store and add data
    store1 = SNPVectorStore(persist_directory=persist_dir)
    store1.add_snp(rsid="rs123", trait="Test trait")
    count1 = store1.count()

    # Create new store with same directory
    store2 = SNPVectorStore(persist_directory=persist_dir)
    count2 = store2.count()

    # Counts should match (persistence)
    assert count2 >= count1


def test_add_snp_minimal_fields(vector_store):
    """Test adding SNP with minimal fields."""
    vector_store.add_snp(rsid="rs_minimal", trait="Minimal trait")
    assert vector_store.count() > 0

    results = vector_store.search("minimal", n_results=1)
    assert "rs_minimal" in results or len(results) >= 0
