"""Optional vector store for semantic SNP retrieval (RAG).

This module provides :class:`SNPVectorStore` -- a ChromaDB-backed semantic
search over SNP trait descriptions.  It is entirely **optional** and only
imported when the ``[rag]`` extra is installed::

    pip install dna-rag[rag]

If the heavy dependencies (``chromadb``, ``sentence-transformers``) are not
installed, importing this module raises :class:`ImportError` with a helpful
message.

Example::

    from dna_rag.vector_store import SNPVectorStore, create_sample_store

    store = SNPVectorStore()
    create_sample_store(store)
    results = store.search("lactose tolerance", n_results=5)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import chromadb  # type: ignore[import-untyped]
    from chromadb.config import Settings as ChromaSettings  # type: ignore[import-untyped]
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
except ImportError as _import_err:
    raise ImportError(
        "The [rag] extra is required for vector store functionality. "
        "Install it with:  pip install dna-rag[rag]"
    ) from _import_err


class SNPVectorStore:
    """ChromaDB-backed semantic search over SNP trait associations.

    Args:
        persist_directory: If given, ChromaDB persists to disk.
            ``None`` means in-memory only.
        embedding_model: HuggingFace sentence-transformers model name.
        collection_name: ChromaDB collection to use.
    """

    def __init__(
        self,
        persist_directory: Path | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "snp_traits",
    ) -> None:
        self._encoder = SentenceTransformer(embedding_model)

        if persist_directory:
            persist_directory.mkdir(parents=True, exist_ok=True)
            settings = ChromaSettings(
                persist_directory=str(persist_directory),
                anonymized_telemetry=False,
            )
            self._client = chromadb.Client(settings)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "SNP trait associations for semantic search"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_snp(
        self,
        rsid: str,
        trait: str,
        gene: str | None = None,
        chromosome: str | None = None,
        position: int | None = None,
        description: str | None = None,
    ) -> None:
        """Add a single SNP to the store."""
        text = _build_text(trait, gene, description)
        embedding = self._encoder.encode(text).tolist()
        metadata = _build_metadata(trait, gene, chromosome, position)

        self._collection.add(
            ids=[rsid],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text],
        )

    def add_snps_batch(self, snps: list[dict[str, Any]]) -> None:
        """Add multiple SNPs at once.

        Each dict must contain ``rsid`` and ``trait``; optional keys:
        ``gene``, ``chromosome``, ``position``, ``description``.
        """
        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        for snp in snps:
            text = _build_text(
                str(snp["trait"]),
                snp.get("gene"),  # type: ignore[arg-type]
                snp.get("description"),  # type: ignore[arg-type]
            )
            ids.append(str(snp["rsid"]))
            embeddings.append(self._encoder.encode(text).tolist())
            metadatas.append(
                _build_metadata(
                    str(snp["trait"]),
                    snp.get("gene"),  # type: ignore[arg-type]
                    snp.get("chromosome"),  # type: ignore[arg-type]
                    snp.get("position"),  # type: ignore[arg-type]
                ),
            )
            documents.append(text)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        logger.info("Added %d SNPs to vector store", len(snps))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 10,
        min_similarity: float = 0.3,
    ) -> dict[str, dict[str, Any]]:
        """Semantic search for SNPs relevant to *query*.

        Returns:
            Mapping of RSID to metadata dicts (trait, gene, chromosome,
            position, similarity).
        """
        query_embedding = self._encoder.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        found: dict[str, dict[str, Any]] = {}
        if not results["ids"] or not results["ids"][0]:
            return found

        for i, rsid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = (
                results["distances"][0][i] if results["distances"] else 1.0
            )
            similarity = 1.0 / (1.0 + distance)
            if similarity < min_similarity:
                continue
            found[rsid] = {
                "trait": meta.get("trait", ""),
                "gene": meta.get("gene", ""),
                "chromosome": meta.get("chromosome", ""),
                "position": meta.get("position", 0),
                "similarity": round(similarity, 4),
            }

        return found

    def count(self) -> int:
        """Number of SNPs in the store."""
        return self._collection.count()

    def clear(self) -> None:
        """Drop all data and recreate the collection."""
        name = self._collection.name
        self._client.delete_collection(name=name)
        self._collection = self._client.create_collection(
            name=name,
            metadata={"description": "SNP trait associations for semantic search"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_text(
    trait: str,
    gene: str | None = None,
    description: str | None = None,
) -> str:
    parts = [trait]
    if gene:
        parts.append(f"gene: {gene}")
    if description:
        parts.append(str(description))
    return " ".join(parts)


def _build_metadata(
    trait: str,
    gene: str | None = None,
    chromosome: str | None = None,
    position: int | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {"trait": trait}
    if gene:
        meta["gene"] = gene
    if chromosome:
        meta["chromosome"] = str(chromosome)
    if position is not None:
        meta["position"] = int(position)
    return meta


# ---------------------------------------------------------------------------
# Demo data loader
# ---------------------------------------------------------------------------

_SAMPLE_SNPS: list[dict[str, Any]] = [
    {
        "rsid": "rs4988235",
        "trait": "Lactose tolerance",
        "gene": "MCM6",
        "chromosome": "2",
        "position": 136608646,
        "description": "Associated with lactase persistence in adults",
    },
    {
        "rsid": "rs1815739",
        "trait": "Athletic performance and muscle fiber type",
        "gene": "ACTN3",
        "chromosome": "11",
        "position": 66560624,
        "description": "R577X variant affects sprint and power performance",
    },
    {
        "rsid": "rs1800497",
        "trait": "Dopamine receptor density",
        "gene": "ANKK1",
        "chromosome": "11",
        "position": 113270828,
        "description": "Associated with reward sensitivity and addiction",
    },
    {
        "rsid": "rs7412",
        "trait": "Cholesterol metabolism and Alzheimer's risk",
        "gene": "APOE",
        "chromosome": "19",
        "position": 44908822,
        "description": "APOE epsilon 2 allele, protective against Alzheimer's",
    },
    {
        "rsid": "rs429358",
        "trait": "Alzheimer's disease risk",
        "gene": "APOE",
        "chromosome": "19",
        "position": 44908684,
        "description": "APOE epsilon 4 allele, increased Alzheimer's risk",
    },
    {
        "rsid": "rs1801133",
        "trait": "Folate metabolism and homocysteine levels",
        "gene": "MTHFR",
        "chromosome": "1",
        "position": 11796321,
        "description": "C677T variant affects vitamin B metabolism",
    },
]


def create_sample_store(store: SNPVectorStore) -> None:
    """Populate *store* with well-known example SNPs for demos."""
    store.add_snps_batch(_SAMPLE_SNPS)
    logger.info("Added %d sample SNPs to vector store", len(_SAMPLE_SNPS))
