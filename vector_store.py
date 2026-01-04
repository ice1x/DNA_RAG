"""Vector store module for RAG-based SNP retrieval.

This module provides a vector database interface for semantic search over
SNP descriptions and trait associations. It uses ChromaDB for storage and
sentence transformers for embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SNPVectorStore:
    """Vector database for SNP trait associations.

    This class provides semantic search over SNP descriptions to find
    relevant SNPs for a given query. It uses embeddings to match questions
    with SNP trait descriptions.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "snp_traits",
    ) -> None:
        """Initialize the vector store.

        Parameters
        ----------
        persist_directory : Optional[Path]
            Directory for persistent storage. If None, uses in-memory storage.
        embedding_model : str
            Name of the sentence-transformers model to use.
        collection_name : str
            Name of the ChromaDB collection.
        """
        self._embedding_model = SentenceTransformer(embedding_model)

        # Configure ChromaDB
        if persist_directory:
            persist_directory.mkdir(parents=True, exist_ok=True)
            settings = Settings(
                persist_directory=str(persist_directory),
                anonymized_telemetry=False,
            )
            self._client = chromadb.Client(settings)
        else:
            self._client = chromadb.Client()

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "SNP trait associations for semantic search"},
        )

    def add_snp(
        self,
        rsid: str,
        trait: str,
        gene: Optional[str] = None,
        chromosome: Optional[str] = None,
        position: Optional[int] = None,
        description: Optional[str] = None,
    ) -> None:
        """Add a SNP to the vector store.

        Parameters
        ----------
        rsid : str
            The RS identifier.
        trait : str
            Associated trait or phenotype.
        gene : Optional[str]
            Gene name.
        chromosome : Optional[str]
            Chromosome.
        position : Optional[int]
            Genomic position.
        description : Optional[str]
            Additional description.
        """
        # Create searchable text
        text_parts = [trait]
        if gene:
            text_parts.append(f"gene: {gene}")
        if description:
            text_parts.append(description)

        text = " ".join(text_parts)

        # Generate embedding
        embedding = self._embedding_model.encode(text).tolist()

        # Prepare metadata
        metadata: Dict[str, str | int] = {"trait": trait}
        if gene:
            metadata["gene"] = gene
        if chromosome:
            metadata["chromosome"] = chromosome
        if position:
            metadata["position"] = position

        # Add to collection
        self._collection.add(
            ids=[rsid],
            embeddings=[embedding],
            metadatas=[metadata],  # type: ignore
            documents=[text],
        )

        logger.debug(f"Added {rsid} to vector store")

    def add_snps_batch(self, snps: List[Dict[str, str | int]]) -> None:
        """Add multiple SNPs to the vector store.

        Parameters
        ----------
        snps : List[Dict[str, Any]]
            List of SNP dictionaries with keys: rsid, trait, gene, etc.
        """
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for snp in snps:
            rsid = str(snp["rsid"])
            trait = str(snp["trait"])
            gene = snp.get("gene")
            chromosome = snp.get("chromosome")
            position = snp.get("position")
            description = snp.get("description")

            # Create searchable text
            text_parts = [trait]
            if gene:
                text_parts.append(f"gene: {gene}")
            if description:
                text_parts.append(str(description))

            text = " ".join(text_parts)

            # Generate embedding
            embedding = self._embedding_model.encode(text).tolist()

            # Prepare metadata
            metadata: Dict[str, str | int] = {"trait": trait}
            if gene:
                metadata["gene"] = str(gene)
            if chromosome:
                metadata["chromosome"] = str(chromosome)
            if position:
                metadata["position"] = int(position)

            ids.append(rsid)
            embeddings.append(embedding)
            metadatas.append(metadata)  # type: ignore
            documents.append(text)

        # Add all at once
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,  # type: ignore
            documents=documents,
        )

        logger.info(f"Added {len(snps)} SNPs to vector store")

    def search(
        self, query: str, n_results: int = 10, min_similarity: float = 0.3
    ) -> Dict[str, Dict[str, str | int]]:
        """Search for SNPs relevant to a query.

        Parameters
        ----------
        query : str
            The search query (e.g., "lactose tolerance").
        n_results : int
            Maximum number of results to return.
        min_similarity : float
            Minimum cosine similarity threshold (0-1).

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping RSIDs to metadata.
        """
        # Generate query embedding
        query_embedding = self._embedding_model.encode(query).tolist()

        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        # Parse results
        snp_dict: Dict[str, Dict[str, str | int]] = {}

        if not results["ids"] or not results["ids"][0]:
            return snp_dict

        for i, rsid in enumerate(results["ids"][0]):
            # Get metadata
            metadata = results["metadatas"][0][i]  # type: ignore

            # Get distance (ChromaDB returns squared L2 distance by default)
            # Convert to similarity score
            distance = results["distances"][0][i] if results["distances"] else 1.0  # type: ignore
            similarity = 1.0 / (1.0 + distance)

            # Filter by similarity threshold
            if similarity < min_similarity:
                continue

            snp_dict[rsid] = {
                "trait": metadata.get("trait", ""),
                "gene": metadata.get("gene", ""),
                "chromosome": metadata.get("chromosome", ""),
                "position": metadata.get("position", 0),
                "similarity": similarity,
            }

        logger.info(f"Found {len(snp_dict)} SNPs for query: {query}")
        return snp_dict

    def count(self) -> int:
        """Return the number of SNPs in the store.

        Returns
        -------
        int
            Number of SNPs.
        """
        return self._collection.count()

    def clear(self) -> None:
        """Clear all SNPs from the store."""
        # Delete and recreate collection
        collection_name = self._collection.name
        self._client.delete_collection(name=collection_name)
        self._collection = self._client.create_collection(
            name=collection_name,
            metadata={"description": "SNP trait associations for semantic search"},
        )
        logger.info("Cleared vector store")


def create_sample_snp_database(store: SNPVectorStore) -> None:
    """Populate the vector store with sample SNP data.

    This function adds common SNPs with well-known trait associations
    for testing and demonstration purposes.

    Parameters
    ----------
    store : SNPVectorStore
        The vector store to populate.
    """
    sample_snps = [
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

    store.add_snps_batch(sample_snps)
    logger.info(f"Added {len(sample_snps)} sample SNPs to vector store")
