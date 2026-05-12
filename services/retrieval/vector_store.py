"""
vector_store.py — Stage 6: Vector Storage with ChromaDB
=========================================================
WHY VECTOR DATABASES EXIST:
  Traditional databases store structured data (rows, columns) and support
  exact-match queries (WHERE name = "John"). They can't answer:
  "Find the text MOST SIMILAR IN MEANING to this query."

  Vector databases store high-dimensional vectors and support:
  - Approximate Nearest Neighbor (ANN) search
  - Cosine similarity ranking
  - Metadata filtering alongside vector search

  This enables: "Find the 20 text chunks most semantically similar to the
  user's question, filtering to only chunks from document X."

HOW CHROMADB WORKS:
  1. You provide: text content + embedding vector + metadata
  2. ChromaDB indexes the vectors using HNSW (Hierarchical Navigable Small World)
     — a graph-based ANN algorithm that's fast and memory-efficient
  3. At query time, it finds nearest vectors using cosine similarity
  4. Returns: matching IDs, distances, documents, metadata

COSINE SIMILARITY EXPLAINED:
  Two vectors have cosine similarity of:
  - 1.0: Identical direction (semantically identical text)
  - 0.0: Perpendicular (unrelated text)
  - -1.0: Opposite direction (semantically opposite — rare in practice)

  Formula: cosine_sim(A, B) = (A · B) / (|A| × |B|)
  ChromaDB uses "distance" (1 - cosine_similarity) where smaller = more similar.

WHY CHROMADB FOR DEVELOPMENT:
  - Zero infrastructure: runs in-process, persists to local disk
  - No server to manage or pay for
  - Supports metadata filtering
  - Drop-in replacement when you're ready for Pinecone/Qdrant

MIGRATING TO PINECONE/QDRANT:
  The VectorStore class below abstracts ChromaDB. To switch:
  1. Implement a new class following the same interface
  2. Update VECTOR_STORE_TYPE in config.py
  (Alternatively, use LangChain's VectorStore abstraction for even easier swapping)

PRODUCTION CONSIDERATIONS:
  - ChromaDB is single-node; use Qdrant/Pinecone for horizontal scaling.
  - Back up the vector_store/ directory regularly.
  - Monitor index size and query latency as collections grow.
  - Use Pinecone for serverless, managed vector storage at scale.
"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from services.ingestion.chunker import TextChunk
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VectorStore:
    """
    Persistent vector store backed by ChromaDB.

    DESIGN:
      - Wraps ChromaDB with a clean interface
      - Handles collection creation and management
      - Supports metadata filtering on retrieval
      - Deduplicates chunks by chunk_id

    USAGE:
        store = VectorStore()
        store.add_chunks(chunks_with_embeddings)
        results = store.search("What is the methodology?", top_k=20)
    """

    def __init__(self):
        persist_dir = str(settings.vector_store_path.absolute())
        logger.info(f"Initializing ChromaDB at: {persist_dir}")

        # PersistentClient saves data to disk across restarts.
        # Use EphemeralClient() for in-memory (testing only).
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Disable usage reporting
            ),
        )

        # A "collection" in ChromaDB is like a table — groups related vectors.
        # We use one collection for all seminar reports.
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",        # Use cosine similarity (not L2)
                "description": "Seminar report chunks",
            },
        )

        count = self._collection.count()
        logger.info(
            f"Vector store ready: collection='{settings.CHROMA_COLLECTION_NAME}', "
            f"existing chunks={count}"
        )

    def add_chunks(
        self,
        chunks_with_embeddings: list[tuple[TextChunk, list[float]]],
    ) -> int:
        """
        Adds embedded chunks to the vector store.

        DEDUPLICATION:
          ChromaDB uses chunk_id as the unique key. If we call add_chunks with
          the same document twice, we use upsert (update-or-insert) to avoid
          duplicate entries.

        BATCH SIZE:
          ChromaDB has a max batch size of ~41,000. We batch at 500 to be safe
          and to log progress for large documents.

        Args:
            chunks_with_embeddings: List of (TextChunk, vector) tuples from EmbeddingGenerator.

        Returns:
            Number of chunks successfully stored.
        """
        if not chunks_with_embeddings:
            logger.warning("No chunks to add to vector store.")
            return 0

        # Prepare data in ChromaDB's expected format
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk, vector in chunks_with_embeddings:
            ids.append(chunk.chunk_id)
            embeddings.append(vector)
            documents.append(chunk.text)
            metadatas.append(chunk.to_metadata())

        # Batch upsert (update or insert)
        BATCH_SIZE = 500
        total_added = 0

        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i:i + BATCH_SIZE]
            batch_embeddings = embeddings[i:i + BATCH_SIZE]
            batch_documents = documents[i:i + BATCH_SIZE]
            batch_metadatas = metadatas[i:i + BATCH_SIZE]

            self._collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )
            total_added += len(batch_ids)
            logger.debug(f"Upserted batch {i // BATCH_SIZE + 1}: {len(batch_ids)} chunks")

        logger.info(
            f"Vector store update complete: {total_added} chunks stored. "
            f"Total in collection: {self._collection.count()}"
        )
        return total_added

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
        filter_source: Optional[str] = None,
    ) -> list[dict]:
        """
        Performs semantic similarity search against stored chunks.

        METADATA FILTERING:
          ChromaDB supports WHERE filters alongside vector search.
          Example: Only search within a specific document:
            filter_doc_id="sha256_of_specific_doc"
          This is critical for focused queries on one report.

        HOW IT WORKS:
          1. Takes the query embedding vector
          2. Computes cosine similarity with all stored vectors
          3. Returns the top_k closest matches with their metadata

        Args:
            query_embedding: The embedded query vector.
            top_k: Number of candidate results to return.
            filter_doc_id: Optional doc_id to restrict search to one document.
            filter_source: Optional source filename to restrict search.

        Returns:
            List of dicts with keys: chunk_id, text, metadata, distance, score
        """
        # Build metadata filter (WHERE clause)
        where_filter = None
        if filter_doc_id:
            where_filter = {"doc_id": {"$eq": filter_doc_id}}
        elif filter_source:
            where_filter = {"source": {"$eq": filter_source}}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        # Parse ChromaDB results into a clean list of dicts
        chunks = []
        if not results["ids"] or not results["ids"][0]:
            return []

        for chunk_id, doc, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Convert distance to similarity score (ChromaDB cosine distance = 1 - similarity)
            similarity_score = 1.0 - distance

            chunks.append({
                "chunk_id": chunk_id,
                "text": doc,
                "metadata": metadata,
                "distance": distance,
                "score": similarity_score,
            })

        logger.debug(
            f"Vector search returned {len(chunks)} results "
            f"(best score: {chunks[0]['score']:.3f} if chunks else 'N/A')"
        )
        return chunks

    def list_documents(self) -> list[dict]:
        """
        Returns a list of unique documents in the vector store.

        IMPLEMENTATION:
          ChromaDB doesn't have a native "list distinct documents" query.
          We get all metadata and deduplicate by doc_id.
          For large collections, this is expensive — consider a separate
          document registry table.

        Returns:
            List of dicts with: doc_id, source, total_pages, chunk_count
        """
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception:
            return []

        docs: dict[str, dict] = {}
        for metadata in result["metadatas"]:
            doc_id = metadata.get("doc_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "source": metadata.get("source", "unknown"),
                    "total_pages": metadata.get("total_pages", 0),
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    def delete_document(self, doc_id: str) -> int:
        """
        Deletes all chunks belonging to a specific document.

        Args:
            doc_id: SHA-256 hash of the document to delete.

        Returns:
            Number of chunks deleted.
        """
        try:
            result = self._collection.get(
                where={"doc_id": {"$eq": doc_id}},
                include=[],
            )
            ids_to_delete = result["ids"]

            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id={doc_id}")
                return len(ids_to_delete)
            else:
                logger.warning(f"No chunks found for doc_id={doc_id}")
                return 0
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return 0

    def get_collection_stats(self) -> dict:
        """Returns basic statistics about the vector store."""
        return {
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "total_chunks": self._collection.count(),
            "persist_directory": str(settings.vector_store_path),
        }
