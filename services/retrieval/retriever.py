"""
retriever.py — Stage 7: Semantic Retrieval
==========================================
WHY THIS EXISTS:
  This module bridges the embedding search with the rest of the pipeline.
  It takes a raw user query → embeds it → searches the vector store →
  returns structured results with metadata.

WHY TOP-K = 20?
  We retrieve MORE candidates than we'll ultimately use (20 candidates, keep 5).
  WHY? Because initial vector similarity is an imperfect signal.

  The vector similarity captures broad semantic relevance but may:
  - Return chunks that are "thematically close" but don't actually answer the question
  - Miss nuanced matches that require more careful reading

  The RERANKER (next stage) does a deeper pairwise comparison of
  (query, chunk) to find the truly most relevant chunks among the 20 candidates.

  This two-stage approach (retrieve many → rerank to few) is the industry standard
  for production RAG systems.

WHY K MATTERS:
  - k too small (k=3): Reranker has too few options; might miss the best chunk
  - k too large (k=100): More API calls, more latency, marginal gains in quality
  - k=20: Good balance for academic documents (10-50 pages)

RETRIEVAL LIMITATIONS:
  - Dense retrieval (embeddings) can miss rare keywords (e.g., "section 3.2.1")
  - Hybrid search (dense + sparse/BM25) handles this better
  - Future enhancement: combine BM25 keyword search with dense semantic search

PRODUCTION CONSIDERATIONS:
  - For very large collections, add approximate search (HNSW is already ANN).
  - Log retrieval latency per query for performance monitoring.
  - Add query preprocessing (spell correction, query expansion) for robustness.
"""

import time
from typing import Optional

from services.ingestion.embedder import EmbeddingGenerator
from services.retrieval.vector_store import VectorStore
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class Retriever:
    """
    Retrieves semantically relevant chunks for a given query.

    PIPELINE:
      1. Embed the user's query
      2. Search the vector store for the top-k similar chunks
      3. Return structured results with metadata

    USAGE:
        retriever = Retriever(vector_store, embedder)
        results = retriever.retrieve("What is the proposed algorithm?", top_k=20)
    """

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingGenerator):
        self._store = vector_store
        self._embedder = embedder
        logger.info("Retriever initialized.")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_doc_id: Optional[str] = None,
        filter_source: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieves the top-k most semantically relevant chunks for a query.

        PROCESS:
          1. Validate query is not empty
          2. Embed the query using the same model used for chunks
             (CRITICAL: query and chunks MUST use the same embedding model)
          3. Perform vector similarity search
          4. Log retrieval statistics

        Args:
            query: The user's natural language question.
            top_k: Number of candidates to retrieve. Defaults to settings value.
            filter_doc_id: Optional hash to restrict retrieval to one document.
            filter_source: Optional filename to restrict retrieval.

        Returns:
            List of dicts with keys: chunk_id, text, metadata, distance, score

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k is None:
            top_k = settings.RETRIEVAL_TOP_K

        start_time = time.time()

        # Embed query
        # WHY same model? Embeddings only make sense when compared using
        # vectors from the same embedding space. Mixing models = garbage results.
        query_embedding = self._embedder.embed_query(query)

        # Search vector store
        results = self._store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_doc_id=filter_doc_id,
            filter_source=filter_source,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        if not results:
            logger.warning(
                f"No results found for query: '{query[:80]}...'. "
                "The vector store may be empty or the query is out-of-distribution."
            )
        else:
            logger.info(
                f"Retrieval complete: {len(results)} chunks in {elapsed_ms:.1f}ms | "
                f"Top score: {results[0]['score']:.3f} | "
                f"Query: '{query[:60]}...'"
            )

        return results
