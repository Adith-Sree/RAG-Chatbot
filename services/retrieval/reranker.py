"""
reranker.py — Stage 8: Cross-Encoder Reranking
================================================
WHY RERANKING EXISTS — THE CORE PROBLEM:
  Dense retrieval (embedding similarity) is excellent at finding broadly
  relevant chunks. But it has a fundamental limitation:

  ENCODING PROBLEM:
    A bi-encoder (like OpenAI embeddings) encodes the QUERY and each DOCUMENT
    INDEPENDENTLY, then compares them. The query and document never "see" each
    other during encoding.

    This is fast (O(n) with ANN search) but loses precision for nuanced queries.

  CONTEXT POLLUTION:
    Without reranking, we might feed the LLM 20 chunks where:
    - 5 chunks directly answer the question ✓
    - 10 chunks are topically related but don't answer it ✗
    - 5 chunks are weakly related noise ✗

    The LLM wastes its context window on irrelevant chunks and may
    hallucinate by trying to connect unrelated information.

RERANKING SOLUTION — CROSS-ENCODERS:
  A cross-encoder receives BOTH the query and document simultaneously:
    score = cross_encoder(query, document)

  This allows the model to attend to interactions between them.
  Result: much more accurate relevance scoring.

  TRADEOFF:
    Cross-encoders are O(n) — they must score every (query, chunk) pair.
    This is why we use them on 20 candidates (not 10,000+).
    The bi-encoder handles the initial broad search; the cross-encoder
    does precise ranking on the shortlist.

RETRIEVER vs RERANKER SUMMARY:
  ┌────────────────┬──────────────────────┬──────────────────────┐
  │                │ Bi-encoder (Retriever)│ Cross-encoder (Reranker)│
  ├────────────────┼──────────────────────┼──────────────────────┤
  │ Speed          │ Very fast (ANN)       │ Slower (linear scan) │
  │ Accuracy       │ Good (broad match)    │ Excellent (precise)  │
  │ Scalability    │ Millions of docs      │ Tens to hundreds     │
  │ Use case       │ Initial retrieval     │ Final ranking        │
  └────────────────┴──────────────────────┴──────────────────────┘

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking dataset
  - 22M parameters (tiny, runs on CPU)
  - Excellent performance for document retrieval tasks
  - Free, local, no API calls required

PRODUCTION CONSIDERATIONS:
  - Cache the cross-encoder model in memory (loaded once at startup).
  - For >100 candidates, GPU inference is needed for acceptable latency.
  - Consider Cohere's rerank API as a managed alternative.
  - BGE-Reranker-Large gives even better quality if you have GPU.
"""

import time
from typing import Optional

from sentence_transformers import CrossEncoder

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class Reranker:
    """
    Reranks retrieved chunks using a cross-encoder model.

    The cross-encoder jointly encodes (query, chunk) pairs and scores
    them for relevance. This is far more accurate than bi-encoder similarity
    but requires evaluating each pair individually.

    USAGE:
        reranker = Reranker()
        top_chunks = reranker.rerank(query="What is the methodology?", chunks=retrieval_results, top_n=5)
    """

    def __init__(self):
        model_name = settings.RERANKER_MODEL
        logger.info(f"Loading cross-encoder reranker: {model_name}")

        try:
            self._model = CrossEncoder(model_name, max_length=512)
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load reranker model '{model_name}': {e}")
            logger.warning("Reranking will be SKIPPED. Returning raw retrieval results.")
            self._model = None

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: Optional[int] = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Reranks retrieval results and returns the top-n most relevant chunks.

        PROCESS:
          1. Create (query, chunk_text) pairs for every retrieved chunk
          2. Score all pairs simultaneously with the cross-encoder
          3. Sort by score (descending)
          4. Filter by minimum score threshold
          5. Return top-n chunks

        Args:
            query: The user's original question.
            chunks: List of retrieval results from the Retriever.
            top_n: Number of chunks to return after reranking. Defaults to settings value.
            min_score: Minimum reranker score to include a chunk. Filters very weak matches.

        Returns:
            List of reranked chunks (subset of input), sorted by relevance.
            Each dict has the original keys + "reranker_score".

        NOTE ON SCORES:
          CrossEncoder scores are raw logits (-inf to +inf).
          Higher = more relevant. There is no strict bound.
          Typical values: strong match ≈ 5-10, weak match ≈ -5 to 0.
        """
        if not chunks:
            logger.warning("No chunks provided to reranker.")
            return []

        if top_n is None:
            top_n = settings.RERANKER_TOP_N

        start_time = time.time()
        logger.debug(f"Reranking {len(chunks)} candidates → top {top_n}")

        # Prepare (query, text) pairs
        if self._model is None:
            logger.warning("Reranker model not available. Returning top-n retrieval results without reranking.")
            for chunk in chunks:
                chunk["reranker_score"] = chunk.get("score", 0.0)
            return chunks[:top_n]

        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs at once (more efficient than one-by-one)
        scores = self._model.predict(pairs)

        # Attach scores to original chunk dicts
        for chunk, score in zip(chunks, scores):
            chunk["reranker_score"] = float(score)

        # Sort by reranker score (highest first)
        reranked = sorted(chunks, key=lambda c: c["reranker_score"], reverse=True)

        # Filter by minimum score
        if min_score > 0:
            before_filter = len(reranked)
            reranked = [c for c in reranked if c["reranker_score"] >= min_score]
            logger.debug(f"Score filter removed {before_filter - len(reranked)} weak chunks")

        # Keep top-n
        final_chunks = reranked[:top_n]

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Reranking complete: {len(chunks)} → {len(final_chunks)} chunks in {elapsed_ms:.1f}ms | "
            f"Top score: {final_chunks[0]['reranker_score']:.3f}" if final_chunks else "No chunks passed filter"
        )

        return final_chunks
