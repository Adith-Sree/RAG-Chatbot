"""
response_formatter.py — Stage 12: Response Formatting
=======================================================
WHY THIS EXISTS:
  The final stage packages everything into a consistent, structured response.

  WHY CONSISTENT STRUCTURE MATTERS:
    - The frontend knows exactly what fields to expect — no guessing
    - The API is self-documenting (the structure IS the documentation)
    - Future versions can add fields without breaking existing clients
    - Logging and analytics can process structured data easily

  A "bag of results" from the LLM is not a production API response.
  We need: answer text, citations, metadata, performance metrics, and source chunks.

PRODUCTION CONSIDERATIONS:
  - Add a response_id for tracking (UUID)
  - Add user_id for per-user analytics
  - Store responses in a database for audit trail
  - Implement response caching for identical queries
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from services.generation.citation_extractor import Citation
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """
    The final structured response from the RAG pipeline.

    This is what gets serialized to JSON and returned to the client.
    Every field serves a purpose — no unnecessary data.
    """
    response_id: str                      # Unique ID for this response
    question: str                         # The original question
    answer: str                           # The generated answer text
    citations: list[dict]                 # Validated citations with source info
    chunks_used: list[dict]               # The actual chunks sent to the LLM
    total_chunks_retrieved: int           # Candidates before reranking
    input_tokens: int                     # LLM input token count
    output_tokens: int                    # LLM output token count
    estimated_cost_usd: float             # Approximate API cost
    retrieval_latency_ms: float           # Time to retrieve chunks
    generation_latency_ms: float          # Time for LLM to generate
    total_latency_ms: float               # End-to-end latency
    model_used: str                       # LLM model name
    hallucinated_citations: int = 0       # Count of unverifiable citations
    has_sufficient_context: bool = True   # False if no chunks were retrieved

    def to_dict(self) -> dict:
        """Serializes the response to a plain dict for JSON output."""
        return {
            "response_id": self.response_id,
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "chunks_used": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "text": c.get("text", "")[:300] + "...",  # Truncate for readability
                    "source": c.get("metadata", {}).get("source"),
                    "page_number": c.get("metadata", {}).get("page_number"),
                    "reranker_score": round(c.get("reranker_score", c.get("score", 0)), 4),
                }
                for c in self.chunks_used
            ],
            "metadata": {
                "total_chunks_retrieved": self.total_chunks_retrieved,
                "chunks_after_reranking": len(self.chunks_used),
                "has_sufficient_context": self.has_sufficient_context,
                "hallucinated_citations": self.hallucinated_citations,
                "model_used": self.model_used,
            },
            "performance": {
                "retrieval_latency_ms": round(self.retrieval_latency_ms, 1),
                "generation_latency_ms": round(self.generation_latency_ms, 1),
                "total_latency_ms": round(self.total_latency_ms, 1),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            },
        }


class ResponseFormatter:
    """
    Packages all pipeline outputs into a clean, structured RAGResponse.

    USAGE:
        formatter = ResponseFormatter()
        response = formatter.format(
            question=question,
            llm_output=llm_output,
            citations=citations,
            reranked_chunks=final_chunks,
            retrieved_chunks=raw_chunks,
            retrieval_latency_ms=45.2,
        )
        return response.to_dict()
    """

    def format(
        self,
        question: str,
        llm_output: dict,
        citations: list[Citation],
        reranked_chunks: list[dict],
        retrieved_chunks: list[dict],
        retrieval_latency_ms: float,
    ) -> RAGResponse:
        """
        Creates a RAGResponse from all pipeline outputs.

        Args:
            question: Original user question.
            llm_output: Dict from LLMChain.generate() with answer + token counts.
            citations: List of Citation objects from CitationExtractor.
            reranked_chunks: Final chunks after reranking (sent to LLM).
            retrieved_chunks: Initial candidates from vector search.
            retrieval_latency_ms: Time taken for vector search.

        Returns:
            RAGResponse dataclass ready for serialization.
        """
        answer = llm_output.get("answer", "")
        generation_latency_ms = llm_output.get("latency_ms", 0)
        total_latency_ms = retrieval_latency_ms + generation_latency_ms

        # Count hallucinated citations (important quality metric)
        hallucinated = sum(1 for c in citations if not c.is_valid)

        # Detect "no context" case
        has_context = bool(reranked_chunks)

        logger.info(
            f"RAGResponse formatted: {len(citations)} citations, "
            f"{len(reranked_chunks)} chunks, {total_latency_ms:.0f}ms total"
        )

        return RAGResponse(
            response_id=str(uuid.uuid4()),
            question=question,
            answer=answer,
            citations=[c.to_dict() for c in citations],
            chunks_used=reranked_chunks,
            total_chunks_retrieved=len(retrieved_chunks),
            input_tokens=llm_output.get("input_tokens", 0),
            output_tokens=llm_output.get("output_tokens", 0),
            estimated_cost_usd=llm_output.get("estimated_cost_usd", 0.0),
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            model_used=settings_model(),
            hallucinated_citations=hallucinated,
            has_sufficient_context=has_context,
        )


def settings_model() -> str:
    """Returns the configured LLM model name."""
    from utils.config import get_settings
    return get_settings().LLM_MODEL
