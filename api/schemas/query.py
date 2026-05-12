"""
schemas/query.py — Pydantic Models for Query & Chat API
"""

from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """One-shot RAG query request."""
    question: str = Field(..., min_length=5, max_length=1000, description="User's question")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Candidates to retrieve")
    top_n: Optional[int] = Field(default=None, ge=1, le=10, description="Chunks to use after reranking")
    filter_doc_id: Optional[str] = Field(default=None, description="Restrict search to a specific document")
    filter_source: Optional[str] = Field(default=None, description="Restrict search by filename")


class CitationModel(BaseModel):
    """A single citation reference."""
    source: str
    page_number: int
    excerpt: str
    chunk_id: Optional[str] = None
    is_valid: bool = True


class ChunkUsed(BaseModel):
    """Summary of a chunk sent to the LLM."""
    chunk_id: Optional[str]
    text: str  # Truncated
    source: Optional[str]
    page_number: Optional[int]
    reranker_score: float


class QueryMetadata(BaseModel):
    """Metadata about the query and retrieval."""
    total_chunks_retrieved: int
    chunks_after_reranking: int
    has_sufficient_context: bool
    hallucinated_citations: int
    model_used: str


class QueryPerformance(BaseModel):
    """Performance metrics for the query."""
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class QueryResponse(BaseModel):
    """Full RAG query response."""
    response_id: str
    question: str
    answer: str
    citations: list[CitationModel]
    chunks_used: list[ChunkUsed]
    metadata: QueryMetadata
    performance: QueryPerformance


class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Multi-turn conversational chat request."""
    question: str = Field(..., min_length=2, max_length=1000)
    session_id: str = Field(..., description="Session identifier for conversation continuity")
    history: list[ChatMessage] = Field(default=[], description="Previous conversation messages")
    filter_doc_id: Optional[str] = Field(default=None)
