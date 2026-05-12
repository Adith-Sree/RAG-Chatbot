"""
api/routes/query.py — RAG Query & Chat Endpoints
=================================================
ENDPOINTS:
  POST /api/v1/query      — One-shot RAG question answering
  POST /api/v1/chat       — Multi-turn conversational chat
  GET  /api/v1/chunks/{doc_id} — Retrieve all chunks for a document (debugging)

THE FULL RAG PIPELINE (CALLED HERE):
  Query → Embed Query → Vector Search → Rerank → Build Prompt → LLM → Extract Citations → Format Response

WHY SEPARATE QUERY AND CHAT?
  - Query: Stateless, single question → answer
  - Chat: Stateful, maintains conversation history, follow-up questions supported

SESSION MANAGEMENT:
  The chat endpoint accepts history in the request body.
  WHY client-side history? Simpler than server-side sessions.
  WHY NOT server-side? Requires a session store (Redis) and TTL management.
  For a learning project, client-side history is much simpler.
  For production: use Redis-backed sessions for multi-instance deployments.
"""

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.schemas.query import ChatRequest, QueryRequest, QueryResponse
from services.generation.citation_extractor import CitationExtractor
from services.generation.llm_chain import LLMChain
from services.generation.prompt_builder import PromptBuilder
from services.generation.response_formatter import ResponseFormatter
from services.retrieval.reranker import Reranker
from services.retrieval.retriever import Retriever
from utils.config import get_settings
from utils.logger import get_logger

router = APIRouter(tags=["Query"])
logger = get_logger(__name__)
settings = get_settings()


# ─── Dependency Getters ──────────────────────────────────────────────────────

def get_retriever() -> Retriever:
    from app.main import get_app_state
    return get_app_state().retriever


def get_reranker() -> Reranker:
    from app.main import get_app_state
    return get_app_state().reranker


def get_llm_chain() -> LLMChain:
    from app.main import get_app_state
    return get_app_state().llm_chain


# ─── One-shot Query ──────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about ingested seminar reports",
    description="""
    Runs the full RAG pipeline:
    1. Embeds your question
    2. Searches the vector store for relevant chunks
    3. Reranks candidates using a cross-encoder
    4. Constructs a grounded prompt
    5. Generates an answer using GPT-4o-mini
    6. Extracts and validates citations
    7. Returns a structured response with citations and performance metrics
    """,
)
async def query(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    reranker: Reranker = Depends(get_reranker),
    llm_chain: LLMChain = Depends(get_llm_chain),
) -> dict:
    """
    One-shot RAG query. See module docstring for full pipeline description.

    ERROR CASES:
      - 400: Empty question
      - 503: Vector store or LLM unavailable
    """
    return await _run_rag_pipeline(
        question=request.question,
        retriever=retriever,
        reranker=reranker,
        llm_chain=llm_chain,
        top_k=request.top_k,
        top_n=request.top_n,
        filter_doc_id=request.filter_doc_id,
        filter_source=request.filter_source,
        conversation_history=None,
    )


# ─── Streaming Query ─────────────────────────────────────────────────────────

@router.post(
    "/query/stream",
    summary="Stream a RAG answer token-by-token",
    description="Same as /query but streams tokens as they are generated. Use Server-Sent Events.",
)
async def query_stream(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    reranker: Reranker = Depends(get_reranker),
    llm_chain: LLMChain = Depends(get_llm_chain),
):
    """
    Streaming RAG endpoint using Server-Sent Events (SSE).

    WHY SSE?
      SSE is simpler than WebSockets for one-directional streaming.
      Each token is sent as "data: <token>\n\n" — easy to parse client-side.

    USAGE (JavaScript):
        const source = new EventSource('/api/v1/query/stream');
        source.onmessage = (e) => console.log(e.data);
    """
    # Retrieval phase (non-streaming)
    retrieval_start = time.time()
    retrieved = retriever.retrieve(
        query=request.question,
        top_k=request.top_k,
        filter_doc_id=request.filter_doc_id,
    )
    reranked = reranker.rerank(request.question, retrieved, top_n=request.top_n)
    retrieval_latency_ms = (time.time() - retrieval_start) * 1000

    # Build prompt
    builder = PromptBuilder()
    system_prompt = builder.get_system_prompt()
    user_prompt = builder.build_answer_prompt(request.question, reranked)

    # Stream generation
    async def event_stream():
        async for token in llm_chain.generate_stream(system_prompt, user_prompt):
            # SSE format: "data: <content>\n\n"
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for streaming
        },
    )


# ─── Conversational Chat ─────────────────────────────────────────────────────

@router.post(
    "/chat",
    summary="Multi-turn conversational chat",
    description="Maintains conversation context across multiple questions using client-provided history.",
)
async def chat(
    request: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    reranker: Reranker = Depends(get_reranker),
    llm_chain: LLMChain = Depends(get_llm_chain),
) -> dict:
    """
    Conversational chat with history.

    The client provides previous messages in `history`.
    We inject this into the prompt so the LLM can handle follow-ups like:
    - "Can you elaborate on that?"
    - "What does it mean in the context of chapter 2?"

    SESSION ID:
      We echo back the session_id so the client can correlate responses.
      In production, use this to look up server-side session state.
    """
    history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    response = await _run_rag_pipeline(
        question=request.question,
        retriever=retriever,
        reranker=reranker,
        llm_chain=llm_chain,
        conversation_history=history,
        filter_doc_id=request.filter_doc_id,
    )

    # Add session_id to response
    response["session_id"] = request.session_id
    return response


# ─── Debug: List Chunks ──────────────────────────────────────────────────────

@router.get(
    "/chunks/{doc_id}",
    summary="List chunks stored for a document (debugging)",
    description="Returns raw chunks for a document. Useful for inspecting chunking quality.",
)
async def get_chunks(
    doc_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Max chunks to return"),
    retriever: Retriever = Depends(get_retriever),
) -> dict:
    """
    Returns stored chunks for a document. Useful for:
    - Verifying chunking quality during development
    - Debugging why a specific answer was generated
    - Understanding what the model can "see"
    """
    from app.main import get_app_state
    vector_store = get_app_state().vector_store

    try:
        result = vector_store._collection.get(
            where={"doc_id": {"$eq": doc_id}},
            include=["documents", "metadatas"],
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks: {e}")

    chunks = [
        {
            "chunk_id": cid,
            "text": doc[:500] + "..." if len(doc) > 500 else doc,
            "metadata": meta,
        }
        for cid, doc, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    ]

    return {"doc_id": doc_id, "chunk_count": len(chunks), "chunks": chunks}


# ─── Health Check ─────────────────────────────────────────────────────────────

@router.get("/health", tags=["Health"], summary="System health check")
async def health_check() -> dict:
    """
    Returns system health status.
    Checks: vector store connectivity, API key presence.
    """
    checks = {
        "api": "ok",
        "google_api_key": "configured" if settings.GOOGLE_API_KEY else "MISSING",
    }
    try:
        from app.main import get_app_state
        state = get_app_state()
        stats = state.vector_store.get_collection_stats()
        checks["vector_store"] = "ok"
        checks["total_chunks"] = stats.get("total_chunks", 0)
    except Exception as e:
        checks["vector_store"] = f"error: {e}"

    status = "healthy" if all(v not in ("MISSING", ) for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks, "version": settings.APP_VERSION}


# ─── Core Pipeline ────────────────────────────────────────────────────────────

async def _run_rag_pipeline(
    question: str,
    retriever: Retriever,
    reranker: Reranker,
    llm_chain: LLMChain,
    top_k: Optional[int] = None,
    top_n: Optional[int] = None,
    filter_doc_id: Optional[str] = None,
    filter_source: Optional[str] = None,
    conversation_history: Optional[list] = None,
) -> dict:
    """
    The central RAG pipeline function.

    ALL query paths (one-shot, chat, streaming) call this function.
    Centralizing here prevents code duplication and makes changes easier.

    PIPELINE STAGES:
      1. Retrieve   — vector similarity search
      2. Rerank     — cross-encoder precise ranking
      3. Prompt     — inject chunks + question into template
      4. Generate   — LLM produces grounded answer
      5. Cite       — extract + validate citations
      6. Format     — package everything into structured response
    """
    logger.info(f"RAG pipeline started | Q: '{question[:80]}...'")

    # ── Stage 7: Retrieve ────────────────────────────────────────────────
    retrieval_start = time.time()
    retrieved_chunks = retriever.retrieve(
        query=question,
        top_k=top_k,
        filter_doc_id=filter_doc_id,
        filter_source=filter_source,
    )
    retrieval_latency_ms = (time.time() - retrieval_start) * 1000

    # ── Stage 8: Rerank ──────────────────────────────────────────────────
    reranked_chunks = reranker.rerank(
        query=question,
        chunks=retrieved_chunks,
        top_n=top_n,
    )

    # ── Stage 9: Build Prompt ────────────────────────────────────────────
    builder = PromptBuilder()
    system_prompt = builder.get_system_prompt()
    user_prompt = builder.build_answer_prompt(
        question=question,
        chunks=reranked_chunks,
        conversation_history=conversation_history,
    )

    # ── Stage 10: Generate ───────────────────────────────────────────────
    try:
        llm_output = await llm_chain.generate(system_prompt, user_prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

    # ── Stage 11: Extract Citations ──────────────────────────────────────
    extractor = CitationExtractor()
    citations = extractor.extract(
        answer_text=llm_output["answer"],
        retrieved_chunks=reranked_chunks,
    )

    # If no inline citations were found, generate citations from chunks directly
    if not citations and reranked_chunks:
        citations = extractor.extract_from_chunks(reranked_chunks)

    # ── Stage 12: Format Response ────────────────────────────────────────
    formatter = ResponseFormatter()
    response = formatter.format(
        question=question,
        llm_output=llm_output,
        citations=citations,
        reranked_chunks=reranked_chunks,
        retrieved_chunks=retrieved_chunks,
        retrieval_latency_ms=retrieval_latency_ms,
    )

    return response.to_dict()
