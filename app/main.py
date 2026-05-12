"""
app/main.py — FastAPI Application Factory
==========================================
WHY AN APPLICATION FACTORY?
  The "application factory" pattern centralizes startup logic.
  WHY NOT global singletons?
    - Testing becomes impossible (can't reset state between tests)
    - Circular imports (module A imports module B imports module A)
    - Hidden dependencies (hard to see what a module needs)

  With the factory pattern:
    - All dependencies are initialized in ONE place (lifespan)
    - Services are injected via FastAPI's Depends() system
    - Testing can create fresh app instances with mocked services

STARTUP EVENTS (lifespan):
  Modern FastAPI uses the `lifespan` context manager (not @app.on_event).
  WHY? It ensures cleanup code (vector store flush, model unload) runs on shutdown.

  On startup:
    1. Validate environment (OPENAI_API_KEY present?)
    2. Create upload and vector store directories
    3. Initialize VectorStore (loads ChromaDB)
    4. Initialize EmbeddingGenerator (configures OpenAI client)
    5. Initialize Reranker (loads cross-encoder model from HuggingFace)
    6. Initialize LLMChain (configures ChatOpenAI)
    7. Initialize Retriever (wires embedder + vector store)

  On shutdown:
    - Log graceful shutdown (extend with resource cleanup as needed)

MIDDLEWARE:
  - CORS: Allow frontend at localhost:8501 (Streamlit) to call the API
  - Logging middleware: Log every request/response

PRODUCTION CONSIDERATIONS:
  - Use Gunicorn + Uvicorn workers for multi-process deployment
  - Add rate limiting middleware (slowapi)
  - Add request ID middleware for distributed tracing
  - Add Prometheus metrics middleware for monitoring
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.routes import ingest as ingest_router
from api.routes import query as query_router
from services.generation.llm_chain import LLMChain
from services.ingestion.embedder import EmbeddingGenerator
from services.retrieval.reranker import Reranker
from services.retrieval.retriever import Retriever
from services.retrieval.vector_store import VectorStore
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class AppState:
    """
    Holds all initialized service instances.

    WHY A DATACLASS?
      Type safety. FastAPI's `app.state` is an untyped namespace.
      Wrapping it in a typed dataclass gives IDE auto-completion
      and catches attribute name typos at type-check time.
    """
    vector_store: VectorStore
    embedder: EmbeddingGenerator
    reranker: Reranker
    llm_chain: LLMChain
    retriever: Retriever


_app_state: AppState | None = None


def get_app_state() -> AppState:
    """Returns the initialized application state."""
    if _app_state is None:
        raise RuntimeError("App state not initialized. Did startup complete?")
    return _app_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown.

    Everything before `yield` runs on startup.
    Everything after `yield` runs on shutdown.

    IMPORTANT: Heavy models (cross-encoder, ChromaDB) load ONCE here,
    not on every request. This is crucial for performance.
    """
    global _app_state

    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Validate critical configuration
    if not settings.GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY is not set! Check your .env file.")
        raise RuntimeError("GOOGLE_API_KEY is required.")

    # Create required directories
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory: {settings.upload_path}")
    logger.info(f"Vector store directory: {settings.vector_store_path}")

    # Initialize services
    # ORDER MATTERS: vector_store and embedder must exist before retriever
    logger.info("Initializing Vector Store...")
    vector_store = VectorStore()

    logger.info("Initializing Embedding Generator...")
    embedder = EmbeddingGenerator()

    logger.info("Loading Reranker model (may take 10-30s first time)...")
    reranker = Reranker()

    logger.info("Initializing LLM Chain...")
    llm_chain = LLMChain()

    logger.info("Initializing Retriever...")
    retriever = Retriever(vector_store=vector_store, embedder=embedder)

    _app_state = AppState(
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        llm_chain=llm_chain,
        retriever=retriever,
    )

    logger.info(f"✓ All services initialized. {settings.APP_NAME} is ready.")

    yield  # App is running here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down... Performing cleanup.")
    # ChromaDB persists automatically; no explicit flush needed.
    # Add cleanup here: close database connections, stop background workers, etc.
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="""
        ## Seminar Report RAG Assistant

        A production-grade Retrieval-Augmented Generation system for querying seminar reports.

        ### Features
        - PDF upload and ingestion
        - Semantic search with embedding-based retrieval
        - Cross-encoder reranking for precision
        - Grounded answer generation with citations
        - Multi-turn conversational chat
        - Streaming responses

        ### Quick Start
        1. `POST /api/v1/ingest/upload` — Upload a PDF
        2. `POST /api/v1/ingest/process/{doc_id}` — Ingest and embed it
        3. `POST /api/v1/query` — Ask questions!
        """,
        lifespan=lifespan,
        docs_url="/docs",        # Swagger UI
        redoc_url="/redoc",      # ReDoc UI
        openapi_url="/openapi.json",
    )

    # ── Middleware ─────────────────────────────────────────────────────────────

    # CORS: Allow frontend (Streamlit at 8501, React at 3000) to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # GZip: Compress large responses (chunks list, full document metadata)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Routers ────────────────────────────────────────────────────────────────
    API_PREFIX = "/api/v1"
    app.include_router(ingest_router.router, prefix=API_PREFIX)
    app.include_router(query_router.router, prefix=API_PREFIX)

    @app.get("/", tags=["Root"])
    async def root():
        """API root — returns basic info and links to docs."""
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# Create the application instance
app = create_app()
