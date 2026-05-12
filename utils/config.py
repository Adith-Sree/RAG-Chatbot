"""
config.py — Central Configuration Module
=========================================
WHY THIS EXISTS:
  All configuration lives here. No magic strings scattered across code.
  Junior devs: NEVER hardcode API keys. Always use environment variables.

HOW IT WORKS:
  pydantic-settings reads values from .env file AND environment variables.
  Environment variables always take priority over .env values.
  This makes the same code work locally (using .env) and in Docker/prod
  (using real env vars injected by the platform).

PRODUCTION CONSIDERATIONS:
  - In production, use a secrets manager (AWS Secrets Manager, GCP Secret Manager).
  - Never commit .env to git. It's in .gitignore.
  - Use separate .env files per environment: .env.dev, .env.staging, .env.prod
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings.
    All values can be overridden by environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "Seminar RAG Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ─── API Keys ─────────────────────────────────────────────────────────────
    # CRITICAL: Never hardcode these. Set them in .env or environment.
    GOOGLE_API_KEY: str = ""

    # Google's OpenAI-compatible endpoint — lets us use the OpenAI SDK with Gemini models.
    # This means no langchain-google-genai version conflicts, no v1beta/v1 issues.
    GEMINI_API_BASE: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # ─── LLM Configuration ────────────────────────────────────────────────────
    LLM_MODEL: str = "gemini-2.0-flash-lite"   # Fast, free-tier via OpenAI-compat endpoint
    LLM_TEMPERATURE: float = 0.1          # Low temperature = more deterministic, fewer hallucinations
    LLM_MAX_TOKENS: int = 2048
    LLM_STREAMING: bool = True

    # ─── Embedding Configuration ──────────────────────────────────────────────
    # Local HuggingFace model — no API needed, zero cost, works offline
    # Google's OpenAI-compat endpoint does NOT support embeddings (501 error)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # 384 dims, fast CPU inference
    EMBEDDING_DIMENSIONS: int = 384
    EMBEDDING_BATCH_SIZE: int = 100        # Max chunks to embed in one API call

    # ─── Chunking Configuration ───────────────────────────────────────────────
    # WHY 800 tokens? Large enough to preserve context, small enough for precise retrieval.
    # WHY 150 overlap? Prevents losing information at chunk boundaries.
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    # Strategy: "semantic" uses embedding-based boundaries; "recursive" is pure text splitting.
    CHUNKING_STRATEGY: Literal["semantic", "recursive"] = "recursive"

    # ─── Retrieval Configuration ──────────────────────────────────────────────
    # WHY top-k=20? Retrieve many candidates so the reranker has enough to work with.
    # WHY top-n=5? After reranking, keep only the most relevant to avoid context pollution.
    RETRIEVAL_TOP_K: int = 20              # Candidates fetched from vector DB
    RERANKER_TOP_N: int = 5               # Final chunks sent to LLM after reranking
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ─── Vector Store Configuration ───────────────────────────────────────────
    VECTOR_STORE_TYPE: Literal["chroma", "pinecone", "qdrant"] = "chroma"
    CHROMA_PERSIST_DIR: str = "./vector_store"
    CHROMA_COLLECTION_NAME: str = "seminar_reports"

    # ─── File Upload Configuration ────────────────────────────────────────────
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list[str] = [".pdf"]

    # ─── Paths ────────────────────────────────────────────────────────────────
    PROMPTS_DIR: str = "./prompts"

    # ─── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:8501", "http://localhost:3000"]

    # ─── Rate Limiting ────────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 30

    @property
    def upload_path(self) -> Path:
        p = Path(self.UPLOAD_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def vector_store_path(self) -> Path:
        p = Path(self.CHROMA_PERSIST_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    lru_cache ensures we only read the .env file once.
    Usage: from utils.config import get_settings; settings = get_settings()
    """
    return Settings()
