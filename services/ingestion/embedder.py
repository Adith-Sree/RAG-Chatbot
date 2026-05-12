"""
embedder.py — Stage 5: Embedding Generation (Local HuggingFace Embeddings)
===========================================================================
WHY LOCAL EMBEDDINGS?
  Google's OpenAI-compatible endpoint does not support the /embeddings API
  (returns 501 UNIMPLEMENTED). Rather than fighting API compatibility issues,
  we use local sentence-transformers — which is already installed as a
  dependency of the reranker.

BENEFITS OF LOCAL EMBEDDINGS:
  - Zero API cost (completely free)
  - No rate limits on embedding calls
  - Deterministic — same text always produces the same vector
  - Faster for large batches (no network round-trip)
  - Works offline

MODEL: all-MiniLM-L6-v2
  - 384-dimensional vectors
  - ~22MB model download (cached in ~/.cache/huggingface/)
  - Fast inference on CPU: ~14ms per sentence
  - Strong semantic similarity for English academic text
  - Widely used, well-tested for RAG retrieval

NOTE ON DIMENSIONS:
  ChromaDB collection must use consistent dimensions. If you switch embedding
  models, delete ./vector_store/ and re-ingest all documents.

PRODUCTION CONSIDERATIONS:
  - For higher-quality embeddings: use BAAI/bge-base-en-v1.5 (768 dims)
  - For GPU inference: set model_kwargs={'device': 'cuda'}
  - Cache embeddings by chunk hash to avoid re-embedding unchanged documents
"""

import time

from langchain_community.embeddings import HuggingFaceEmbeddings

from services.ingestion.chunker import TextChunk
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingGenerator:
    """
    Generates vector embeddings for text chunks using a local HuggingFace model.

    USAGE:
        embedder = EmbeddingGenerator()
        embedded_chunks = embedder.embed_chunks(chunks)
        query_vector   = embedder.embed_query("What is the main finding?")
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL} (local, no API)")
        self._embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # Cosine similarity works best with normalised vectors
        )
        logger.info(
            f"EmbeddingGenerator ready: model={settings.EMBEDDING_MODEL}, "
            f"dims={settings.EMBEDDING_DIMENSIONS}"
        )

    def embed_chunks(self, chunks: list[TextChunk]) -> list[tuple[TextChunk, list[float]]]:
        """
        Embeds a list of TextChunks and returns (chunk, embedding_vector) pairs.

        Args:
            chunks: List of TextChunk objects to embed.

        Returns:
            List of (TextChunk, embedding_vector) tuples.
        """
        if not chunks:
            return []

        # Filter out empty chunks
        valid_chunks = [c for c in chunks if c.text.strip()]
        skipped = len(chunks) - len(valid_chunks)
        if skipped:
            logger.warning(f"Skipped {skipped} empty chunk(s).")

        if not valid_chunks:
            return []

        logger.info(f"Embedding {len(valid_chunks)} chunks locally...")
        start_time = time.time()

        texts = [c.text for c in valid_chunks]

        try:
            embeddings = self._embeddings_model.embed_documents(texts)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Embedded {len(valid_chunks)} chunks in {latency_ms:.0f}ms "
            f"({latency_ms / len(valid_chunks):.1f}ms/chunk avg)"
        )

        return list(zip(valid_chunks, embeddings))

    def embed_query(self, query: str) -> list[float]:
        """
        Embeds a single query string for similarity search.

        Args:
            query: The user's question string.

        Returns:
            Embedding vector as list of floats (384 dims for all-MiniLM-L6-v2).
        """
        if not query.strip():
            raise ValueError("Cannot embed empty query string.")

        logger.debug(f"Embedding query: '{query[:80]}...'")
        return self._embeddings_model.embed_query(query)
