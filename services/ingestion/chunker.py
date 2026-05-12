"""
chunker.py — Stage 4: Text Chunking
=====================================
WHY CHUNKING EXISTS:
  Language models have limited context windows. More importantly, vector
  similarity search works best on focused, self-contained units of text.

  PROBLEM with using full documents:
    - "What is the conclusion?" would retrieve the whole 40-page report,
      most of which is irrelevant noise.
    - Embedding a 40-page report produces ONE vector that averages all topics,
      making precise retrieval impossible.

  SOLUTION — chunking:
    - Split documents into focused segments (chunks)
    - Each chunk gets its OWN embedding vector
    - Retrieval finds the EXACT section that answers the question

WHY CHUNK SIZE MATTERS:
  - Too small (< 200 tokens): Chunks lose context. "The algorithm..." becomes
    meaningless without knowing which algorithm.
  - Too large (> 1500 tokens): Chunk embedding averages too many topics.
    Retrieval matches but relevant info is buried.
  - SWEET SPOT: 500–1000 tokens preserves context while staying focused.

WHY OVERLAP MATTERS:
  - A key sentence might fall exactly at a chunk boundary, getting split.
  - Overlap (100–200 tokens) duplicates boundary regions in adjacent chunks.
  - This ensures no important information is lost at boundaries.
  - TRADEOFF: Overlap increases storage and embedding cost.

CHUNKING STRATEGIES:

  1. RECURSIVE CHARACTER SPLITTING (Default):
     - Splits on: paragraphs → sentences → words → characters
     - Tries to keep sentences intact
     - Fast and reliable, no ML required
     - Best for: well-structured academic reports

  2. SEMANTIC CHUNKING (Premium):
     - Uses embeddings to detect semantic boundaries
     - Groups sentences with similar meaning together
     - Produces more coherent chunks
     - COST: One embedding API call per sentence (expensive!)
     - Best for: unstructured documents, creative writing

PRODUCTION CONSIDERATIONS:
  - For very large docs, chunk in background workers (Celery/RQ).
  - Consider token-based splitting (tiktoken) over character splitting
    for more accurate chunk sizes with LLMs.
  - Experiment with chunk sizes and overlap for your specific document types.
"""

from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

from services.ingestion.pdf_parser import ParsedPage
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class TextChunk:
    """
    Represents a single text chunk ready for embedding.

    WHY A DATACLASS?
      Explicit field names prevent bugs from dict key typos.
      Type hints enable IDE auto-completion and static analysis.
    """
    chunk_id: str        # Unique ID: "{doc_id}_p{page_num}_c{chunk_idx}"
    text: str            # The chunk text content
    doc_id: str          # Parent document hash
    source: str          # Original filename
    page_number: int     # Source page number
    total_pages: int     # Total pages in document
    file_path: str       # Absolute path to source file
    chunk_index: int     # Position within the document
    token_estimate: int  # Rough token count (char_count / 4)

    def to_metadata(self) -> dict:
        """Flat metadata dict for ChromaDB storage."""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "page_number": self.page_number,
            "total_pages": self.total_pages,
            "file_path": self.file_path,
            "chunk_index": self.chunk_index,
            "token_estimate": self.token_estimate,
        }


class DocumentChunker:
    """
    Splits parsed PDF pages into overlapping text chunks.

    DESIGN:
      - Primary strategy: RecursiveCharacterTextSplitter (reliable, fast)
      - Optional strategy: SemanticChunker (better quality, expensive)
      - Strategy is configured globally but can be overridden per call

    USAGE:
        chunker = DocumentChunker()
        pages = parser.parse("report.pdf", doc_id="abc")
        chunks = chunker.chunk_pages(pages)
    """

    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.strategy = settings.CHUNKING_STRATEGY

        # Initialize the recursive splitter (always available)
        # WHY these separators? We try to split at paragraph breaks first,
        # then sentence breaks, then word breaks, then characters (last resort).
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,       # Convert tokens → chars (≈4 chars/token)
            chunk_overlap=self.chunk_overlap * 4,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"DocumentChunker initialized: strategy={self.strategy}, "
            f"chunk_size={self.chunk_size} tokens, overlap={self.chunk_overlap} tokens"
        )

    def chunk_pages(self, pages: list[ParsedPage]) -> list[TextChunk]:
        """
        Chunks all pages of a document into TextChunk objects.

        IMPLEMENTATION NOTE:
          We chunk per-page, not the whole document at once.
          WHY? This preserves page number metadata accurately.
          If we concatenated everything and split, we'd lose which page each chunk came from.

        Args:
            pages: List of ParsedPage objects from the PDF parser.

        Returns:
            List of TextChunk objects across all pages.
        """
        if not pages:
            logger.warning("No pages provided to chunk.")
            return []

        all_chunks: list[TextChunk] = []
        global_chunk_idx = 0

        for page in pages:
            if not page.text.strip():
                logger.debug(f"Skipping empty page {page.page_number}")
                continue

            page_chunks = self._chunk_page(page, global_chunk_idx)
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        logger.info(
            f"Chunking complete: {len(pages)} pages → {len(all_chunks)} chunks "
            f"(avg {len(all_chunks)/len(pages):.1f} chunks/page)"
        )
        return all_chunks

    def _chunk_page(self, page: ParsedPage, start_idx: int) -> list[TextChunk]:
        """
        Chunks a single page's text.

        Args:
            page: The parsed page to chunk.
            start_idx: Global chunk index offset for this page.

        Returns:
            List of TextChunk objects for this page.
        """
        if self.strategy == "semantic":
            try:
                raw_chunks = self._semantic_split(page.text)
            except Exception as e:
                logger.warning(
                    f"Semantic chunking failed on page {page.page_number}: {e}. "
                    "Falling back to recursive chunking."
                )
                raw_chunks = self._recursive_split(page.text)
        else:
            raw_chunks = self._recursive_split(page.text)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk = TextChunk(
                chunk_id=f"{page.doc_id}_p{page.page_number}_c{start_idx + i}",
                text=chunk_text,
                doc_id=page.doc_id,
                source=page.source,
                page_number=page.page_number,
                total_pages=page.total_pages,
                file_path=page.file_path,
                chunk_index=start_idx + i,
                token_estimate=len(chunk_text) // 4,  # Rough: 4 chars ≈ 1 token
            )
            chunks.append(chunk)

        return chunks

    def _recursive_split(self, text: str) -> list[str]:
        """
        Splits text using LangChain's RecursiveCharacterTextSplitter.

        ALGORITHM:
          1. Try to split on paragraph boundaries (\n\n)
          2. If chunks are still too large, split on newlines (\n)
          3. If still too large, split on sentence endings (". ")
          4. If still too large, split on spaces (word boundaries)
          5. Last resort: split on individual characters

        This "recursive" approach preserves semantic units as long as possible.
        """
        return self._recursive_splitter.split_text(text)

    def _semantic_split(self, text: str) -> list[str]:
        """
        Splits text using local embedding-based semantic boundaries.

        HOW IT WORKS:
          1. Splits text into sentences
          2. Embeds each sentence using the local HuggingFace model
          3. Finds divergence points where topic changes occur
          4. Those points become chunk boundaries

        COST: CPU compute per sentence (no API cost).
        BENEFIT: Chunk boundaries align with topic changes, not character counts.
        """
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )
        return splitter.split_text(text)
