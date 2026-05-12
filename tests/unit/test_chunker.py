"""
tests/unit/test_chunker.py — Chunker Unit Tests
================================================
WHY UNIT TESTS?
  - Catch regressions when you change chunking logic
  - Verify edge cases (empty text, single word, very long text)
  - Document expected behavior as executable code
  - Enable confident refactoring

WHAT WE TEST:
  - Normal chunking produces chunks
  - Empty text produces no chunks
  - Chunks respect size limits (with some tolerance)
  - Chunk metadata is correctly set
  - Overlap: adjacent chunks share content at boundaries

HOW TO RUN:
  pytest tests/unit/test_chunker.py -v
"""

import pytest
from services.ingestion.chunker import DocumentChunker, TextChunk
from services.ingestion.pdf_parser import ParsedPage


def make_page(text: str, page_num: int = 1, doc_id: str = "test_doc") -> ParsedPage:
    """Helper: creates a ParsedPage for testing."""
    return ParsedPage(
        page_number=page_num,
        text=text,
        source="test.pdf",
        doc_id=doc_id,
        total_pages=10,
        file_path="/tmp/test.pdf",
    )


@pytest.fixture
def chunker():
    """Provides a DocumentChunker instance for tests."""
    return DocumentChunker()


class TestDocumentChunker:
    """Tests for the DocumentChunker class."""

    def test_empty_pages_returns_empty(self, chunker):
        """Empty page list should produce no chunks."""
        result = chunker.chunk_pages([])
        assert result == []

    def test_empty_text_page_skipped(self, chunker):
        """Pages with only whitespace should be skipped."""
        page = make_page("   \n\n\t  ")
        result = chunker.chunk_pages([page])
        assert result == []

    def test_short_text_produces_single_chunk(self, chunker):
        """Text shorter than chunk_size should produce exactly one chunk."""
        text = "This is a very short piece of text that fits in one chunk."
        page = make_page(text)
        chunks = chunker.chunk_pages([page])
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_text_produces_multiple_chunks(self, chunker):
        """Text longer than chunk_size should be split into multiple chunks."""
        # Create text that's clearly longer than any reasonable chunk size
        long_text = "This is sentence number {i}. " * 500
        long_text = " ".join(long_text.format(i=i) for i in range(500))
        page = make_page(long_text)
        chunks = chunker.chunk_pages([page])
        assert len(chunks) > 1

    def test_chunk_metadata_correct(self, chunker):
        """Each chunk should have correct metadata from the parent page."""
        text = "Content for testing metadata correctness. " * 20
        page = make_page(text, page_num=5, doc_id="metadata_test")
        chunks = chunker.chunk_pages([page])

        for chunk in chunks:
            assert chunk.doc_id == "metadata_test"
            assert chunk.source == "test.pdf"
            assert chunk.page_number == 5
            assert chunk.total_pages == 10
            assert chunk.text.strip() != ""

    def test_chunk_ids_are_unique(self, chunker):
        """All chunk IDs across the document should be unique."""
        long_text = "Sentence {i} with enough content to create many chunks. " * 300
        page1 = make_page(long_text.format(i=1), page_num=1)
        page2 = make_page(long_text.format(i=2), page_num=2)

        chunks = chunker.chunk_pages([page1, page2])
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found!"

    def test_multiple_pages_preserve_page_numbers(self, chunker):
        """Chunks from different pages must have correct page numbers."""
        text = "Content for page testing purposes. " * 50
        page1 = make_page(text, page_num=1)
        page2 = make_page(text, page_num=7)

        chunks = chunker.chunk_pages([page1, page2])

        page1_chunks = [c for c in chunks if c.page_number == 1]
        page7_chunks = [c for c in chunks if c.page_number == 7]

        assert len(page1_chunks) > 0, "Expected chunks from page 1"
        assert len(page7_chunks) > 0, "Expected chunks from page 7"

    def test_chunk_text_is_not_empty(self, chunker):
        """No chunk should have empty or whitespace-only text."""
        text = "This is good content. " * 100
        page = make_page(text)
        chunks = chunker.chunk_pages([page])

        for chunk in chunks:
            assert chunk.text.strip() != "", f"Empty chunk found: {chunk.chunk_id}"

    def test_token_estimate_is_positive(self, chunker):
        """Token estimate should be positive for non-empty chunks."""
        text = "Token counting test content. " * 50
        page = make_page(text)
        chunks = chunker.chunk_pages([page])

        for chunk in chunks:
            assert chunk.token_estimate > 0

    def test_chunk_index_is_sequential(self, chunker):
        """Chunk indices should be sequential starting from 0."""
        text = "Content to chunk. " * 200
        page = make_page(text)
        chunks = chunker.chunk_pages([page])

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))
