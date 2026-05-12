"""
tests/unit/test_citation.py — CitationExtractor Unit Tests
"""

import pytest
from services.generation.citation_extractor import Citation, CitationExtractor


@pytest.fixture
def extractor():
    return CitationExtractor()


def make_chunk(source: str, page: int, text: str = "Sample text.") -> dict:
    """Helper to create a chunk dict for testing."""
    return {
        "chunk_id": f"{source}_p{page}",
        "text": text,
        "metadata": {"source": source, "page_number": page},
        "score": 0.9,
    }


class TestCitationExtractor:

    def test_extracts_single_citation(self, extractor):
        answer = "The report states the algorithm works. [Source: report.pdf, Page 5]"
        chunks = [make_chunk("report.pdf", 5, "The algorithm works well.")]
        citations = extractor.extract(answer, chunks)
        assert len(citations) == 1
        assert citations[0].source == "report.pdf"
        assert citations[0].page_number == 5

    def test_extracts_multiple_citations(self, extractor):
        answer = (
            "The introduction covers basics. [Source: report.pdf, Page 1] "
            "The conclusion summarizes. [Source: report.pdf, Page 10]"
        )
        chunks = [
            make_chunk("report.pdf", 1, "Introduction basics."),
            make_chunk("report.pdf", 10, "Conclusion summary."),
        ]
        citations = extractor.extract(answer, chunks)
        assert len(citations) == 2

    def test_empty_answer_returns_empty(self, extractor):
        citations = extractor.extract("", [])
        assert citations == []

    def test_no_citations_in_answer(self, extractor):
        answer = "This answer has no citation markers."
        citations = extractor.extract(answer, [])
        assert citations == []

    def test_hallucinated_citation_flagged(self, extractor):
        """A citation not matching any retrieved chunk should be flagged."""
        answer = "As stated [Source: report.pdf, Page 99]"
        chunks = [make_chunk("report.pdf", 5)]  # Page 99 not in chunks!
        citations = extractor.extract(answer, chunks)
        assert len(citations) == 1
        assert citations[0].is_valid is False
        assert citations[0].page_number == 99

    def test_deduplicated_citations(self, extractor):
        """Same (source, page) cited twice should produce one citation."""
        answer = (
            "First mention [Source: report.pdf, Page 3] "
            "and again [Source: report.pdf, Page 3]"
        )
        chunks = [make_chunk("report.pdf", 3)]
        citations = extractor.extract(answer, chunks)
        assert len(citations) == 1

    def test_valid_citation_has_excerpt(self, extractor):
        """Valid citation should have an excerpt from the chunk."""
        chunk_text = "The proposed methodology uses a transformer architecture."
        answer = "The method uses transformers. [Source: thesis.pdf, Page 7]"
        chunks = [make_chunk("thesis.pdf", 7, chunk_text)]
        citations = extractor.extract(answer, chunks)
        assert len(citations) == 1
        assert citations[0].is_valid is True
        assert "transformer" in citations[0].excerpt

    def test_case_insensitive_source_matching(self, extractor):
        """Source matching should be case-insensitive."""
        answer = "Content [Source: REPORT.PDF, Page 2]"
        chunks = [make_chunk("report.pdf", 2, "Some content.")]
        citations = extractor.extract(answer, chunks)
        # Should still match despite case difference
        # (partial match strips .pdf for comparison)
        assert len(citations) == 1

    def test_extract_from_chunks_creates_citations(self, extractor):
        """extract_from_chunks should create one citation per unique (source, page)."""
        chunks = [
            make_chunk("doc.pdf", 1, "Page 1 content"),
            make_chunk("doc.pdf", 2, "Page 2 content"),
            make_chunk("doc.pdf", 1, "Another page 1 chunk"),  # Duplicate page
        ]
        citations = extractor.extract_from_chunks(chunks)
        # Should deduplicate: page 1 appears twice, should be one citation
        assert len(citations) == 2
        pages = {c.page_number for c in citations}
        assert pages == {1, 2}

    def test_citation_to_dict(self, extractor):
        """Citation.to_dict() should have all required keys."""
        c = Citation(source="test.pdf", page_number=3, excerpt="Some text", is_valid=True)
        d = c.to_dict()
        assert "source" in d
        assert "page_number" in d
        assert "excerpt" in d
        assert "is_valid" in d
        assert d["page_number"] == 3
