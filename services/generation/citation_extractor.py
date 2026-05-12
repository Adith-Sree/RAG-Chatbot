"""
citation_extractor.py — Stage 11: Citation Extraction & Validation
====================================================================
WHY CITATIONS MATTER:
  In a RAG system, citations are the TRUST LAYER between the AI's answer
  and the original source documents. They allow users to:
  1. Verify the AI's answer against the original text
  2. Find the exact page for deeper reading
  3. Detect hallucinations (if a cited page doesn't contain the claimed info)

  Without citations, RAG answers are just "AI says so" — not useful for
  academic or professional contexts.

HOW CITATIONS ARE GENERATED:
  STEP 1 (In PromptBuilder): We inject chunk metadata into the prompt:
    "[CHUNK 1 | Source: report.pdf | Page: 5]"

  STEP 2 (In LLMChain): The LLM generates an answer with inline citations:
    "...the algorithm achieves 95% accuracy [Source: report.pdf, Page 5]..."

  STEP 3 (Here): We extract those citation markers from the answer text
    and validate them against the retrieved chunks.

VALIDATION — WHY IT'S CRITICAL:
  LLMs can "hallucinate" citations — they might write "[Source: report.pdf, Page 99]"
  even if page 99 was never in the retrieved context.

  SOLUTION: Cross-reference every extracted citation against the actual chunks
  we sent to the LLM. If a citation isn't in the retrieved set, flag it.

COMMON FAILURE POINTS:
  - LLM uses different citation format than instructed (regex may miss it)
  - LLM fabricates page numbers (validation catches this)
  - Same source cited multiple times with different formats (deduplication needed)

PRODUCTION CONSIDERATIONS:
  - Use more structured output (function calling / JSON mode) for reliable citation format
  - Store citations in a database for audit trail
  - Build a citation verification UI that highlights the cited text in the PDF
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Citation:
    """
    Represents a validated citation from the LLM's answer.

    FIELDS:
        source: Document filename (e.g., "seminar_report.pdf")
        page_number: Page number from the document
        excerpt: Brief excerpt from the chunk that was cited
        chunk_id: ID of the chunk this citation maps to
        is_valid: True if this citation maps to an actual retrieved chunk
    """
    source: str
    page_number: int
    excerpt: str
    chunk_id: Optional[str] = None
    is_valid: bool = True

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "page_number": self.page_number,
            "excerpt": self.excerpt,
            "chunk_id": self.chunk_id,
            "is_valid": self.is_valid,
        }


class CitationExtractor:
    """
    Extracts and validates citations from LLM-generated answers.

    TWO EXTRACTION APPROACHES:
      1. REGEX-BASED: Fast, extracts inline [Source: X, Page Y] markers from the answer text
      2. LLM-BASED: Use a second LLM call to extract citations as structured JSON (more reliable)

    We use regex as primary (zero cost) with LLM fallback for complex cases.

    USAGE:
        extractor = CitationExtractor()
        citations = extractor.extract(
            answer_text="...the result [Source: report.pdf, Page 5]...",
            retrieved_chunks=reranked_chunks,
        )
    """

    # Regex pattern for "[Source: filename.pdf, Page 5]" format
    # Also handles variations like "Page: 5", "p.5", "p5"
    CITATION_PATTERN = re.compile(
        r"\[Source:\s*([^,\]]+?),?\s*(?:Page|p\.?):?\s*(\d+)\]",
        re.IGNORECASE,
    )

    def extract(
        self,
        answer_text: str,
        retrieved_chunks: list[dict],
    ) -> list[Citation]:
        """
        Extracts and validates citations from the LLM's answer.

        PROCESS:
          1. Find all [Source: X, Page Y] markers in the answer text
          2. For each citation, find the matching chunk in retrieved_chunks
          3. Extract a brief excerpt from the matching chunk
          4. Flag invalid citations (hallucinated page numbers)
          5. Deduplicate by (source, page) pair

        Args:
            answer_text: The full text of the LLM's generated answer.
            retrieved_chunks: The chunks sent to the LLM (from reranker).

        Returns:
            List of Citation objects (validated, deduplicated).
        """
        if not answer_text:
            return []

        # Extract raw citations from text using regex
        raw_citations = self._extract_regex(answer_text)

        if not raw_citations:
            logger.debug("No citations found in answer text via regex.")
            return []

        # Validate and enrich citations against retrieved chunks
        validated = []
        seen = set()  # For deduplication

        for source, page_num in raw_citations:
            dedup_key = (source.lower(), page_num)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Find matching chunk in retrieved set
            matching_chunk = self._find_matching_chunk(source, page_num, retrieved_chunks)

            if matching_chunk:
                excerpt = self._create_excerpt(matching_chunk["text"])
                citation = Citation(
                    source=source,
                    page_number=page_num,
                    excerpt=excerpt,
                    chunk_id=matching_chunk.get("chunk_id"),
                    is_valid=True,
                )
            else:
                # Citation doesn't match any retrieved chunk — HALLUCINATION DETECTED
                logger.warning(
                    f"Hallucinated citation detected: [{source}, Page {page_num}] "
                    "was not in the retrieved chunks."
                )
                citation = Citation(
                    source=source,
                    page_number=page_num,
                    excerpt="[Citation could not be verified in retrieved context]",
                    chunk_id=None,
                    is_valid=False,
                )

            validated.append(citation)

        logger.info(
            f"Citation extraction: {len(validated)} unique citations | "
            f"{sum(1 for c in validated if c.is_valid)} valid | "
            f"{sum(1 for c in validated if not c.is_valid)} hallucinated"
        )

        return validated

    def extract_from_chunks(self, chunks: list[dict]) -> list[Citation]:
        """
        Generates citations directly from retrieved chunks (bypass LLM output).
        Useful when you want to show "sources consulted" regardless of LLM output.

        Args:
            chunks: Reranked chunks sent to the LLM.

        Returns:
            List of Citation objects for all chunks.
        """
        seen = set()
        citations = []

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Unknown")
            page = meta.get("page_number", 0)

            key = (source.lower(), page)
            if key in seen:
                continue
            seen.add(key)

            citations.append(Citation(
                source=source,
                page_number=page,
                excerpt=self._create_excerpt(chunk.get("text", "")),
                chunk_id=chunk.get("chunk_id"),
                is_valid=True,
            ))

        return citations

    def _extract_regex(self, text: str) -> list[tuple[str, int]]:
        """
        Extracts (source, page_number) pairs from citation markers.

        Returns:
            List of (source_str, page_int) tuples.
        """
        matches = self.CITATION_PATTERN.findall(text)
        results = []
        for source, page_str in matches:
            try:
                results.append((source.strip(), int(page_str)))
            except ValueError:
                continue
        return results

    @staticmethod
    def _find_matching_chunk(
        source: str,
        page_number: int,
        chunks: list[dict],
    ) -> Optional[dict]:
        """
        Finds the chunk in retrieved_chunks that matches the citation.

        MATCHING LOGIC:
          - Source filename match (case-insensitive partial match)
          - Page number exact match

        PARTIAL MATCH WHY?
          The LLM might cite "seminar_report" but the actual source is
          "seminar_report_2024.pdf". Partial matching handles this.
        """
        source_lower = source.lower().replace(".pdf", "")

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            chunk_source = meta.get("source", "").lower().replace(".pdf", "")
            chunk_page = meta.get("page_number", -1)

            # Check if source name matches (partial, case-insensitive)
            source_match = (
                source_lower in chunk_source or
                chunk_source in source_lower or
                source_lower == chunk_source
            )

            if source_match and chunk_page == page_number:
                return chunk

        return None

    @staticmethod
    def _create_excerpt(text: str, max_length: int = 200) -> str:
        """Creates a short excerpt from chunk text for display."""
        text = text.strip()
        if len(text) <= max_length:
            return text
        # Truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated + "..."
