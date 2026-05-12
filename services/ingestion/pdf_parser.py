"""
pdf_parser.py — Stage 1 & 2: Document Ingestion & PDF Parsing
===============================================================
WHY THIS EXISTS:
  PDFs are the most common format for academic/seminar reports, but they are
  notoriously hard to parse reliably. This module handles:
    1. Loading the PDF file from disk
    2. Extracting clean text from each page
    3. Preserving critical metadata: page numbers, source file, total pages

WHY PyMuPDF (fitz)?
  - Fastest PDF parser in Python (written in C)
  - Handles complex layouts, multi-column text, tables
  - Extracts text with position info (useful for future layout-aware chunking)
  - Better Unicode support than older libraries

WHY pdfplumber as fallback?
  - Better at extracting text from PDFs with complex formatting or tables
  - Some PDFs render better with one tool vs the other

COMMON FAILURE POINTS:
  - Scanned PDFs (images of text) → need OCR (Tesseract); we detect and warn.
  - Password-protected PDFs → we detect and raise clear errors.
  - Corrupted PDFs → handled with try/except and clear error messages.
  - PDFs with mostly images → low text yield; we warn the user.

METADATA WE PRESERVE:
  {
    "source": "seminar_report_2024.pdf",
    "page_number": 3,
    "total_pages": 42,
    "doc_id": "sha256_hash",
    "file_path": "/data/uploads/seminar_report_2024.pdf"
  }

PRODUCTION CONSIDERATIONS:
  - For scanned PDFs, integrate Tesseract OCR (pytesseract).
  - For very large PDFs (500+ pages), process in async batches.
  - Cache parsed text to avoid re-parsing unchanged files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum characters per page to consider it "text-rich"
# Pages below this threshold might be scanned images or blank.
MIN_TEXT_THRESHOLD = 50


@dataclass
class ParsedPage:
    """
    Represents a single extracted page from a PDF.

    Keeping this as a dataclass (not a dict) gives us:
    - Type safety and auto-completion in IDEs
    - Clear documentation of what data a "page" contains
    - Easy serialization to dict for downstream processing
    """
    page_number: int           # 1-indexed page number
    text: str                  # Extracted raw text content
    source: str                # Original filename (e.g., "seminar_report.pdf")
    doc_id: str                # SHA-256 hash of the file for deduplication
    total_pages: int           # Total pages in the document
    file_path: str             # Absolute path to the file
    word_count: int = 0        # Computed word count
    is_likely_scanned: bool = False  # Flag if page has too little text

    def __post_init__(self):
        self.word_count = len(self.text.split())
        # If a page has very few characters, it's likely a scanned image
        self.is_likely_scanned = len(self.text.strip()) < MIN_TEXT_THRESHOLD

    def to_metadata(self) -> dict:
        """
        Returns a flat metadata dict suitable for ChromaDB storage.
        ChromaDB requires all metadata values to be str, int, or float.
        """
        return {
            "source": self.source,
            "page_number": self.page_number,
            "total_pages": self.total_pages,
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "word_count": self.word_count,
        }


class PDFParser:
    """
    Handles PDF loading and text extraction using PyMuPDF.

    DESIGN PRINCIPLE:
      Single Responsibility — this class ONLY parses PDFs.
      Cleaning, chunking, and embedding are handled by separate modules.

    USAGE:
        parser = PDFParser()
        pages = parser.parse("/data/uploads/report.pdf", doc_id="abc123")
        for page in pages:
            print(f"Page {page.page_number}: {page.word_count} words")
    """

    def parse(self, file_path: str | Path, doc_id: str) -> list[ParsedPage]:
        """
        Parses a PDF file and returns a list of ParsedPage objects.

        Args:
            file_path: Path to the PDF file.
            doc_id: SHA-256 hash of the file (from validators.compute_file_hash).

        Returns:
            List of ParsedPage objects, one per page.

        Raises:
            ValueError: If the PDF is password-protected or corrupted.
            FileNotFoundError: If the file doesn't exist.

        IMPLEMENTATION NOTES:
          - We use fitz.open() which raises fitz.FileDataError for corrupt files.
          - We use page.get_text("text") for clean plain-text extraction.
          - "text" mode extracts in reading order; use "blocks" for layout-aware extraction.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.info(f"Parsing PDF: {file_path.name}")

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            raise ValueError(f"Failed to open PDF '{file_path.name}': {e}") from e

        # Check for password protection
        if doc.is_encrypted:
            doc.close()
            raise ValueError(
                f"PDF '{file_path.name}' is password-protected. "
                "Please provide an unlocked version."
            )

        total_pages = doc.page_count
        logger.info(f"PDF has {total_pages} pages: {file_path.name}")

        pages: list[ParsedPage] = []
        scanned_count = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # Convert to 1-indexed

            try:
                fitz_page = doc[page_idx]

                # "text" mode: plain text in reading order
                # "blocks" mode: text with position data (useful for layout analysis)
                raw_text = fitz_page.get_text("text")

                parsed_page = ParsedPage(
                    page_number=page_num,
                    text=raw_text,
                    source=file_path.name,
                    doc_id=doc_id,
                    total_pages=total_pages,
                    file_path=str(file_path.absolute()),
                )

                if parsed_page.is_likely_scanned:
                    scanned_count += 1
                    logger.warning(
                        f"Page {page_num}/{total_pages} may be scanned (only "
                        f"{len(raw_text.strip())} chars extracted). "
                        "Consider OCR for this document."
                    )

                pages.append(parsed_page)

            except Exception as e:
                # Don't crash the whole parse job for one bad page
                logger.error(f"Failed to parse page {page_num}: {e}")
                continue

        doc.close()

        if scanned_count > total_pages * 0.5:
            logger.warning(
                f"Over 50% of pages appear scanned ({scanned_count}/{total_pages}). "
                f"Text quality may be poor. Consider OCR preprocessing."
            )

        total_words = sum(p.word_count for p in pages)
        logger.info(
            f"Parsing complete: {len(pages)} pages, {total_words} total words extracted."
        )

        return pages
