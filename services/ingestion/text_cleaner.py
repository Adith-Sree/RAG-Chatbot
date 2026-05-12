"""
text_cleaner.py — Stage 3: Text Cleaning & Normalization
==========================================================
WHY THIS EXISTS:
  Raw PDF text is messy. It contains:
  - Running headers/footers repeated on every page ("Department of CS | Page 5")
  - Broken words from hyphenation ("computa-\ntional" → "computational")
  - Excessive whitespace and newlines
  - Special characters that confuse tokenizers

  Clean text dramatically improves embedding quality because:
  - Embeddings capture semantic meaning; noise dilutes this signal
  - Cleaner text chunks lead to more precise similarity matches
  - Consistent formatting helps the LLM understand context better

WHAT WE DON'T DO:
  - We don't remove ALL punctuation (punctuation conveys meaning)
  - We don't stem/lemmatize (embeddings handle semantic similarity)
  - We don't lowercase (capitalization can be meaningful)

COMMON FAILURE POINTS:
  - Overly aggressive cleaning removes meaningful content
  - Regex patterns that work on English fail on other languages
  - Header/footer detection is heuristic — may miss unusual formats

PRODUCTION CONSIDERATIONS:
  - For multi-language documents, use language-aware cleaning
  - Log before/after statistics to verify cleaning quality
  - Allow per-document cleaning profiles for different report formats
"""

import re
from utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """
    Cleans and normalizes raw PDF-extracted text.

    DESIGN PRINCIPLE:
      Each cleaning operation is a separate method. This makes it easy to:
      - Test each step independently
      - Disable/enable specific steps per use case
      - Understand exactly what transformation was applied

    USAGE:
        cleaner = TextCleaner()
        clean_text = cleaner.clean("raw pdf text here...")
    """

    def clean(self, raw_text: str) -> str:
        """
        Applies the full cleaning pipeline to raw PDF text.

        Pipeline order matters:
          1. Fix hyphenation first (before whitespace normalization)
          2. Remove headers/footers (before line joining)
          3. Normalize whitespace last (catches all the above)

        Args:
            raw_text: Raw text extracted from a PDF page.

        Returns:
            Cleaned text string.
        """
        if not raw_text or not raw_text.strip():
            return ""

        text = raw_text

        # Step 1: Fix hyphenated line breaks
        # "computa-\ntional" → "computational"
        # WHY: PDF line-break hyphenation is a formatting artifact, not real content.
        text = self._fix_hyphenation(text)

        # Step 2: Remove common header/footer patterns
        # WHY: Page headers repeat on every page and pollute chunk embeddings.
        text = self._remove_headers_footers(text)

        # Step 3: Normalize Unicode characters
        # WHY: PDFs often use special dash/quote characters that tokenizers mishandle.
        text = self._normalize_unicode(text)

        # Step 4: Remove excessive whitespace and blank lines
        # WHY: Multiple consecutive blank lines add no meaning.
        text = self._normalize_whitespace(text)

        return text.strip()

    def _fix_hyphenation(self, text: str) -> str:
        """
        Fixes word-break hyphenation introduced by PDF line wrapping.

        Examples:
          "computa-\ntional" → "computational"
          "semi-\nnar"       → "seminar"

        Note: We're careful to only fix mid-word hyphens, not intentional dashes.
        """
        # Pattern: word characters, hyphen, optional spaces, newline, word characters
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        return text

    def _remove_headers_footers(self, text: str) -> str:
        """
        Removes common academic report header/footer patterns.

        HEURISTIC APPROACH:
          We can't perfectly identify headers/footers without layout data.
          Instead, we remove lines that match common patterns:
          - "Page X of Y" or "Page X"
          - Standalone page numbers
          - Lines that appear to be running headers (short, all caps)

        LIMITATION:
          This is heuristic and may occasionally remove valid content.
          For production, use PyMuPDF block-level parsing to identify
          header/footer regions by their Y-coordinates.
        """
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip standalone page numbers: "5", "- 5 -", "Page 5", "Page 5 of 42"
            if re.fullmatch(r"[-–—\s]*Page\s+\d+(\s+of\s+\d+)?[-–—\s]*", stripped, re.IGNORECASE):
                continue
            if re.fullmatch(r"[-–—\s]*\d+[-–—\s]*", stripped):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _normalize_unicode(self, text: str) -> str:
        """
        Replaces special Unicode characters with ASCII equivalents.

        WHY:
          PDFs use special typographic characters (curly quotes, em dashes, etc.)
          that are semantically identical to ASCII but cause tokenizer issues.

        Examples:
          \u2019 (right single quote) → '
          \u2013 (en dash)            → -
          \u2014 (em dash)            → -
          \u00a0 (non-breaking space) → space
        """
        replacements = {
            "\u2018": "'",  # left single quotation mark
            "\u2019": "'",  # right single quotation mark
            "\u201c": '"',  # left double quotation mark
            "\u201d": '"',  # right double quotation mark
            "\u2013": "-",  # en dash
            "\u2014": "-",  # em dash
            "\u2022": "-",  # bullet
            "\u00a0": " ",  # non-breaking space
            "\u00ad": "",   # soft hyphen
            "\uf0b7": "-",  # private use bullet (common in PDFs)
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Collapses excessive whitespace while preserving paragraph structure.

        We keep single newlines (within paragraphs) but collapse:
        - 3+ consecutive newlines → 2 newlines (paragraph break)
        - Multiple spaces → single space
        """
        # Collapse runs of spaces/tabs (but not newlines)
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse 3+ consecutive newlines to 2 (paragraph separator)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text
