"""
validators.py — File & Input Validation
=========================================
WHY THIS EXISTS:
  Never trust user input. Validate early and fail fast.
  Bad uploads can crash the pipeline, consume resources, or introduce security risks.

COMMON FAILURE POINTS:
  - Users uploading non-PDF files with .pdf extension
  - Corrupted PDFs that cause parser crashes
  - Extremely large files that exhaust memory
  - Empty PDFs with no extractable text

PRODUCTION CONSIDERATIONS:
  - Scan uploaded files for malware in high-security environments.
  - Validate file signatures (magic bytes), not just extensions.
  - Implement per-user upload quotas.
"""

import hashlib
from pathlib import Path
from fastapi import UploadFile, HTTPException
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# PDF magic bytes — first 4 bytes of a valid PDF file
PDF_MAGIC_BYTES = b"%PDF"


def validate_pdf_upload(file: UploadFile) -> None:
    """
    Validates an uploaded file for type, size, and content.
    Raises HTTPException with descriptive messages on failure.

    WHY validate magic bytes?
      A user can rename any file to .pdf. Reading the first 4 bytes
      confirms it's actually a PDF, preventing parser crashes downstream.

    Args:
        file: The uploaded file from FastAPI.

    Raises:
        HTTPException 400: If validation fails.
    """
    # 1. Check file extension
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{suffix}'. Only {settings.ALLOWED_EXTENSIONS} are allowed.",
        )
    logger.debug(f"Extension check passed: {file.filename}")


async def validate_pdf_content(file_path: Path) -> None:
    """
    Validates the actual file content after saving to disk.
    Checks magic bytes and file size.

    Args:
        file_path: Path to the saved file on disk.

    Raises:
        HTTPException 400: If the file is invalid.
        HTTPException 413: If the file is too large.
    """
    # 2. Check file size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        file_path.unlink(missing_ok=True)  # Clean up oversized file
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Max allowed: {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    # 3. Check PDF magic bytes
    with open(file_path, "rb") as f:
        header = f.read(4)

    if header != PDF_MAGIC_BYTES:
        file_path.unlink(missing_ok=True)  # Clean up invalid file
        raise HTTPException(
            status_code=400,
            detail="File is not a valid PDF (incorrect file signature).",
        )

    logger.info(f"File validation passed: {file_path.name} ({size_mb:.2f}MB)")


def compute_file_hash(file_path: Path) -> str:
    """
    Computes SHA-256 hash of a file for deduplication.

    WHY?
      If a user uploads the same PDF twice, we detect it here and skip
      re-embedding, saving significant time and API cost.

    Args:
        file_path: Path to the file.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
