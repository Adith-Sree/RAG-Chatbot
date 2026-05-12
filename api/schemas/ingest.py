"""
schemas/ingest.py — Pydantic Models for Ingestion API
=======================================================
WHY PYDANTIC SCHEMAS?
  FastAPI uses Pydantic models to:
  1. AUTO-VALIDATE incoming request data (types, constraints)
  2. AUTO-SERIALIZE outgoing response data to JSON
  3. AUTO-GENERATE OpenAPI/Swagger documentation

  Without schemas, you'd manually parse request JSON and format responses.
  With schemas, all of this is declarative and self-documenting.

DESIGN PRINCIPLE:
  Separate request schemas from response schemas.
  Request: what the client sends us.
  Response: what we send back.
  Never mix them — they evolve independently.
"""

from typing import Optional
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Information about an ingested document."""
    doc_id: str = Field(..., description="SHA-256 hash of the document")
    source: str = Field(..., description="Original filename")
    total_pages: int = Field(..., description="Total number of pages")
    chunk_count: int = Field(..., description="Number of chunks stored in vector DB")
    status: str = Field(..., description="Ingestion status: 'success' or 'error'")


class UploadResponse(BaseModel):
    """Response after uploading a PDF."""
    doc_id: str = Field(..., description="Document identifier (SHA-256 hash)")
    filename: str = Field(..., description="Saved filename")
    file_size_mb: float = Field(..., description="File size in MB")
    message: str = Field(..., description="Status message")


class IngestRequest(BaseModel):
    """Request to ingest an already-uploaded document."""
    doc_id: str = Field(..., description="Document ID from the upload step")
    chunking_strategy: Optional[str] = Field(
        default=None,
        description="Override: 'semantic' or 'recursive'. Uses config default if not set."
    )


class IngestResponse(BaseModel):
    """Response after ingesting and embedding a document."""
    doc_id: str
    source: str
    pages_parsed: int
    chunks_created: int
    chunks_embedded: int
    chunks_stored: int
    status: str
    message: str


class DocumentListResponse(BaseModel):
    """Response listing all ingested documents."""
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int
