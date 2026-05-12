"""
api/routes/ingest.py — Document Ingestion Endpoints
=====================================================
ENDPOINTS:
  POST /api/v1/ingest/upload   — Upload a PDF file
  POST /api/v1/ingest/process  — Ingest (parse, chunk, embed, store) a document
  GET  /api/v1/ingest/documents — List all ingested documents
  DELETE /api/v1/ingest/{doc_id} — Remove a document

DESIGN — TWO-STEP INGESTION:
  Upload and ingest are separate steps. WHY?
  1. Upload is fast (just save the file to disk)
  2. Ingestion is SLOW (parsing, embedding API calls can take minutes for large docs)

  With a two-step approach:
  - Upload returns immediately with a doc_id
  - Frontend can show a "Processing..." state while ingestion runs
  - Ingestion can run as a background task (Celery in production)
  - User can upload multiple files then process them in batch

PRODUCTION ENHANCEMENT:
  Replace the sync ingestion with a Celery background task:
    @router.post("/process/{doc_id}")
    async def process_document(doc_id: str, background_tasks: BackgroundTasks):
        background_tasks.add_task(ingest_pipeline, doc_id)  # Non-blocking!
        return {"status": "processing", "message": "Ingestion started in background"}
"""

import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.schemas.ingest import (
    DocumentListResponse,
    DocumentInfo,
    IngestResponse,
    UploadResponse,
)
from services.ingestion.chunker import DocumentChunker
from services.ingestion.embedder import EmbeddingGenerator
from services.ingestion.pdf_parser import PDFParser
from services.ingestion.text_cleaner import TextCleaner
from services.retrieval.vector_store import VectorStore
from utils.config import get_settings
from utils.logger import get_logger
from utils.validators import compute_file_hash, validate_pdf_content, validate_pdf_upload

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = get_logger(__name__)
settings = get_settings()

# ─── Dependency Injection ─────────────────────────────────────────────────────
# FastAPI's dependency injection creates shared service instances.
# WHY? Services like VectorStore and EmbeddingGenerator are expensive to initialize.
# We create them once at app startup, not per request.

def get_vector_store() -> VectorStore:
    """Returns the singleton VectorStore instance."""
    from app.main import get_app_state
    return get_app_state().vector_store


def get_embedder() -> EmbeddingGenerator:
    """Returns the singleton EmbeddingGenerator instance."""
    from app.main import get_app_state
    return get_app_state().embedder


# ─── Upload Endpoint ──────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a PDF seminar report",
    description="Accepts a PDF file, validates it, saves it to disk, and returns a doc_id for ingestion.",
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Step 1: Upload a PDF file.

    PROCESS:
      1. Validate file type (extension + magic bytes)
      2. Save to uploads directory
      3. Compute SHA-256 hash (used as doc_id)
      4. Return doc_id for the next step (ingestion)

    ERROR CASES:
      - 400: Not a PDF, invalid content, or file already exists
      - 413: File exceeds size limit
    """
    # Fast check: validate extension before reading the file
    validate_pdf_upload(file)

    # Save file to disk
    upload_dir = settings.upload_path
    safe_filename = Path(file.filename).name  # Strip any path traversal
    save_path = upload_dir / safe_filename

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        await file.close()

    # Deep validation: check magic bytes and size
    await validate_pdf_content(save_path)

    # Compute doc_id from file hash (enables deduplication)
    doc_id = compute_file_hash(save_path)
    file_size_mb = save_path.stat().st_size / (1024 * 1024)

    logger.info(f"File uploaded: {safe_filename} ({file_size_mb:.2f}MB) → doc_id={doc_id[:8]}...")

    return UploadResponse(
        doc_id=doc_id,
        filename=safe_filename,
        file_size_mb=round(file_size_mb, 2),
        message=f"Upload successful. Use doc_id '{doc_id}' to ingest this document.",
    )


# ─── Ingest Endpoint ─────────────────────────────────────────────────────────

@router.post(
    "/process/{doc_id}",
    response_model=IngestResponse,
    summary="Process and embed an uploaded document",
    description="Runs the full ingestion pipeline: parse → clean → chunk → embed → store.",
)
async def ingest_document(
    doc_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
    embedder: EmbeddingGenerator = Depends(get_embedder),
) -> IngestResponse:
    """
    Step 2: Run the ingestion pipeline on an uploaded PDF.

    This is the CORE INGESTION FLOW:
      PDF → Parse → Clean → Chunk → Embed → Store

    WHY SYNC?
      For simplicity in this implementation. In production, use a background
      task queue (Celery + Redis) to avoid blocking the API thread.

    ERROR CASES:
      - 404: doc_id not found in uploads directory
      - 500: Parsing, embedding, or storage failure
    """
    # Find the file for this doc_id
    file_path = _find_file_by_doc_id(doc_id)
    if not file_path:
        raise HTTPException(
            status_code=404,
            detail=f"No uploaded file found for doc_id='{doc_id}'. Please upload first.",
        )

    start_time = time.time()
    logger.info(f"Starting ingestion pipeline for: {file_path.name} (doc_id={doc_id[:8]}...)")

    try:
        # Stage 1-2: Parse PDF
        parser = PDFParser()
        pages = parser.parse(file_path, doc_id=doc_id)
        pages_parsed = len(pages)

        # Stage 3: Clean text
        cleaner = TextCleaner()
        for page in pages:
            page.text = cleaner.clean(page.text)

        # Stage 4: Chunk
        chunker = DocumentChunker()
        chunks = chunker.chunk_pages(pages)
        chunks_created = len(chunks)

        # Stage 5: Embed
        chunks_with_embeddings = embedder.embed_chunks(chunks)
        chunks_embedded = len(chunks_with_embeddings)

        # Stage 6: Store in vector DB
        chunks_stored = vector_store.add_chunks(chunks_with_embeddings)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed for doc_id={doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline error: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"Ingestion complete: {pages_parsed} pages, {chunks_created} chunks, "
        f"{chunks_stored} stored in {elapsed:.1f}s"
    )

    return IngestResponse(
        doc_id=doc_id,
        source=file_path.name,
        pages_parsed=pages_parsed,
        chunks_created=chunks_created,
        chunks_embedded=chunks_embedded,
        chunks_stored=chunks_stored,
        status="success",
        message=f"Successfully ingested '{file_path.name}' in {elapsed:.1f}s.",
    )


# ─── List Documents ───────────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_documents(
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentListResponse:
    """Returns all documents currently stored in the vector database."""
    docs_raw = vector_store.list_documents()
    stats = vector_store.get_collection_stats()

    documents = [
        DocumentInfo(
            doc_id=d["doc_id"],
            source=d["source"],
            total_pages=d.get("total_pages", 0),
            chunk_count=d["chunk_count"],
            status="ingested",
        )
        for d in docs_raw
    ]

    return DocumentListResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=stats.get("total_chunks", 0),
    )


# ─── Delete Document ─────────────────────────────────────────────────────────

@router.delete(
    "/{doc_id}",
    summary="Remove a document from the vector store",
)
async def delete_document(
    doc_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    """Deletes all chunks for a specific document from the vector store."""
    deleted = vector_store.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document found with doc_id='{doc_id}'")
    return {"doc_id": doc_id, "chunks_deleted": deleted, "status": "deleted"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _find_file_by_doc_id(doc_id: str) -> Path | None:
    """
    Finds the uploaded file corresponding to a doc_id.

    WHY? We don't store a doc_id→filename mapping in a database.
    Instead, we recompute the hash of each file in the uploads directory
    and find the one that matches.

    PRODUCTION ENHANCEMENT:
      Maintain a simple SQLite or Redis mapping: {doc_id: filename}
      to avoid scanning the uploads directory on every ingest request.
    """
    from utils.validators import compute_file_hash

    upload_dir = settings.upload_path
    for file_path in upload_dir.glob("*.pdf"):
        try:
            if compute_file_hash(file_path) == doc_id:
                return file_path
        except Exception:
            continue
    return None
