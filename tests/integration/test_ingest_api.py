"""
tests/integration/test_ingest_api.py — Integration Tests for Ingestion API
===========================================================================
WHY INTEGRATION TESTS?
  Unit tests verify individual functions in isolation.
  Integration tests verify the whole system works END-TO-END.

  Here we test the actual FastAPI endpoints, including:
  - File upload validation
  - Correct error responses for bad inputs
  - Successful ingestion flow

HOW TO RUN:
  pytest tests/integration/ -v

REQUIREMENTS:
  - Backend must be running OR we use TestClient (which runs FastAPI in-process)
  - No actual OpenAI API calls (we mock them)

DESIGN — MOCK OPENAI:
  We mock the OpenAI API calls to avoid:
  - Actual API costs during testing
  - Flaky tests due to network issues
  - Rate limiting during CI/CD
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def create_minimal_pdf() -> bytes:
    """
    Creates a minimal valid PDF in memory for testing.
    This is a real PDF with the correct magic bytes and structure.
    """
    # Minimal valid PDF content
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << >> /MediaBox [0 0 612 792]
/Contents 4 0 R >>
endobj

4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Test Content) Tj ET
endstream
endobj

xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n

trailer
<< /Size 5 /Root 1 0 R >>
startxref
360
%%EOF"""
    return pdf_content


@pytest.fixture
def mock_app_state():
    """Mocks the application state to avoid real service initialization."""
    mock_state = MagicMock()
    mock_state.vector_store.add_chunks.return_value = 10
    mock_state.vector_store.list_documents.return_value = []
    mock_state.vector_store.get_collection_stats.return_value = {"total_chunks": 0}
    mock_state.embedder.embed_chunks.return_value = [(MagicMock(), [0.1] * 1536)]
    return mock_state


@pytest.fixture
def client(mock_app_state, tmp_path, monkeypatch):
    """Creates a test client with mocked services."""
    # Override settings to use temp directories
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-testing")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "vector_store"))

    with patch("app.main.get_app_state", return_value=mock_app_state):
        with patch("app.main.VectorStore", return_value=mock_app_state.vector_store):
            with patch("app.main.EmbeddingGenerator", return_value=mock_app_state.embedder):
                with patch("app.main.Reranker", return_value=mock_app_state.reranker):
                    with patch("app.main.LLMChain", return_value=mock_app_state.llm_chain):
                        with patch("app.main.Retriever", return_value=mock_app_state.retriever):
                            app = create_app()
                            with TestClient(app) as tc:
                                yield tc


class TestUploadEndpoint:

    def test_root_returns_200(self, client):
        """Root endpoint should be reachable."""
        response = client.get("/")
        assert response.status_code == 200

    def test_upload_non_pdf_rejected(self, client):
        """Uploading a .txt file should be rejected with 400."""
        response = client.post(
            "/api/v1/ingest/upload",
            files={"file": ("test.txt", b"Not a PDF", "text/plain")},
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_upload_fake_pdf_rejected(self, client, tmp_path):
        """Uploading a file with .pdf extension but wrong content should fail."""
        # File has .pdf extension but is just text (wrong magic bytes)
        response = client.post(
            "/api/v1/ingest/upload",
            files={"file": ("fake.pdf", b"This is not a PDF at all", "application/pdf")},
        )
        # Should fail at magic bytes check
        assert response.status_code in (400, 422)

    def test_upload_valid_pdf_succeeds(self, client, tmp_path):
        """Uploading a valid PDF should succeed and return doc_id."""
        pdf_content = create_minimal_pdf()
        response = client.post(
            "/api/v1/ingest/upload",
            files={"file": ("test_report.pdf", pdf_content, "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "doc_id" in data
        assert "filename" in data
        assert len(data["doc_id"]) == 64  # SHA-256 hex = 64 chars

    def test_list_documents_returns_list(self, client):
        """List documents endpoint should return a list."""
        response = client.get("/api/v1/ingest/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)


class TestHealthEndpoint:

    def test_health_check_returns_status(self, client):
        """Health endpoint should return status and checks."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data
