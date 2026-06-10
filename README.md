# 📚 Seminar RAG Assistant — Production-Grade RAG System

> A complete, production-oriented Retrieval-Augmented Generation (RAG) system for querying and interacting with seminar reports. Built for junior AI engineers to learn from, extend, and deploy.

---

## 🏗️ Architecture Overview

```
User Question
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  FastAPI    │────▶│  Retriever   │────▶│ Vector Store │
│  /query     │     │  (Embed Q)   │     │ (ChromaDB)   │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │ Top-K chunks
                                                 ▼
                                         ┌──────────────┐
                                         │   Reranker   │
                                         │(Cross-Encoder)│
                                         └──────┬───────┘
                                                 │ Top-N chunks
                                                 ▼
                                         ┌──────────────┐
                                         │PromptBuilder │
                                         │(Inject Context)│
                                         └──────┬───────┘
                                                 │
                                                 ▼
                                         ┌──────────────┐
                                         │  LLM Chain   │
                                         │ (GPT-4o-mini)│
                                         └──────┬───────┘
                                                 │
                                                 ▼
                                   ┌──────────────────────────┐
                                   │   Answer + Citations     │
                                   └──────────────────────────┘
```

### The 12-Stage Pipeline

| # | Stage | Module | Description |
|---|---|---|---|
| 1 | **Document Ingestion** | `api/routes/ingest.py` | PDF upload & validation |
| 2 | **PDF Parsing** | `services/ingestion/pdf_parser.py` | PyMuPDF text extraction |
| 3 | **Text Cleaning** | `services/ingestion/text_cleaner.py` | Normalize, fix hyphenation |
| 4 | **Chunking** | `services/ingestion/chunker.py` | Recursive/semantic splitting |
| 5 | **Embedding** | `services/ingestion/embedder.py` | OpenAI text-embedding-3-small |
| 6 | **Vector Storage** | `services/retrieval/vector_store.py` | ChromaDB persistent store |
| 7 | **Retrieval** | `services/retrieval/retriever.py` | Cosine similarity top-K |
| 8 | **Reranking** | `services/retrieval/reranker.py` | Cross-encoder ms-marco |
| 9 | **Prompt Construction** | `services/generation/prompt_builder.py` | Grounded prompt assembly |
| 10 | **LLM Generation** | `services/generation/llm_chain.py` | GPT-4o-mini with streaming |
| 11 | **Citation Extraction** | `services/generation/citation_extractor.py` | Parse + validate citations |
| 12 | **Response Formatting** | `services/generation/response_formatter.py` | Structured JSON output |

---

## 📁 Project Structure

```
RAG/
├── app/
│   └── main.py                    # FastAPI application factory, startup, middleware
│
├── api/                           # HTTP layer (routes + schemas only)
│   ├── routes/
│   │   ├── ingest.py              # Upload, process, list, delete documents
│   │   └── query.py               # RAG query, streaming, chat, health
│   └── schemas/
│       ├── ingest.py              # Pydantic request/response models for ingestion
│       └── query.py               # Pydantic models for queries and chat
│
├── services/                      # Core business logic (pipeline stages)
│   ├── ingestion/
│   │   ├── pdf_parser.py          # Stage 1-2: PDF text extraction with metadata
│   │   ├── text_cleaner.py        # Stage 3: Text normalization pipeline
│   │   ├── chunker.py             # Stage 4: Semantic + recursive chunking
│   │   └── embedder.py            # Stage 5: Batch embedding with retry
│   ├── retrieval/
│   │   ├── vector_store.py        # Stage 6: ChromaDB persistent vector storage
│   │   ├── retriever.py           # Stage 7: Top-K semantic retrieval
│   │   └── reranker.py            # Stage 8: Cross-encoder reranking
│   └── generation/
│       ├── prompt_builder.py      # Stage 9: Grounded prompt construction
│       ├── llm_chain.py           # Stage 10: Async LLM generation + streaming
│       ├── citation_extractor.py  # Stage 11: Citation parsing + validation
│       └── response_formatter.py  # Stage 12: Structured response packaging
│
├── prompts/                       # Editable prompt templates (no code changes needed)
│   ├── system_prompt.txt          # System instruction (grounding, anti-hallucination)
│   ├── answer_prompt.txt          # Answer template with context injection
│   └── citation_prompt.txt        # Citation extraction template
│
├── data/uploads/                  # Uploaded PDFs (gitignored)
├── vector_store/                  # ChromaDB data (gitignored)
│
├── frontend/
│   └── app.py                     # Streamlit UI (upload, chat, citations)
│
├── utils/
│   ├── config.py                  # Pydantic-settings configuration
│   ├── logger.py                  # Structured logging
│   ├── token_counter.py           # tiktoken-based token counting + cost estimation
│   └── validators.py              # File validation (magic bytes, size, hash)
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py        # Chunking edge cases and metadata
│   │   ├── test_cleaner.py        # Text cleaning transformations
│   │   └── test_citation.py       # Citation extraction and validation
│   └── integration/
│       └── test_ingest_api.py     # API endpoint tests with mocked services
│
├── docker/
│   ├── Dockerfile.backend         # Multi-stage FastAPI container
│   ├── Dockerfile.frontend        # Streamlit container
│   └── docker-compose.yml         # Full stack orchestration
│
├── .env.example                   # Environment variable template
├── .gitignore
├── requirements.txt
├── pyproject.toml                 # pytest configuration
└── README.md
```

---

## ⚡ Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- An OpenAI API key

### 1. Clone & Setup

```bash
git clone <your-repo>
cd RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Start the Backend

```bash
uvicorn app.main:app --reload --port 8000
```

Visit: http://localhost:8000/docs — Interactive API documentation

### 4. Start the Frontend

```bash
streamlit run frontend/app.py
```

Visit: http://localhost:8501

### 5. Use the System

1. **Upload** a seminar PDF via the sidebar
2. Click **"Upload & Process"** — this runs the full ingestion pipeline
3. **Ask questions** in the chat input
4. View **citations** with page references below each answer

---

## 🐳 Docker Deployment

```bash
cd docker

# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop everything
docker-compose down

# View logs
docker-compose logs -f backend
```

**Services:**
- Backend: http://localhost:8000
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs

---

## 🔌 API Reference

### Upload a PDF
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/upload" \
  -F "file=@your_report.pdf"
# Returns: {"doc_id": "sha256...", "filename": "...", "file_size_mb": 1.2}
```

### Ingest (Embed) the Document
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/process/{doc_id}"
# Returns: {"pages_parsed": 42, "chunks_created": 180, "chunks_stored": 180}
```

### Ask a Question
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the proposed methodology?"}'
```

### Conversational Chat
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can you elaborate on that?",
    "session_id": "my-session-123",
    "history": [
      {"role": "user", "content": "What is the main contribution?"},
      {"role": "assistant", "content": "The main contribution is..."}
    ]
  }'
```

### Stream an Answer
```bash
curl -N "http://localhost:8000/api/v1/query/stream" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the key findings"}'
```

---

## 🧪 Running Tests

```bash
# All tests
pytest

# Unit tests only (fast, no API calls)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage report
pytest --cov=services --cov=utils --cov-report=html
```

---

## ⚙️ Configuration Reference

All settings are in `.env`. See `.env.example` for the complete list.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *required* | Your OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | LLM for answer generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `800` | Tokens per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `20` | Candidates from vector search |
| `RERANKER_TOP_N` | `5` | Chunks after reranking |
| `CHUNKING_STRATEGY` | `recursive` | `recursive` or `semantic` |

---

## 🔧 Extending the System

### Swap the Vector Database (Pinecone)
```python
# services/retrieval/vector_store.py
# Replace ChromaDB client with Pinecone:
import pinecone
pinecone.init(api_key=settings.PINECONE_API_KEY, environment="us-east1-gcp")
index = pinecone.Index("seminar-reports")
```

### Swap the Embedding Model (Local BGE)
```python
# services/ingestion/embedder.py
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
self._embeddings_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
```

### Swap the LLM (Claude)
```python
# services/generation/llm_chain.py
from langchain_anthropic import ChatAnthropic
self._llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
```

### Add Background Processing (Celery)
```python
# For large document ingestion, use Celery + Redis:
from celery import Celery
app = Celery("rag", broker="redis://localhost:6379/0")

@app.task
def ingest_document_task(doc_id: str):
    # Run ingestion pipeline in background
    ...
```

---

## 🏭 Production Checklist

- [ ] Set `DEBUG=false` in production
- [ ] Use Gunicorn: `gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker`
- [ ] Add nginx reverse proxy with SSL/TLS
- [ ] Use Redis-backed sessions for multi-instance chat
- [ ] Set up log aggregation (ELK, CloudWatch)
- [ ] Monitor with Prometheus + Grafana
- [ ] Implement per-user API key authentication
- [ ] Back up `vector_store/` directory regularly
- [ ] Consider Pinecone/Qdrant for production vector storage
- [ ] Set up LangSmith for LLM call tracing

---

## 🧠 Key Design Decisions

### Why Two-Stage Retrieval (Retrieve → Rerank)?
Dense retrieval (embeddings) is fast but imprecise. The cross-encoder reranker reads both the query and each chunk together for much more accurate relevance scoring. This is the industry standard approach.

### Why Chunk at the Page Level?
Chunking per-page preserves page number metadata accurately. If we chunked the whole document at once, we'd lose track of which page each piece of text came from.

### Why Low Temperature (0.1)?
We want the LLM to extract facts from context, not invent them. Low temperature makes the model more deterministic and less likely to hallucinate.

### Why Citation Validation?
LLMs can hallucinate citation page numbers. By cross-referencing every citation against the retrieved chunks, we detect and flag unsupported claims.

---

## 📊 Example Response

```json
{
  "response_id": "a3f2bc...",
  "question": "What is the proposed algorithm?",
  "answer": "The paper proposes a transformer-based architecture for real-time processing [Source: report.pdf, Page 5]. The key innovation is the attention mechanism that reduces computational complexity from O(n²) to O(n log n) [Source: report.pdf, Page 7].",
  "citations": [
    {
      "source": "report.pdf",
      "page_number": 5,
      "excerpt": "We propose a transformer-based architecture...",
      "is_valid": true
    }
  ],
  "metadata": {
    "total_chunks_retrieved": 20,
    "chunks_after_reranking": 5,
    "has_sufficient_context": true,
    "hallucinated_citations": 0
  },
  "performance": {
    "retrieval_latency_ms": 45.2,
    "generation_latency_ms": 1823.4,
    "total_latency_ms": 1868.6,
    "input_tokens": 3241,
    "output_tokens": 187,
    "estimated_cost_usd": 0.000598
  }
}
```

---


# FIrst-RAG-pro

