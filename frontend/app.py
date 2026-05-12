"""
frontend/app.py — Streamlit RAG Assistant UI
=============================================
WHY STREAMLIT?
  - Pure Python: no HTML/CSS/JS required
  - Built-in state management (st.session_state)
  - Easy file upload widgets
  - Real-time UI updates with st.empty() and st.spinner()
  - Perfect for ML/AI demos and internal tools

ARCHITECTURE:
  The frontend ONLY communicates with the backend via HTTP API calls.
  It never directly imports or calls any service code.
  WHY? Clean separation of concerns. The frontend could be replaced with
  a React app, mobile app, or CLI tool without changing the backend.

FEATURES:
  - Sidebar: Document management (upload, process, list documents)
  - Main area: Chat interface with streaming responses
  - Citation panel: Expandable source references per answer
  - Session state: Persistent conversation history
"""

import time
import uuid
from pathlib import Path

import requests
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Seminar RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State Initialization ─────────────────────────────────────────────
# Streamlit reruns the entire script on every interaction.
# st.session_state persists data across reruns.

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, citations, metadata}]

if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []

if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None


# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e8e8f0;
    }

    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        color: white;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .assistant-message {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 85%;
        color: #e8e8f0;
        font-size: 0.95rem;
        backdrop-filter: blur(10px);
    }

    /* Citation card */
    .citation-card {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85rem;
        color: #a8b5ff;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        font-size: 0.8rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.8);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Streamlit button overrides */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    /* Input field */
    .stTextInput > div > div > input, .stTextArea textarea {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        color: white;
    }

    /* Expander */
    details {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def api_post(endpoint: str, **kwargs) -> dict | None:
    """Makes a POST request to the API with error handling."""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        response = requests.post(url, timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Is the backend running? `uvicorn app.main:app --reload`")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The document may be too large.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"API Error: {detail}")
        return None


def api_get(endpoint: str, **kwargs) -> dict | None:
    """Makes a GET request to the API with error handling."""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        response = requests.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def load_documents():
    """Fetches the list of ingested documents from the API."""
    data = api_get("ingest/documents")
    if data:
        st.session_state.ingested_docs = data.get("documents", [])
    return st.session_state.ingested_docs


# ─── Sidebar: Document Management ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## My Own RAG")
    st.markdown("*NotebookLM")
    st.divider()

    # ── Upload Section ──────────────────────────────────────────────────────
    st.markdown("###  Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF report",
        type=["pdf"],
        help="Upload a seminar report to analyze",
        label_visibility="collapsed",
    )

    if uploaded_file:
        col1, col2 = st.columns(2)
        file_size_mb = uploaded_file.size / (1024 * 1024)
        col1.metric("Size", f"{file_size_mb:.1f} MB")
        col2.metric("Type", "PDF")

        if st.button("🚀 Upload & Process", use_container_width=True):
            with st.status("Processing document...", expanded=True) as status:
                # Step 1: Upload
                st.write("📤 Uploading PDF...")
                upload_result = api_post(
                    "ingest/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                )

                if upload_result:
                    doc_id = upload_result["doc_id"]
                    st.write(f"✅ Uploaded: `{upload_result['filename']}`")

                    # Step 2: Ingest
                    st.write("🔍 Parsing PDF and extracting text...")
                    st.write("✂️ Chunking document...")
                    st.write("🧠 Generating embeddings (this may take a minute)...")

                    ingest_result = api_post(f"ingest/process/{doc_id}")

                    if ingest_result:
                        st.write(f"✅ Stored {ingest_result['chunks_stored']} chunks in vector DB")
                        status.update(label="✅ Document ready for querying!", state="complete")

                        # Store selected doc
                        st.session_state.selected_doc_id = doc_id
                        load_documents()
                        st.rerun()

    st.divider()

    # ── Document List ───────────────────────────────────────────────────────
    st.markdown("### 📋 Ingested Documents")

    docs = load_documents()
    if docs:
        for doc in docs:
            is_selected = st.session_state.selected_doc_id == doc["doc_id"]
            icon = "🟢" if is_selected else "📄"

            with st.expander(f"{icon} {doc['source']}", expanded=False):
                st.markdown(f"""
                - **Pages:** {doc['total_pages']}
                - **Chunks:** {doc['chunk_count']}
                - **ID:** `{doc['doc_id'][:16]}...`
                """)
                col1, col2 = st.columns(2)
                if col1.button("Select", key=f"sel_{doc['doc_id']}", use_container_width=True):
                    st.session_state.selected_doc_id = doc["doc_id"]
                    st.rerun()
                if col2.button("Delete", key=f"del_{doc['doc_id']}", use_container_width=True):
                    try:
                        r = requests.delete(f"{API_BASE_URL}/ingest/{doc['doc_id']}")
                        if r.ok:
                            st.success("Deleted!")
                            load_documents()
                            st.rerun()
                    except Exception:
                        st.error("Delete failed")
    else:
        st.info("No documents ingested yet. Upload a PDF to get started.")

    st.divider()

    # ── Settings ────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    if st.session_state.selected_doc_id:
        selected_name = next(
            (d["source"] for d in docs if d["doc_id"] == st.session_state.selected_doc_id),
            "Unknown",
        )
        st.success(f"Active: **{selected_name}**")
    else:
        st.warning("No document selected — will search all documents")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ── Health Status ───────────────────────────────────────────────────────
    st.divider()
    health = api_get("health")
    if health:
        status_color = "🟢" if health.get("status") == "healthy" else "🟡"
        st.markdown(f"{status_color} Backend: **{health.get('status', 'unknown')}**")
        st.markdown(f"📦 Chunks in DB: **{health.get('checks', {}).get('total_chunks', 0)}**")
    else:
        st.markdown("🔴 Backend: **offline**")


# ─── Main Area: Chat Interface ────────────────────────────────────────────────

st.markdown("## 💬 Seminar Report Assistant")
st.markdown("Ask questions about your uploaded seminar reports. I'll answer with page-level citations.")

# ── Render Chat History ──────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-message">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Show citations
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📎 {len(citations)} Source(s)", expanded=False):
                    for c in citations:
                        validity = "✅" if c.get("is_valid") else "⚠️ Unverified"
                        st.markdown(f"""
<div class="citation-card">
{validity} <strong>{c.get('source', 'Unknown')}</strong> — Page {c.get('page_number', '?')}<br>
<em>{c.get('excerpt', '')}</em>
</div>
""", unsafe_allow_html=True)

            # Show performance metrics
            perf = msg.get("performance", {})
            meta = msg.get("metadata", {})
            if perf:
                cols = st.columns(4)
                cols[0].markdown(f'<div class="metric-card">⏱️ {perf.get("total_latency_ms", 0):.0f}ms</div>', unsafe_allow_html=True)
                cols[1].markdown(f'<div class="metric-card">🎯 {meta.get("chunks_after_reranking", 0)} chunks</div>', unsafe_allow_html=True)
                cols[2].markdown(f'<div class="metric-card">🔢 {perf.get("input_tokens", 0) + perf.get("output_tokens", 0)} tokens</div>', unsafe_allow_html=True)
                cols[3].markdown(f'<div class="metric-card">💰 ${perf.get("estimated_cost_usd", 0):.5f}</div>', unsafe_allow_html=True)


# ── Input Area ───────────────────────────────────────────────────────────────

st.divider()

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    user_input = col1.text_area(
        "Ask a question...",
        placeholder="e.g., What is the main contribution of this seminar report?",
        height=80,
        label_visibility="collapsed",
    )
    submitted = col2.form_submit_button("Send 🚀", use_container_width=True)

# ── Example Prompts ──────────────────────────────────────────────────────────
st.markdown("**Try asking:**")
example_cols = st.columns(3)
examples = [
    "What is the main topic of this report?",
    "Summarize the key findings",
    "What methodology was used?",
]
for i, example in enumerate(examples):
    if example_cols[i].button(example, key=f"ex_{i}", use_container_width=True):
        user_input = example
        submitted = True


# ── Handle Submission ────────────────────────────────────────────────────────
if submitted and user_input and user_input.strip():
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    # Build conversation history for multi-turn
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # Exclude current question
        if m["role"] in ("user", "assistant")
    ]

    with st.spinner("🔍 Searching relevant sections and generating answer..."):
        result = api_post(
            "chat",
            json={
                "question": user_input.strip(),
                "session_id": st.session_state.session_id,
                "history": history[-6:],  # Last 3 exchanges
                "filter_doc_id": st.session_state.selected_doc_id,
            },
        )

    if result:
        # Add assistant message with metadata
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", "I couldn't generate an answer."),
            "citations": result.get("citations", []),
            "performance": result.get("performance", {}),
            "metadata": result.get("metadata", {}),
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ Failed to get a response. Please check the backend is running.",
            "citations": [],
        })

    st.rerun()
