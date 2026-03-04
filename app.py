import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import extract_text, build_vectorstore, get_answer

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a cleaner look ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .stChatMessage { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Pipeline")
    st.caption("Upload a document · Ask anything")
    st.divider()

    # API key (falls back to .env if set)
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )

    model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        help="Larger models give better answers; smaller ones are faster.",
    )

    st.divider()
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, Word (.docx), plain text",
    )

    process_btn = st.button("⚡ Process Document", type="primary", use_container_width=True)

# ── Document processing ───────────────────────────────────────────────────────
if process_btn:
    if not groq_api_key:
        st.sidebar.error("Please enter your Groq API key.")
    elif not uploaded_file:
        st.sidebar.error("Please upload a document first.")
    else:
        with st.sidebar:
            with st.spinner("Extracting text & building index…"):
                ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

                # Write to a temp file so pypdf / python-docx can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    text = extract_text(tmp_path, ext)
                finally:
                    os.unlink(tmp_path)

                if not text.strip():
                    st.error("Could not extract any text from this file.")
                else:
                    vectorstore = build_vectorstore(text)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.messages = []          # reset chat
                    st.success(f"✅ Ready — {uploaded_file.name}")

# ── Main chat area ────────────────────────────────────────────────────────────
if "vectorstore" in st.session_state:

    st.markdown(f"### 💬 Chat with **{st.session_state.doc_name}**")
    st.divider()

    # Replay chat history
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Source passages"):
                    for i, chunk in enumerate(msg["sources"], 1):
                        st.markdown(f"**Passage {i}**")
                        preview = chunk[:500] + ("…" if len(chunk) > 500 else "")
                        st.text(preview)

    # Chat input
    if query := st.chat_input("Ask a question about your document…"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = get_answer(
                    st.session_state.vectorstore,
                    query,
                    groq_api_key,
                    model,
                )
            answer = result["result"]
            sources = [doc.page_content for doc in result.get("source_documents", [])]

            st.markdown(answer)

            if sources:
                with st.expander("📚 Source passages"):
                    for i, chunk in enumerate(sources, 1):
                        st.markdown(f"**Passage {i}**")
                        preview = chunk[:500] + ("…" if len(chunk) > 500 else "")
                        st.text(preview)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

else:
    # Welcome screen
    st.markdown("## Welcome to RAG Document Q&A 👋")
    st.markdown(
        """
        **How it works:**
        1. Enter your **Groq API key** in the sidebar
        2. **Upload** a PDF, DOCX, or TXT file
        3. Click **Process Document**
        4. Start **asking questions** — the model will answer using only your document

        > Embeddings are generated locally (no data sent for embedding). Only your query
        > and the relevant passages are sent to Groq.
        """
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Supported formats", "PDF · DOCX · TXT")
    col2.metric("Embedding model", "all-MiniLM-L6-v2")
    col3.metric("Vector store", "FAISS (in-memory)")
