"""
UI Components Module
--------------------
Contains modular Streamlit UI components for the RAG Chatbot.
"""

import os
import tempfile
import logging
import streamlit as st
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from document_processor import process_pdf
from vector_store_manager import get_vector_store, delete_vector_db, get_vector_data
from rag_engine import get_rag_chain

logger = logging.getLogger(__name__)

# --- Session State Keys ---
KEY_VECTOR_STORE = "vector_store"
KEY_RAG_CHAIN = "rag_chain"
KEY_MESSAGES = "messages"

_SESSION_KEYS = [KEY_VECTOR_STORE, KEY_RAG_CHAIN, KEY_MESSAGES]


# --- Helpers ---

def _build_chat_history(messages: list) -> List[BaseMessage]:
    """Converts stored message dicts into LangChain message objects."""
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


def _extract_source_pages(context: list) -> str:
    """Returns a formatted source annotation from retrieved document chunks."""
    pages = sorted({doc.metadata.get("page", 0) + 1 for doc in context})
    return f"\n\n*(Sources: Page {', '.join(map(str, pages))})*"


def _save_temp_pdf(uploaded_file) -> str:
    """Writes an uploaded file to a temp path and returns the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def _clear_session() -> None:
    for key in _SESSION_KEYS:
        st.session_state.pop(key, None)


# --- Page Setup ---

def setup_page_config() -> None:
    """Configures the Streamlit page metadata and layout."""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🤖 RAG Chatbot")
    st.markdown("---")


# --- Sidebar ---

def sidebar_configuration() -> None:
    """Handles sidebar uploads and settings."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("🔑 API Key is managed via backend environment variables.")
        st.divider()

        st.header("📄 Knowledge Base")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Process", use_container_width=True):
                handle_document_processing(uploaded_file)
        with col2:
            if st.button("🗑️ Reset", use_container_width=True):
                _handle_reset()


def _handle_reset() -> None:
    vs = st.session_state.get(KEY_VECTOR_STORE)
    if delete_vector_db(vector_store=vs):
        _clear_session()
        st.success("Database reset!")
        st.rerun()


# --- Document Processing ---

def handle_document_processing(uploaded_file) -> None:
    """Processes an uploaded PDF and initialises the RAG chain."""
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing GOOGLE_API_KEY in .env file!")
        return

    if not uploaded_file:
        st.warning("Please upload a PDF first.")
        return

    tmp_path = _save_temp_pdf(uploaded_file)
    status = st.empty()

    try:
        with st.spinner("⏳ Processing..."):
            status.info("📑 Splitting chunks...")
            chunks = process_pdf(tmp_path)

            if not chunks:
                st.warning("⚠️ PDF is empty or contains no extractable text.")
                return

            status.info(f"💾 Saving {len(chunks)} chunks...")
            vector_store = get_vector_store(chunks)
            st.session_state[KEY_VECTOR_STORE] = vector_store
            st.session_state[KEY_RAG_CHAIN] = get_rag_chain(vector_store)

            status.empty()
            st.success(f"✅ Indexed {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        st.error(f"Processing error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- Chat Interface ---

def chat_interface() -> None:
    """Manages chat messages and user input."""
    if KEY_MESSAGES not in st.session_state:
        st.session_state[KEY_MESSAGES] = []

    for message in st.session_state[KEY_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state[KEY_MESSAGES].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if KEY_RAG_CHAIN in st.session_state:
            _handle_rag_response(prompt)
        else:
            with st.chat_message("assistant"):
                st.info("👋 Upload and process a document to start chatting.")


def _handle_rag_response(prompt: str) -> None:
    """Invokes the RAG chain and renders the assistant response."""
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching..."):
            try:
                chain = st.session_state[KEY_RAG_CHAIN]
                history = _build_chat_history(st.session_state[KEY_MESSAGES][:-1])
                result = chain.invoke({"input": prompt, "chat_history": history})

                response = result["answer"]
                if result.get("context"):
                    response += _extract_source_pages(result["context"])

                st.markdown(response)
                st.session_state[KEY_MESSAGES].append({"role": "assistant", "content": response})
            except Exception as e:
                logger.error(f"RAG chain invocation failed: {e}")
                st.error(f"❌ Error: {e}")


# --- Vector Inspector ---

def vector_inspector() -> None:
    """Visualisation tab for the vector store."""
    st.header("🔍 Vector Store Inspector")

    if KEY_VECTOR_STORE not in st.session_state:
        st.info("No vector store loaded.")
        return

    try:
        vs = st.session_state[KEY_VECTOR_STORE]
        db_data = get_vector_data(vs)

        st.metric("Total Chunks", len(db_data))

        for entry in db_data:
            with st.expander(f"📦 Chunk: {entry['id']}"):
                st.text(entry["text"])
                st.json(entry["metadata"])
    except Exception as e:
        logger.error(f"Vector inspector error: {e}")
        st.error(f"Error reading vector store: {e}")