"""
Vector Store Manager Module
---------------------------
Manages the lifecycle of the Chroma vector database.
"""

import os
import shutil
import logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# --- Constants ---

PERSIST_DIRECTORY = "./chroma_db"

_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# --- Public API ---

def get_vector_store(
    chunks: Optional[List[Document]] = None,
    persist_directory: str = PERSIST_DIRECTORY,
) -> Chroma:
    """
    Creates, appends to, or loads a Chroma vector store.

    Args:
        chunks: Documents to index. If None, loads the existing store.
        persist_directory: Path to the Chroma persistence directory.

    Returns:
        An initialised Chroma vector store.
    """
    if chunks:
        return _upsert_chunks(chunks, persist_directory)
    return _load_store(persist_directory)


def delete_vector_db(
    vector_store: Optional[Chroma] = None,
    persist_directory: str = PERSIST_DIRECTORY,
) -> bool:
    """
    Deletes the Chroma collection and its persistence directory.
    Handles Windows file locks gracefully.

    Returns:
        True if the operation succeeded, False otherwise.
    """
    try:
        if vector_store is not None:
            logger.info("Deleting Chroma collection via API.")
            vector_store.delete_collection()

        if os.path.exists(persist_directory):
            logger.warning(f"Removing persistence directory: {persist_directory}")
            shutil.rmtree(persist_directory, ignore_errors=True)

            if os.path.exists(persist_directory):
                logger.warning("Directory still exists — likely a file lock (Windows). Collection was cleared.")

        return True
    except Exception as e:
        logger.error(f"Failed to reset vector database: {e}")
        return False


def get_vector_data(
    vector_store: Chroma,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Retrieves stored chunks for visualisation.

    Args:
        vector_store: An active Chroma vector store.
        limit: Maximum number of entries to return.

    Returns:
        List of dicts with 'id', 'text', and 'metadata' keys.
        Returns an empty list on failure.
    """
    try:
        results = vector_store.get(limit=limit, include=["documents", "metadatas"])
        return [
            {"id": id_, "text": text, "metadata": meta}
            for id_, text, meta in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            )
        ]
    except Exception as e:
        logger.error(f"Failed to fetch vector data: {e}")
        return []


# --- Helpers ---

def _load_store(persist_directory: str) -> Chroma:
    logger.info(f"Loading existing vector store from '{persist_directory}'")
    return Chroma(persist_directory=persist_directory, embedding_function=_EMBEDDINGS)


def _upsert_chunks(chunks: List[Document], persist_directory: str) -> Chroma:
    """Adds chunks to an existing store, or creates a new one if none exists."""
    logger.info(f"Indexing {len(chunks)} chunks into '{persist_directory}'")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        store = _load_store(persist_directory)
        store.add_documents(chunks)
        return store

    return Chroma.from_documents(
        documents=chunks,
        embedding=_EMBEDDINGS,
        persist_directory=persist_directory,
    )