"""
Document Processor Module
-------------------------
Handles loading PDF files and splitting them into manageable text chunks.
"""

import os
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)


def _validate_path(pdf_path: str) -> None:
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        raise FileNotFoundError(f"The file '{pdf_path}' was not found.")


def _load_pdf(pdf_path: str) -> List[Document]:
    logger.info(f"Loading PDF: {pdf_path}")
    docs = list(PyPDFLoader(pdf_path).lazy_load())
    logger.info(f"Loaded {len(docs)} pages from '{pdf_path}'")
    return docs


def process_pdf(pdf_path: str) -> List[Document]:
    """
    Loads a PDF and splits it into text chunks.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of LangChain Document chunks, or an empty list if no text is extracted.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    _validate_path(pdf_path)

    docs = _load_pdf(pdf_path)

    if not docs:
        logger.warning(f"No text extracted from '{pdf_path}'. PDF may be empty or image-based.")
        return []

    chunks = _TEXT_SPLITTER.split_documents(docs)
    logger.info(f"Split {len(docs)} pages into {len(chunks)} chunks.")
    return chunks