"""
RAG Engine Module
-----------------
Handles the LangChain RAG pipeline, including memory management and LLM configuration.
"""

import os
import logging
from typing import Any, Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)

_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question, which might reference "
     "the chat history, reformulate it as a standalone question. "
     "Do NOT answer — return it as-is if no reformulation is needed."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for question-answering tasks. "
     "Use the retrieved context below to answer concisely. "
     "If the answer isn't in the documents, say so.\n\n{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def _build_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0,
    )


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _resolve_api_key(api_key: Optional[str]) -> str:
    resolved = api_key or os.getenv("GOOGLE_API_KEY")
    if not resolved:
        logger.error("Google API key is missing.")
        raise ValueError("Provide GOOGLE_API_KEY as an argument or environment variable.")
    return resolved


def get_rag_chain(vector_store: Any, google_api_key: Optional[str] = None) -> RunnableLambda:
    """
    Constructs a stateful RAG pipeline with conversational memory using LCEL.

    Args:
        vector_store: A LangChain-compatible vector store.
        google_api_key: Optional API key (falls back to GOOGLE_API_KEY env var).

    Returns:
        A RunnableLambda that accepts {"input": str, "chat_history": list}
        and returns {"answer": str, "context": list[Document]}.
    """
    api_key = _resolve_api_key(google_api_key)
    llm = _build_llm(api_key)

    contextualize_chain = _CONTEXTUALIZE_PROMPT | llm | StrOutputParser()
    qa_chain = _QA_PROMPT | llm | StrOutputParser()
    retriever: VectorStoreRetriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def rag_logic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        standalone_question = contextualize_chain.invoke(input_dict)
        docs = retriever.invoke(standalone_question)

        answer = qa_chain.invoke({
            "context": _format_docs(docs),
            "chat_history": input_dict["chat_history"],
            "input": input_dict["input"],
        })

        return {"answer": answer, "context": docs}

    return RunnableLambda(rag_logic)