"""
RAG Chatbot Entry Point
-----------------------
Main entry point for the Streamlit application.
Orchestrates UI components and manages session initialization.
"""

import streamlit as st
from ui_components import (
    setup_page_config,
    sidebar_configuration,
    chat_interface,
    vector_inspector,
)


def main() -> None:
    setup_page_config()
    sidebar_configuration()

    tab1, tab2 = st.tabs(["💬 Chat", "🔍 Vector DB Inspector"])

    with tab1:
        chat_interface()

    with tab2:
        vector_inspector()


if __name__ == "__main__":
    main()