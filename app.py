"""
GovMind — Streamlit interface for the RAG chatbot.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ---------------------------------------------------------
# Paths / environment
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from rag import RAGPipeline


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="GovMind",
    page_icon="🏛️",
    layout="centered",
)


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

def main():

    st.title("GovMind")

    st.caption(
        "AI-powered question answering over Odisha government "
        "policies, circulars, inspection reports, and public data."
    )

    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------

    with st.sidebar:

        st.header("GovMind")

        st.write(
            "Ask questions about the documents in the GovMind "
            "knowledge base."
        )

        k = st.slider(
            "Number of sources",
            min_value=1,
            max_value=10,
            value=10,
        )

        st.divider()

        st.caption("Knowledge base")
        st.write("Odisha government documents and datasets")

    # -----------------------------------------------------
    # Question input
    # -----------------------------------------------------

    question = st.text_input(
        "Ask a question",
        placeholder=(
            "e.g. What are the recommendations in the dengue advisory?"
        ),
    )

    search_clicked = st.button(
        "Ask GovMind",
        type="primary",
    )

    # -----------------------------------------------------
    # Answer
    # -----------------------------------------------------

    if search_clicked:

        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Searching the knowledge base..."):

            try:
                # Use the selected number of sources.
                rag = RAGPipeline(k=k)

                # RAGPipeline.answer() returns a dictionary.
                result = rag.answer(question)

            except Exception as e:
                st.error(
                    "Something went wrong while answering the question."
                )
                st.code(str(e))
                return

        # -------------------------------------------------
        # Display answer
        # -------------------------------------------------

        st.subheader("Answer")

        answer = result.answer

        if answer:
            st.write(answer)
        else:
            st.info("No answer was generated.")

        # -------------------------------------------------
        # Display sources
        # -------------------------------------------------

        st.subheader("Sources")

        chunks = result.chunks
        sources = result.get("sources", [])

        if not chunks:
            st.info("No relevant sources were found.")

        else:

            for i, chunk in enumerate(chunks, start=1):

                # RetrievedChunk fields
                metadata = getattr(chunk, "metadata", {}) or {}

                source_file = (
                    metadata.get("source_file")
                    or metadata.get("source")
                    or "Unknown source"
                )

                doc_type = (
                    metadata.get("doc_type")
                    or metadata.get("category")
                    or "unknown"
                )

                score = getattr(chunk, "final_score", None)

                if score is None:
                    score_text = "N/A"
                else:
                    score_text = f"{score:.3f}"

                # RetrievedChunk has no guaranteed `citation` attribute,
                # so construct the citation ourselves.
                citation = (
                    f"{source_file} "
                    f"({doc_type}) "
                    f"(score ~{score_text})"
                )

                with st.expander(
                    f"[{i}] {citation}"
                ):

                    st.write(chunk.text)

                    # Show useful metadata when available.
                    if metadata:

                        st.caption("Document metadata")

                        metadata_display = {
                            key: value
                            for key, value in metadata.items()
                            if value is not None
                        }

                        st.json(metadata_display)


# ---------------------------------------------------------
# Run application
# ---------------------------------------------------------

if __name__ == "__main__":
    main()

