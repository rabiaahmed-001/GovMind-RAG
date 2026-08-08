"""
rag.py — Retrieval + answer-generation pipeline for GovMind.

Queries the EXISTING Chroma vector store (built by your working ingest.py /
loaders.py / chunker.py / embeddings.py) and generates answers grounded in
the top-k retrieved chunks, with source citations.

This module is read-only with respect to the vector store: it never
creates, seeds, or upserts into the collection, and none of your
ingestion code is touched or reimplemented here.

LLM backend: Google Gemini via the official `google-genai` SDK
(model: gemini-2.5-flash). Retrieval logic is unchanged from the OpenAI
version — only the client and generation function were swapped.

Requires:
    pip install google-genai chromadb
    GEMINI_API_KEY (or GOOGLE_API_KEY) set in the environment

Usage (standalone CLI):
    python backend/rag.py "What is the EV policy timeline?"

Usage (as a module, e.g. from app.py):
    from rag import RAGPipeline
    rag = RAGPipeline()
    result = rag.answer("your question")
    print(result.answer)
    print(result.formatted_sources())
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

import chromadb

from config import COLLECTION_NAME, EMBEDDING_BACKEND, PERSIST_DIR
from embeddings import get_embedding_function

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_K = 8  # k=5 was missing chunks needed for complete answers (e.g.
# allocation + utilization figures living in separate CSV-row chunks);
# k=8 reliably pulls both in. Bump further if you see similar gaps.

SYSTEM_PROMPT = (
    "You are GovMind, an assistant that answers questions strictly using the "
    "provided government document excerpts (policy PDFs, circulars, "
    "inspection reports, complaints and budget CSVs). Rules:\n"
    "1. Only use information present in the excerpts below. Do not use "
    "outside knowledge.\n"
    "2. If the excerpts don't contain enough information to answer, say so "
    "plainly instead of guessing.\n"
    "3. Every factual claim must be followed by a citation marker like "
    "[1], [2] referring to the numbered excerpt it came from.\n"
    "4. Be concise and factual. Do not editorialize."
)


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    distance: Optional[float]
    chunk_id: str

    @property
    def citation(self) -> str:
        """Best-effort, human-readable citation from whatever metadata your
        chunker.py attached to this chunk. Tries several common key names
        and degrades gracefully if a key is absent, instead of crashing.
        If your real metadata schema differs, this is the one place to
        adjust it."""
        meta = self.metadata or {}
        source = (
            meta.get("source")
            or meta.get("filename")
            or meta.get("file_name")
            or meta.get("doc_id")
            or "unknown source"
        )
        page = meta.get("page") or meta.get("page_number")
        row = meta.get("row") or meta.get("row_index")
        doc_type = meta.get("doc_type") or meta.get("category")

        parts = [str(source)]
        if page is not None:
            parts.append(f"page {page}")
        if row is not None:
            parts.append(f"row {row}")
        label = ", ".join(parts)
        return f"[{label}]" + (f" ({doc_type})" if doc_type else "")


@dataclass
class RAGResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    model: str

    def formatted_sources(self) -> str:
        """Numbered source list matching the [1], [2] markers in the answer."""
        lines = [f"[{i}] {c.citation}" for i, c in enumerate(self.chunks, start=1)]
        return "\n".join(lines) if lines else "No sources retrieved."


class RAGPipeline:
    """Opens the existing Chroma collection and answers questions grounded
    in the top-k retrieved chunks via Gemini (gemini-2.5-flash)."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_backend: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        k: int = DEFAULT_K,
    ):
        self.persist_dir = str(persist_dir or PERSIST_DIR)
        self.collection_name = collection_name or COLLECTION_NAME
        self.embedding_backend = embedding_backend or EMBEDDING_BACKEND
        self.model = model
        self.k = k

        # --- Vector store (read-only access to your existing collection) ---
        # UNCHANGED from the OpenAI version of this file.
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._embedding_fn = get_embedding_function(self.embedding_backend)

        try:
            self._collection = self._client.get_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not open collection '{self.collection_name}' at "
                f"'{self.persist_dir}'. Make sure your ingestion pipeline has "
                f"already been run and EMBEDDING_BACKEND in config.py matches "
                f"what was used at ingestion time. Original error: {e}"
            ) from e

        # --- LLM client (Gemini via google-genai) ---
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "The 'google-genai' package is required for rag.py. Install "
                "it with `pip install google-genai`."
            ) from e

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Set it as an "
                "environment variable (or pass api_key explicitly) before "
                "using RAGPipeline."
            )
        self._genai_client = genai.Client(api_key=key)

    # ------------------------------------------------------------------
    # Retrieval — UNCHANGED from the OpenAI version of this file.
    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Semantic search over the collection. Works uniformly across
        PDFs and CSVs since provenance lives in each chunk's metadata,
        not in separate indexes."""
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        results = self._collection.query(
            query_texts=[question],
            n_results=k or self.k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[None] * len(docs)])[0]

        return [
            RetrievedChunk(text=doc, metadata=meta or {}, distance=dist, chunk_id=cid)
            for cid, doc, meta, dist in zip(ids, docs, metas, dists)
        ]

    # ------------------------------------------------------------------
    # Answer generation — this is the part that changed (OpenAI -> Gemini).
    # ------------------------------------------------------------------
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        from google.genai import types

        context = self._build_context(chunks)
        user_prompt = (
            f"Excerpts:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the excerpts above, with [n] citation markers."
        )

        response = self._genai_client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
            ),
        )
        return response.text or ""

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "(no relevant excerpts were retrieved)"
        blocks = []
        for i, c in enumerate(chunks, start=1):
            blocks.append(f"[{i}] Source: {c.citation}\n{c.text.strip()}")
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------
    def answer(self, question: str, k: Optional[int] = None) -> RAGResult:
        chunks = self.retrieve(question, k=k)
        answer_text = self.generate(question, chunks)
        return RAGResult(question=question, answer=answer_text, chunks=chunks, model=self.model)


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python rag.py "your question here" [k]')
        sys.exit(1)

    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_K

    rag = RAGPipeline(k=k)
    result = rag.answer(question)

    print(f"Q: {result.question}\n")
    print(f"A ({result.model}):\n{result.answer}\n")
    print("Sources:")
    print(result.formatted_sources())


if __name__ == "__main__":
    main()
