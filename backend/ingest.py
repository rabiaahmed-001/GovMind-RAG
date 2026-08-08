"""
GovMind RAG — ingestion entrypoint.

Usage:
    python ingest.py                # full ingestion, chroma_default embeddings
    python ingest.py --backend tfidf --rebuild
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid

import chromadb

import config
from chunker import chunk_documents
from embeddings import get_embedding_function
from loaders import Document, load_all_pdfs, load_budget_csv, load_complaints_csv


def load_all_documents() -> list[Document]:
    docs: list[Document] = []

    docs.extend(load_all_pdfs(config.POLICY_DIR, {
        "policy": config.DOCS["policy"],
    }))

    docs.extend(load_all_pdfs(config.CIRCULAR_DIR, {
        "circular": config.DOCS["circular"],
    }))

    docs.extend(load_all_pdfs(config.INSPECTION_DIR, {
        "inspection": config.DOCS["inspection"],
    }))

    docs.extend(
        load_complaints_csv(
            config.COMPLAINT_DIR / config.DOCS["complaints_csv"]
        )
    )

    docs.extend(
        load_budget_csv(
            config.BUDGET_DIR / config.DOCS["budget_csv"]
        )
    )

    return docs


def build_index(backend: str, rebuild: bool) -> None:
    t0 = time.time()

    print("Loading source documents...")
    raw_docs = load_all_documents()
    print(f"  {len(raw_docs)} source documents loaded")

    print("Chunking...")
    chunks = chunk_documents(
        raw_docs, config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS
    )
    print(f"  {len(chunks)} chunks produced")

    print(f"Connecting to Chroma at {config.PERSIST_DIR} ...")
    client = chromadb.PersistentClient(path=str(config.PERSIST_DIR))
    embedding_fn = get_embedding_function(backend)

    if rebuild:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            print(f"  Dropped existing collection '{config.COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding + upserting {len(chunks)} chunks (backend={backend})...")
    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        collection.upsert(
            ids=[str(uuid.uuid4()) for _ in batch],
            documents=[c.text for c in batch],
            metadatas=[_clean_metadata(c.metadata) for c in batch],
        )
        print(f"  {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print(f"Done in {time.time() - t0:.1f}s. Collection count: {collection.count()}")


def _clean_metadata(meta: dict) -> dict:
    # Chroma metadata values must be str/int/float/bool — drop Nones.
    return {k: v for k, v in meta.items() if v is not None}


def quick_search(query: str, backend: str, k: int = 5) -> None:
    client = chromadb.PersistentClient(path=str(config.PERSIST_DIR))
    embedding_fn = get_embedding_function(backend)
    collection = client.get_collection(
        name=config.COLLECTION_NAME, embedding_function=embedding_fn
    )
    results = collection.query(query_texts=[query], n_results=k)
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        print(f"\n[{i+1}] dist={dist:.4f} source={meta.get('source_file')}")
        print(f"    {doc[:220]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GovMind RAG ingestion")
    parser.add_argument("--backend", default=config.EMBEDDING_BACKEND, choices=["chroma_default", "tfidf"])
    parser.add_argument("--rebuild", action="store_true", help="drop and recreate the collection")
    parser.add_argument("--query", help="run a quick test search after ingesting (or standalone)")
    parser.add_argument("--search-only", action="store_true", help="skip ingestion, just run --query")
    args = parser.parse_args()

    if not args.search_only:
        build_index(backend=args.backend, rebuild=args.rebuild)

    if args.query:
        print("\n--- quick search ---")
        quick_search(args.query, backend=args.backend)

    sys.exit(0)
