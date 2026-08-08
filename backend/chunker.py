"""
Chunking strategy:
- PDF documents (circulars/policy/inspection): sliding window over
  whitespace tokens, ~500 tokens/chunk with 50 token overlap, so no
  chunk straddles a full page boundary and context isn't lost at splits.
- CSV row documents (complaints/budget): already atomic (one row = one
  fact), so they pass through unchanged. Chunking a 2-sentence row would
  just fragment it for no benefit.
"""
from __future__ import annotations

from loaders import Document


def _split_tokens(text: str, size: int, overlap: int) -> list[str]:
    tokens = text.split()
    if len(tokens) <= size:
        return [text]

    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size >= len(tokens):
            break
    return chunks


def chunk_document(
    doc: Document, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    if doc.metadata.get("doc_type") in ("complaint", "budget"):
        return [doc]  # atomic row-level docs, don't sub-chunk

    pieces = _split_tokens(doc.text, chunk_size, chunk_overlap)
    out = []
    for i, piece in enumerate(pieces):
        meta = dict(doc.metadata)
        meta["chunk_index"] = i
        meta["chunk_count"] = len(pieces)
        out.append(Document(text=piece, metadata=meta))
    return out


def chunk_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    result: list[Document] = []
    for doc in docs:
        result.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return result
