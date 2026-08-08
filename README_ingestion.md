# GovMind RAG — Ingestion Pipeline

## Layout
- `config.py` — paths, chunk size/overlap, collection name, embedding backend
- `loaders.py` — PDF loader (circulars/policy/inspection, with table extraction
  and letter-no/date/subject metadata parsing) + CSV row-to-text loaders for
  complaints and budget data
- `chunker.py` — 500-token / 50-token-overlap sliding window for PDFs; CSV rows
  pass through unchunked (already atomic facts)
- `embeddings.py` — pluggable backend: `chroma_default` (ONNX MiniLM, needs
  network on first run) or `tfidf` (local-only fallback for offline dev)
- `ingest.py` — orchestrates load → chunk → embed → upsert into a persistent
  Chroma collection at `./chroma_store`
- `eval_questions.json` — 25 eval questions (single vs multi-doc, easy/med/hard)

## Run it
```bash
pip install -r requirements.txt

# Real semantic embeddings (needs outbound network for the model download):
python ingest.py --rebuild

# Offline smoke test (no network, TF-IDF fallback):
python ingest.py --backend tfidf --rebuild --query "dengue hotspot districts"
```

Verified locally in this sandbox with the `tfidf` backend: 8 PDFs/CSVs → 252
chunks ingested, and a query for "Cuttack water supply complaint" correctly
surfaces the matching complaint rows *and* the corresponding low-utilization
PHE/Jal Jeevan Mission budget row — confirming metadata and text are aligned
for the cross-document eval questions.

## Known gaps / next steps
1. **Swap `tfidf` for a real embedding model** before evaluating retrieval
   quality — TF-IDF is bag-of-words and won't generalize past exact
   vocabulary matches. Point `EMBEDDING_BACKEND` at `chroma_default` (or an
   API-based embedding model) once you have network access to the model host.
2. **Retriever + eval harness** — `ingest.py` only builds the index. Next:
   a `retrieve.py` (top-k + optional metadata filters, e.g. district/dept)
   and a `run_eval.py` that runs `eval_questions.json` through it and scores
   retrieval hit-rate against `expected_sources`.
3. **Tokenizer** — chunking currently splits on whitespace as a token proxy;
   swap in `tiktoken` if you need exact token-count guarantees for your LLM's
   context window.
4. **Table fidelity** — inspection-report tables extract via pdfplumber but
   some rows in the source PDFs are truncated mid-cell (e.g. "Fill post
   withi...") — worth a manual QA pass on the two inspection reports before
   trusting exact figures pulled from their tables.
