"""Central configuration for the GovMind RAG ingestion pipeline."""
from pathlib import Path

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(".")

POLICY_DIR = BASE_DIR / "policy_pdfs"
CIRCULAR_DIR = BASE_DIR / "circulars"
INSPECTION_DIR = BASE_DIR / "inspection_reports"
COMPLAINT_DIR = BASE_DIR / "complaints_csv"
BUDGET_DIR = BASE_DIR / "budget_csv"

PERSIST_DIR = Path("./chroma_store")

DOCS = {
    "policy": [
        "HFW_Dengue_VBD_Advisory_2024.pdf",
        "Odisha_EV_Policy_2021_Extension_2026.pdf",
    ],
    "circular": [
        "Finance_Dept_DA_Hike_Circular_2024.pdf",
        "GA_PG_Contractual_Remuneration_Circular_2024.pdf",
    ],
    "inspection": [
        "CCI_Inspection_Report_Mayurbhanj_2024.pdf",
        "School_Infrastructure_Inspection_Ganjam_2024.pdf",
    ],
    "complaints_csv": "citizen_complaints_odisha_2024.csv",
    "budget_csv": "budget_allocation_utilization_odisha_2023_24.csv",
}

# --- Chunking ------------------------------------------------------------
# Token counts approximated via whitespace splitting (swap in a real
# tokenizer, e.g. tiktoken, if you need exact token boundaries).
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# --- Vector store ----------------------------------------------------
COLLECTION_NAME = "govmind_odisha"

# --- Embeddings --------------------------------------------------------
# "chroma_default" downloads a small ONNX MiniLM model on first run (needs
# network access to Chroma's model host). "tfidf" is a pure local fallback
# with no downloads, useful for offline dev/smoke-testing the pipeline.
EMBEDDING_BACKEND = "chroma_default"  # or "tfidf"
