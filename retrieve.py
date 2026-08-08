"""
retrieve.py — Hybrid retrieval layer for GovMind.

Read-only access to the existing Chroma vector store.

Pipeline:
    Query
      ↓
    Semantic retrieval
      ↓
    Lexical scoring
      ↓
    Entity scoring
      ↓
    Metadata scoring
      ↓
    Intent scoring
      ↓
    Final reranking
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chromadb


# ============================================================================
# PATHS / BACKEND
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import COLLECTION_NAME, EMBEDDING_BACKEND, PERSIST_DIR
from embeddings import get_embedding_function


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    distance: Optional[float]
    chunk_id: str

    semantic_score: float = 0.0
    lexical_score: float = 0.0
    entity_score: float = 0.0
    metadata_score: float = 0.0
    intent_score: float = 0.0
    final_score: float = 0.0

    @property
    def citation(self) -> str:

        meta = self.metadata or {}

        source = (
            meta.get("source")
            or meta.get("source_file")
            or meta.get("filename")
            or meta.get("file_name")
            or meta.get("doc_id")
            or "unknown source"
        )

        page = meta.get("page") or meta.get("page_number")
        row = meta.get("row") or meta.get("row_index")

        doc_type = (
            meta.get("doc_type")
            or meta.get("category")
        )

        parts = [str(source)]

        if page is not None:
            parts.append(f"page {page}")

        if row is not None:
            parts.append(f"row {row}")

        citation = f"[{', '.join(parts)}]"

        if doc_type:
            citation += f" ({doc_type})"

        return citation


# ============================================================================
# TEXT UTILITIES
# ============================================================================

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "has", "have", "how", "in", "is",
    "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "which", "who",
    "with", "what", "when", "where", "why", "people",
    "person", "persons",
}


def normalize_text(text: str) -> str:
    text = str(text or "").lower()

    text = re.sub(
        r"[^a-z0-9₹]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in normalize_text(text).split()
        if token not in STOPWORDS and len(token) > 1
    ]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


# ============================================================================
# INTENT DETECTION
# ============================================================================

def detect_intent(question: str) -> str:

    q = normalize_text(question)

    if any(term in q for term in (
        "citizen complaint",
        "citizen complaints",
        "complaint",
        "complaints",
        "grievance",
        "grievances",
    )):
        return "complaint"

    if any(term in q for term in (
        "budget",
        "allocation",
        "allocated",
        "utilized",
        "utilisation",
        "utilization",
        "unutilized",
        "spending",
        "expenditure",
        "funds",
        "funding",
    )):
        return "budget"

    if any(term in q for term in (
        "da hike",
        "da increase",
        "dearness allowance",
        "dearness",
        "da rate",
        "revised dearness allowance",
        "revised da",
    )):
        return "circular"

    if any(term in q for term in (
        "cci",
        "ccis",
        "child care institution",
        "child care institutions",
        "inspection",
        "inspected",
        "inspection report",
        "fire safety",
        "fire safety noc",
        "safety noc",
        "noc",
        "deficiency",
        "risk level",
        "action directed",
        "action was directed",
        "directed action",
        "directed for",
        "action required",
        "corrective action",
        "renew noc",
        "renewal of noc",
        "interim fire drill",
        "nirmal nivas",
        "mayurbhanj cci",
    )):
        return "inspection"

    return "general"


# ============================================================================
# RETRIEVER
# ============================================================================

class Retriever:

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_backend: Optional[str] = None,
    ):

        self.persist_dir = str(
            persist_dir or PERSIST_DIR
        )

        self.collection_name = (
            collection_name or COLLECTION_NAME
        )

        self.embedding_backend = (
            embedding_backend or EMBEDDING_BACKEND
        )

        print(
            f"Opening Chroma collection: "
            f"{self.collection_name}"
        )

        print(
            f"Persist directory: "
            f"{self.persist_dir}"
        )

        print(
            f"Embedding backend: "
            f"{self.embedding_backend}"
        )

        self._client = chromadb.PersistentClient(
            path=self.persist_dir
        )

        self._embedding_fn = get_embedding_function(
            self.embedding_backend
        )

        try:

            self._collection = self._client.get_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
            )

        except Exception as exc:

            raise RuntimeError(
                f"\nCould not open collection "
                f"'{self.collection_name}' "
                f"at '{self.persist_dir}'.\n\n"
                f"Make sure ingest.py has been run "
                f"and the embedding backend matches.\n\n"
                f"Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------------
    # Semantic retrieval
    # ------------------------------------------------------------------------

    def _semantic_candidates(
        self,
        question: str,
        candidate_k: int,
        where: Optional[dict] = None,
    ) -> list[RetrievedChunk]:

        count = self.count()

        if count == 0:
            return []

        candidate_k = min(
            max(candidate_k, 1),
            count,
        )

        results = self._collection.query(
            query_texts=[question],
            n_results=candidate_k,
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get(
            "distances",
            [[None] * len(docs)],
        )[0]

        return [
            RetrievedChunk(
                text=doc or "",
                metadata=meta or {},
                distance=distance,
                chunk_id=chunk_id,
            )
            for chunk_id, doc, meta, distance
            in zip(
                ids,
                docs,
                metas,
                distances,
            )
        ]

    # ------------------------------------------------------------------------
    # Lexical score
    # ------------------------------------------------------------------------

    def _lexical_score(
        self,
        question: str,
        chunk: RetrievedChunk,
    ) -> float:

        query_tokens = token_set(question)

        if not query_tokens:
            return 0.0

        metadata = chunk.metadata or {}

        metadata_text = " ".join(
            str(metadata.get(key, ""))
            for key in (
                "source",
                "source_file",
                "filename",
                "file_name",
                "subject",
                "department",
                "doc_type",
                "category",
                "letter_no",
            )
        )

        full_text = normalize_text(
            metadata_text + " " + chunk.text
        )

        document_tokens = set(
            full_text.split()
        )

        if not document_tokens:
            return 0.0

        overlap = len(
            query_tokens & document_tokens
        ) / len(query_tokens)

        # Exact phrase bonus.
        question_normalized = normalize_text(question)

        phrase_bonus = 0.0

        important_phrases = [
            "nirmal nivas",
            "fire safety noc",
            "jal jeevan mission",
            "dearness allowance",
            "piped water",
            "citizen complaint",
        ]

        for phrase in important_phrases:

            if (
                phrase in question_normalized
                and phrase in full_text
            ):
                phrase_bonus += 0.15

        return min(
            1.0,
            overlap + phrase_bonus,
        )

    # ------------------------------------------------------------------------
    # Entity score
    # ------------------------------------------------------------------------

    def _entity_score(
        self,
        question: str,
        chunk: RetrievedChunk,
    ) -> float:

        q = normalize_text(question)
        text = normalize_text(chunk.text)

        if not q or not text:
            return 0.0

        score = 0.0

        entities = [
            "nirmal nivas",
            "udala",
            "mayurbhanj",
            "cuttack",
            "ganjam",
            "koraput",
            "kalahandi",
            "jajpur",
            "puri",
            "khordha",
            "balasore",
            "bhadrak",
            "rourkela",
        ]

        for entity in entities:

            if entity in q and entity in text:
                score += 0.5

        # Exact multi-word entity gets stronger weight.
        if (
            "nirmal nivas" in q
            and "nirmal nivas" in text
        ):
            score += 0.5

        return min(
            1.0,
            score,
        )

    # ------------------------------------------------------------------------
    # Metadata score
    # ------------------------------------------------------------------------

    def _metadata_score(
        self,
        question: str,
        chunk: RetrievedChunk,
    ) -> float:

        q = normalize_text(question)
        meta = chunk.metadata or {}

        source = normalize_text(
            str(
                meta.get("source")
                or meta.get("source_file")
                or meta.get("filename")
                or ""
            )
        )

        subject = normalize_text(
            str(meta.get("subject", ""))
        )

        department = normalize_text(
            str(meta.get("department", ""))
        )

        doc_type = normalize_text(
            str(
                meta.get("doc_type")
                or meta.get("category")
                or ""
            )
        )

        blob = " ".join([
            source,
            subject,
            department,
            doc_type,
        ])

        score = 0.0

        # District match.
        districts = [
            "mayurbhanj",
            "cuttack",
            "ganjam",
            "koraput",
            "kalahandi",
            "jajpur",
            "puri",
            "khordha",
            "balasore",
            "bhadrak",
            "rourkela",
        ]

        for district in districts:

            if (
                district in q
                and district in blob
            ):
                score += 0.25

        # Concept matches.
        concepts = [
            ("dearness allowance", "dearness allowance"),
            ("fire safety", "fire safety"),
            ("noc", "noc"),
            ("cci", "cci"),
            ("child care", "child care"),
            ("jal jeevan mission", "jal jeevan mission"),
        ]

        for query_term, metadata_term in concepts:

            if (
                query_term in q
                and metadata_term in blob
            ):
                score += 0.20

        return min(
            1.0,
            score,
        )

    # ------------------------------------------------------------------------
    # Intent score
    # ------------------------------------------------------------------------

    def _intent_score(
        self,
        question: str,
        chunk: RetrievedChunk,
    ) -> float:

        intent = detect_intent(question)

        meta = chunk.metadata or {}

        doc_type = normalize_text(
            str(
                meta.get("doc_type")
                or meta.get("category")
                or ""
            )
        )

        source = normalize_text(
            str(
                meta.get("source")
                or meta.get("source_file")
                or meta.get("filename")
                or ""
            )
        )

        if intent == "complaint":

            if "complaint" in doc_type:
                return 1.0

            if "complaint" in source:
                return 0.95

            if "budget" in doc_type:
                return 0.0

            return 0.05

        if intent == "budget":

            if "budget" in doc_type:
                return 1.0

            if "budget" in source:
                return 0.95

            if "complaint" in doc_type:
                return 0.0

            return 0.05

        if intent == "circular":

            if "circular" in doc_type:
                return 1.0

            if (
                "da" in source
                or "dearness" in source
            ):
                return 0.95

            return 0.05

        if intent == "inspection":

            if "inspection" in doc_type:
                return 1.0

            if "inspection" in source:
                return 0.95

            return 0.05

        return 0.20

    # ------------------------------------------------------------------------
    # Semantic normalization
    # ------------------------------------------------------------------------

    @staticmethod
    def _semantic_score(
        chunk: RetrievedChunk,
        candidates: list[RetrievedChunk],
    ) -> float:

        distances = [
            c.distance
            for c in candidates
            if c.distance is not None
        ]

        if (
            chunk.distance is None
            or not distances
        ):
            return 0.0

        minimum = min(distances)
        maximum = max(distances)

        if maximum == minimum:
            return 1.0

        score = (
            (maximum - chunk.distance)
            / (maximum - minimum)
        )

        return max(
            0.0,
            min(1.0, score),
        )

    # ------------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------------

    def _rerank(
        self,
        question: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:

        for chunk in candidates:

            chunk.semantic_score = (
                self._semantic_score(
                    chunk,
                    candidates,
                )
            )

            chunk.lexical_score = (
                self._lexical_score(
                    question,
                    chunk,
                )
            )

            chunk.entity_score = (
                self._entity_score(
                    question,
                    chunk,
                )
            )

            chunk.metadata_score = (
                self._metadata_score(
                    question,
                    chunk,
                )
            )

            chunk.intent_score = (
                self._intent_score(
                    question,
                    chunk,
                )
            )

            # Entity matching is important for questions
            # about specific institutions, locations, etc.
            chunk.final_score = (
                0.25 * chunk.semantic_score
                + 0.20 * chunk.lexical_score
                + 0.20 * chunk.entity_score
                + 0.10 * chunk.metadata_score
                + 0.25 * chunk.intent_score
            )

            chunk.final_score = max(
                0.0,
                min(1.0, chunk.final_score),
            )

        candidates.sort(
            key=lambda chunk: chunk.final_score,
            reverse=True,
        )

        return candidates

    # ------------------------------------------------------------------------
    # Public query
    # ------------------------------------------------------------------------

    def query(
        self,
        question: str,
        k: int = 5,
        where: Optional[dict] = None,
    ) -> list[RetrievedChunk]:

        if not question or not question.strip():
            raise ValueError(
                "question must be a non-empty string"
            )

        if k < 1:
            raise ValueError(
                "k must be >= 1"
            )

        collection_count = self.count()

        if collection_count == 0:
            return []

        candidates = self._semantic_candidates(
            question=question,
            candidate_k=collection_count,
            where=where,
        )

        if not candidates:
            return []

        ranked = self._rerank(
            question,
            candidates,
        )

        if not ranked:
            return []

        # Keep the strongest result and remove clearly unrelated results.
        top_score = ranked[0].final_score

        relevant = [
            chunk
            for chunk in ranked
            if (
                chunk.final_score >= 0.35
                or chunk.final_score >= top_score * 0.60
            )
        ]

        if not relevant:
            relevant = ranked[:1]

        relevant.sort(
            key=lambda chunk: (
                chunk.final_score,
                chunk.lexical_score,
                chunk.intent_score,
            ),
            reverse=True,
        )

        return relevant[:k]

    

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
 
def format_results(
    chunks: list[RetrievedChunk],
) -> str:

    if not chunks:
        return "No results found."

    lines = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        preview = (
            chunk.text
            .strip()
            .replace("\n", " ")
        )

        if len(preview) > 400:
            preview = preview[:400] + "..."

        lines.append(
            f"{index}. {chunk.citation}\n"
            f"   score ~{chunk.final_score:.3f} | "
            f"semantic ~{chunk.semantic_score:.3f} | "
            f"lexical ~{chunk.lexical_score:.3f} | "
            f"entity ~{chunk.entity_score:.3f} | "
            f"metadata ~{chunk.metadata_score:.3f} | "
            f"intent ~{chunk.intent_score:.3f}\n"
            f"   {preview}"
        )

    return "\n\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:

    if len(sys.argv) < 2:

        print(
            'Usage: python retrieve.py '
            '"your question here" [k]'
        )

        sys.exit(1)

    question = sys.argv[1]

    try:
        k = (
            int(sys.argv[2])
            if len(sys.argv) > 2
            else 5
        )

    except ValueError:

        print("k must be an integer.")
        sys.exit(1)

    try:

        retriever = Retriever()

        print()
        print(
            f"Collection '{retriever.collection_name}' "
            f"has {retriever.count()} chunks."
        )

        print()
        print(
            f"Detected query intent: "
            f"{detect_intent(question)}"
        )

        chunks = retriever.query(
            question,
            k=k,
        )

        print()
        print(
            f"Top {len(chunks)} results for: "
            f"{question!r}"
        )

        print()
        print(format_results(chunks))

    except Exception as exc:

        print()
        print("ERROR:")
        print(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()