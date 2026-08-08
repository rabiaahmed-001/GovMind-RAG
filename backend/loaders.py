"""
Loaders turn raw source files into a list of Document dicts:
    {"text": str, "metadata": {...}}

Metadata is what makes cross-document reasoning and filtered retrieval
possible later (e.g. "only search budget rows where district=Mayurbhanj"),
so it's populated generously rather than left as an afterthought.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber


@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# PDF loader (circulars, policy notifications, inspection reports)
# ---------------------------------------------------------------------

_LETTER_NO_RE = re.compile(r"Letter No\.\s*([^\|]+)\|\s*Dated:\s*([^\n]+)")
_SUBJECT_RE = re.compile(r"Subject:\s*(.+?)(?:\n[A-Z][a-z]|\Z)", re.DOTALL)
_DEPT_LINE_RE = re.compile(r"^GOVERNMENT OF ODISHA\n([^\n]+)", re.MULTILINE)


def load_pdf(path: Path, doc_type: str) -> Document:
    """Extract text (and any tables) from a single government PDF.

    doc_type is one of "policy" | "circular" | "inspection" and is stored
    in metadata so retrieval can be filtered/boosted by document class.
    """
    full_text_parts: list[str] = []
    tables_as_text: list[str] = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_text_parts.append(page_text)

            for table in page.extract_tables():
                # Render each table as a small markdown-ish block so it
                # survives chunking as readable text rather than a raw
                # list-of-lists.
                rows = [
                    " | ".join(cell.strip() if cell else "" for cell in row)
                    for row in table
                    if row
                ]
                if rows:
                    tables_as_text.append("\n".join(rows))

    full_text = "\n".join(full_text_parts).strip()
    if tables_as_text:
        full_text += "\n\n" + "\n\n".join(tables_as_text)

    dept_match = _DEPT_LINE_RE.search(full_text)
    letter_match = _LETTER_NO_RE.search(full_text)
    subject_match = _SUBJECT_RE.search(full_text)

    metadata = {
        "source_file": path.name,
        "doc_type": doc_type,
        "department": dept_match.group(1).strip() if dept_match else None,
        "letter_no": letter_match.group(1).strip() if letter_match else None,
        "date": letter_match.group(2).strip() if letter_match else None,
        "subject": (
            subject_match.group(1).strip().replace("\n", " ")
            if subject_match
            else None
        ),
    }
    return Document(text=full_text, metadata=metadata)


def load_all_pdfs(data_dir: Path, doc_map: dict[str, list[str]]) -> list[Document]:
    docs: list[Document] = []
    for doc_type, filenames in doc_map.items():
        for filename in filenames:
            docs.append(load_pdf(data_dir / filename, doc_type=doc_type))
    return docs


# ---------------------------------------------------------------------
# CSV loaders — row-to-text templating (per README guidance)
# ---------------------------------------------------------------------

def load_complaints_csv(path: Path) -> list[Document]:
    """One citizen complaint row -> one retrievable document."""
    docs: list[Document] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = row["status"]
            resolution = (
                f", resolved in {row['resolution_days']} days"
                if row.get("resolution_days")
                else ""
            )
            text = (
                f"Citizen complaint {row['complaint_id']} filed on {row['date_filed']} "
                f"in {row['block_or_ward']}, {row['district']} district, regarding the "
                f"{row['department']} department: {row['complaint_text']} "
                f"Priority: {row['priority']}. Status: {status}{resolution}. "
                f"Filed via {row['citizen_channel']}."
            )
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source_file": path.name,
                        "doc_type": "complaint",
                        "complaint_id": row["complaint_id"],
                        "district": row["district"],
                        "department": row["department"],
                        "priority": row["priority"],
                        "status": status,
                        "date_filed": row["date_filed"],
                    },
                )
            )
    return docs


def load_budget_csv(path: Path) -> list[Document]:
    """One budget line -> one natural-language sentence, per README template."""
    docs: list[Document] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (
                f"In {row['district']}, the {row['department']} department allocated "
                f"₹{row['budget_allocated_cr']} Cr under {row['scheme']} in FY"
                f"{row['financial_year']} and utilized ₹{row['budget_utilized_cr']} Cr "
                f"({row['utilization_pct']}% utilization, ₹{row['unutilized_cr']} Cr "
                f"unutilized). Status flag: {row['flag']}."
            )
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source_file": path.name,
                        "doc_type": "budget",
                        "budget_id": row["budget_id"],
                        "district": row["district"],
                        "department": row["department"],
                        "scheme": row["scheme"],
                        "utilization_pct": float(row["utilization_pct"]),
                        "flag": row["flag"],
                    },
                )
            )
    return docs
