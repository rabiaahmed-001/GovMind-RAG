# GovMind

GovMind is a retrieval augmented generation (RAG) assistant for querying
Odisha government documents including policy PDFs, circulars, inspection reports
and structured citizen complaint / budget CSV data using natural
language, with every answer grounded in cited source documents.

## Why RAG, not just an LLM

Government data changes is siloed across formats (PDF policy docs, CSV
budget tables, inspection reports).and answers need to be trace back
to a source. GovMind retrieves the actual
relevant excerpts from a local knowledge base before generating an
answer and every claim in the answer is tied to a numbered source which means you can verify it against the original document rather than trusting the model's word for it.

## How it works

```text
                GOVMIND
                   │
          ┌────────┴────────┐
          │                 │
      Documents          CSV Data
     (policy, circulars,   (complaints,
      inspection reports)   budget)
          │                 │
          └────────┬────────┘
                    ↓
               Loaders
                    ↓
                Chunking
                    ↓
              Embeddings
                    ↓
              ChromaDB
                    ↓
               Retrieval
                    ↓
                 Gemini
                    ↓
          Grounded Response
             + Citations
                    ↓
             Streamlit UI
```

## Tech stack

- Python
- Streamlit
- ChromaDB (vector store, persisted locally)
- Google Gemini API (`gemini-2.5-flash`) via `google-genai`
- Pandas (CSV handling)

## Example

**Question:**
> How much was allocated and utilized for Jal Jeevan Mission in Mayurbhanj?

**GovMind:**
> For the Jal Jeevan Mission (Rural Piped Water) in Mayurbhanj during
> FY2023-24: Allocated ₹5.6 Cr, Utilized ₹5.04 Cr (90.0% utilization),
> Unutilized ₹0.56 Cr. [1]

**Sources:**
> [1] budget_allocation_utilization_odisha_2023_24.csv

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with:

```text
GEMINI_API_KEY=your_key_here
```

Run ingestion once to build the local vector store (only needed the
first time, or when source documents change):

```powershell
python backend/ingest.py
```

Run the app:

```powershell
streamlit run app.py
```

Or query from the command line:

```powershell
python backend/rag.py "your question here"
```

## Project structure

```text
GovMind RAG/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── backend/
│   ├── config.py
│   ├── ingest.py
│   ├── loaders.py
│   ├── chunker.py
│   ├── embeddings.py
│   └── rag.py
├── policy_pdfs/
├── circulars/
├── inspection_reports/
├── complaints_csv/
├── budget_csv/
└── chroma_store/          (generated locally, not committed)
```

## Notes

- `chroma_store/` and `.env` are intentionally excluded from version
  control (see `.gitignore`) clone the repo and run ingestion locally
  to regenerate the vector store.
- Answers are generated strictly from retrieved context; the system is
  designed to say so when the knowledge base doesn't contain enough
  information to answer rather than fall back on the model's general
  knowledge.
