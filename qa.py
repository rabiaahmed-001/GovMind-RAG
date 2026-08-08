"""
qa.py — GovMind command-line interface.

The actual RAG logic lives in backend/rag.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag import RAGPipeline


def main() -> None:

    if len(sys.argv) < 2:

        print(
            'Usage: python qa.py "your question here" [k]'
        )

        sys.exit(1)

    question = sys.argv[1]

    try:

        k = (
            int(sys.argv[2])
            if len(sys.argv) >= 3
            else 5
        )

    except ValueError:

        print("ERROR: k must be an integer.")
        sys.exit(1)

    try:

        pipeline = RAGPipeline(k=k)

        result = pipeline.answer(
            question,
            k=k,
        )

    except Exception as exc:

        print()
        print("ERROR:")
        print(str(exc))
        sys.exit(1)

    print()
    print("=" * 70)
    print("GOVMIND ANSWER")
    print("=" * 70)
    print(result["answer"])

    print()
    print("-" * 70)
    print("SOURCES")
    print("-" * 70)

    sources = result.get("sources", [])

    if not sources:

        print("No sources found.")

    else:

        for i, source in enumerate(
            sources,
            start=1,
        ):

            score = source.get("score")

            score_text = (
                "N/A"
                if score is None
                else f"{score:.3f}"
            )

            print(
                f"{i}. {source['citation']} "
                f"(score ~{score_text})"
            )

    print("=" * 70)


if __name__ == "__main__":
    main()