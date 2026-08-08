from __future__ import annotations

import json
import sys
from pathlib import Path

# Make backend modules importable
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag import RAGPipeline


EVAL_FILE = PROJECT_ROOT / "eval_questions.json"


def main():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    rag = RAGPipeline(k=5)

    print(f"Loaded {len(questions)} evaluation questions.")
    print("=" * 80)

    for item in questions:
        question_id = item["id"]
        question = item["question"]

        print(f"\n{question_id}: {question}")

        try:
            result = rag.answer(question)

            print("\nANSWER:")
            print(result.answer)

            print("\nSOURCES:")
            for chunk in result.chunks:
                print(f"  - {chunk.citation}")

        except Exception as e:
            print(f"\nERROR: {e}")

        print("-" * 80)


if __name__ == "__main__":
    main()