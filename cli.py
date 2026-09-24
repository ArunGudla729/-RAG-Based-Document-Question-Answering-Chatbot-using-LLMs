"""Command-line interface for the RAG pipeline (no Streamlit required).

Examples:
    # Ask a single question against one or more PDFs
    python cli.py --pdf report.pdf --pdf invoice.pdf -q "What is the total due?"

    # Interactive REPL
    python cli.py --pdf paper.pdf

    # Pick provider / model / strategy
    python cli.py --pdf doc.pdf --provider openai --model gpt-4o-mini \
        --strategy cite_first -q "Summarise section 2"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.llm_client import build_client
from src.prompts import STRATEGIES, DEFAULT_STRATEGY
from src.rag_pipeline import RAGPipeline


def build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    llm = build_client(args.provider, args.model)
    pipe = RAGPipeline(llm=llm, top_k=args.top_k, min_score=args.min_score)
    for pdf_path in args.pdf:
        path = Path(pdf_path)
        if not path.exists():
            sys.exit(f"File not found: {path}")
        added = pipe.add_pdf(path.read_bytes(), path.name)
        print(f"Indexed {path.name}: {added} chunks", file=sys.stderr)
    return pipe


def print_answer(answer) -> None:
    print("\n" + answer.text)
    if answer.retrieved:
        print("\n--- Sources ---")
        for i, r in enumerate(answer.retrieved, start=1):
            print(f"[Source {i}] {r.chunk.citation()} (similarity {r.score:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Document Q&A (CLI)")
    parser.add_argument("--pdf", action="append", required=True,
                        help="Path to a PDF (repeatable)")
    parser.add_argument("-q", "--question", help="Single question; omit for REPL")
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai"])
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY,
                        choices=list(STRATEGIES))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.25)
    args = parser.parse_args()

    pipe = build_pipeline(args)

    if args.question:
        print_answer(pipe.answer(args.question, strategy=args.strategy))
        return

    print("Interactive mode — type a question (Ctrl-D or 'exit' to quit).",
          file=sys.stderr)
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            print_answer(pipe.answer(q, strategy=args.strategy))


if __name__ == "__main__":
    main()
