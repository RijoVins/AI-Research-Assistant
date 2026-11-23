"""Command-line interface: chat with a PDF in the terminal.

Examples:
    research-assistant                    # uses the sample PDF
    research-assistant paper.pdf          # any PDF (indexed on first use)
    research-assistant paper.pdf --rebuild
"""

from __future__ import annotations

import argparse
import logging

from .assistant import ResearchAssistant
from .config import settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-assistant", description="Ask questions about a PDF using RAG."
    )
    parser.add_argument("pdf", nargs="?", default=str(settings.sample_pdf), help="PDF file")
    parser.add_argument("--rebuild", action="store_true", help="re-index the PDF from scratch")
    parser.add_argument(
        "-k", type=int, default=settings.top_k, help="number of chunks to retrieve per question"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    for noisy in ("httpx", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        assistant = ResearchAssistant(args.pdf, k=args.k)
    except FileNotFoundError as e:
        parser.error(str(e))
    assistant.ensure_index(rebuild=args.rebuild)

    print(f"\nReady to answer questions about '{assistant.pdf_path.name}' (type 'exit' to quit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        try:
            answer = assistant.ask(question)
        except Exception as e:  # keep the chat loop alive on API/network errors
            print(f"Error: {e}\n")
            continue

        print(f"\nAssistant: {answer.text}")
        if answer.pages:
            print(f"Sources: pages {', '.join(map(str, answer.pages))}")
        print()


if __name__ == "__main__":
    main()
