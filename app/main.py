"""Interactive command-line AI Research Assistant over your PDF notes.

Usage (from the project root):
    python app/main.py                 # ask questions about data/pdfs/Notes.pdf
    python app/main.py path/to/any.pdf # ask questions about any PDF (indexed on first use)
    python app/main.py --rebuild       # force re-indexing of the PDF
"""

import argparse

from rag_pipeline import DEFAULT_PDF, AIResearchAssistant


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your PDF notes.")
    parser.add_argument("pdf", nargs="?", default=str(DEFAULT_PDF), help="Path to the PDF file")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector store")
    parser.add_argument("-k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    try:
        assistant = AIResearchAssistant(pdf_path=args.pdf, k=args.k)
    except FileNotFoundError as e:
        parser.error(str(e))
    assistant.setup(rebuild=args.rebuild)

    print(f"\nAI Research Assistant ready for '{assistant.pdf_path.name}'! "
          "Ask a question (type 'exit' to quit).\n")
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
            result = assistant.ask(question)
        except Exception as e:  # keep the chat loop alive on API/network errors
            print(f"Error: {e}\n")
            continue

        print(f"\nAssistant: {result['answer']}")
        if result["pages"]:
            print(f"Sources: pages {', '.join(map(str, result['pages']))}")
        print()


if __name__ == "__main__":
    main()
