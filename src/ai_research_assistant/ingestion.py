"""PDF loading and chunking."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class NoExtractableTextError(ValueError):
    """Raised when a PDF has no selectable text (e.g. a scanned image)."""


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Read a PDF into one Document per page, keeping the 1-based page number as metadata.

    Pages without extractable text are skipped.
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(page_content=text, metadata={"source": pdf_path.name, "page": page_number})
            )
    if not pages:
        raise NoExtractableTextError(
            f"No extractable text found in {pdf_path.name}. It may be a scanned image; "
            "try a PDF with selectable text."
        )
    return pages


def count_pages(pdf_path: str | Path) -> int:
    return len(PdfReader(pdf_path).pages)


def split_documents(
    pages: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """Split pages into overlapping chunks. Each chunk inherits its page's metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)
