import pytest

from ai_research_assistant.ingestion import (
    NoExtractableTextError,
    count_pages,
    load_pdf,
    split_documents,
)

from .conftest import make_pdf


def test_load_pdf_keeps_page_numbers(sample_pdf):
    pages = load_pdf(sample_pdf)

    assert [p.metadata["page"] for p in pages] == [1, 2, 3]
    assert "Photosynthesis" in pages[0].page_content
    assert all(p.metadata["source"] == sample_pdf.name for p in pages)


def test_load_pdf_skips_blank_pages(tmp_path):
    pdf = make_pdf(tmp_path / "gaps.pdf", ["First page text", "", "Third page text"])

    pages = load_pdf(pdf)

    assert [p.metadata["page"] for p in pages] == [1, 3]
    assert count_pages(pdf) == 3


def test_load_pdf_without_text_raises(tmp_path):
    pdf = make_pdf(tmp_path / "scanned.pdf", ["", ""])

    with pytest.raises(NoExtractableTextError):
        load_pdf(pdf)


def test_split_documents_overlaps_and_keeps_metadata(sample_pdf):
    chunks = split_documents(load_pdf(sample_pdf), chunk_size=60, chunk_overlap=15)

    assert len(chunks) > 3
    assert all(len(c.page_content) <= 60 for c in chunks)
    assert {c.metadata["page"] for c in chunks} == {1, 2, 3}
