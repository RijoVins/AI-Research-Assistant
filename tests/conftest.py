"""Shared fixtures. Tests run offline: no model downloads and no API keys needed."""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
import zlib
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from ai_research_assistant.config import settings as real_settings


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal valid PDF with one page per string (each line becomes a text line)."""

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    n = len(pages)
    objs = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{3 + 2 * i} 0 R" for i in range(n)), n
        ),
    }
    font_id = 3 + 2 * n
    for i, text in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        ops = " ".join(f"({esc(line)}) Tj T*" for line in text.splitlines())
        stream = f"BT /F1 12 Tf 14 TL 72 720 Td {ops} ET"
        objs[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )
        objs[content_id] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    objs[font_id] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for obj_id in sorted(objs):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n{objs[obj_id]}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for obj_id in sorted(objs):
        out += f"{offsets[obj_id]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


class KeywordEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: texts sharing words are close in vector space.

    Good enough to test that retrieval ranks the relevant chunk first, without a real model.
    """

    dim = 512

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z]{3,}", text.lower()):
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


SAMPLE_PAGES = [
    "Photosynthesis converts sunlight into chemical energy.\n"
    "Plants use chlorophyll in their leaves to capture light.",
    "Supervised learning trains models on labeled examples.\n"
    "Classification and regression are supervised learning tasks.",
    "The French Revolution began in 1789.\nIt transformed politics across Europe.",
]


@pytest.fixture
def settings(tmp_path):
    """Settings pointing at temporary directories, with small chunks for fast tests."""
    return dataclasses.replace(
        real_settings,
        db_dir=tmp_path / "db",
        upload_dir=tmp_path / "uploads",
        sample_pdf=make_pdf(tmp_path / "sample.pdf", SAMPLE_PAGES),
        chunk_size=200,
        chunk_overlap=20,
        top_k=2,
    )


@pytest.fixture
def sample_pdf(settings) -> Path:
    return settings.sample_pdf


@pytest.fixture
def embeddings() -> Embeddings:
    return KeywordEmbeddings()


def fake_llm(reply: str = "Supervised learning uses labeled data (p. 2).") -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.repeat(reply))
