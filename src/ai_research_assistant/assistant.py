"""Core RAG engine: index a PDF into Chroma, retrieve relevant chunks, answer with an LLM."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from .config import Settings
from .config import settings as default_settings
from .ingestion import count_pages, load_pdf, split_documents
from .prompts import QA_PROMPT

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
_INDEX_COMPLETE_MARKER = ".index_complete"


class MissingAPIKeyError(RuntimeError):
    """Raised when no GROQ_API_KEY is configured."""


def load_embeddings(model_name: str | None = None) -> Embeddings:
    """Load the sentence-transformers embedding model. Slow: create once and reuse."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name or default_settings.embedding_model)


def default_llm(cfg: Settings) -> BaseChatModel:
    """Create the Groq chat model, failing early with a clear message if no key is set."""
    if not os.getenv("GROQ_API_KEY"):
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Add it to a .env file in the project root "
            "(see .env.example) or export it in your shell."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(model=cfg.llm_model, temperature=cfg.temperature)


@dataclass(frozen=True)
class Retrieval:
    """The chunks retrieved for a question."""

    documents: list[Document]

    @property
    def pages(self) -> list[int]:
        return sorted({d.metadata["page"] for d in self.documents if "page" in d.metadata})

    @property
    def context(self) -> str:
        return "\n\n".join(
            f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in self.documents
        )


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Document] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        return Retrieval(self.sources).pages


class ResearchAssistant:
    """Question answering over a single PDF using retrieval-augmented generation.

    Each PDF gets its own persistent Chroma index, keyed by file name and content hash, so
    switching or editing PDFs never serves stale results.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        *,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | Runnable | None = None,
        k: int | None = None,
    ):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        self.settings = settings or default_settings
        self.k = k or self.settings.top_k
        self.embeddings = embeddings or load_embeddings(self.settings.embedding_model)
        self._llm = llm
        self.vector_store: Chroma | None = None

        digest = hashlib.sha1(self.pdf_path.read_bytes()).hexdigest()[:10]
        self.index_dir = self.settings.db_dir / f"{self.pdf_path.stem}_{digest}"

    # ------------------------------------------------------------------ indexing
    @property
    def is_ready(self) -> bool:
        return self.vector_store is not None

    @property
    def _marker(self) -> Path:
        # Written only after indexing finishes, so an interrupted run is never reused.
        return self.index_dir / _INDEX_COMPLETE_MARKER

    def _open_store(self) -> Chroma:
        return Chroma(persist_directory=str(self.index_dir), embedding_function=self.embeddings)

    def load_index(self) -> bool:
        """Load a previously built index. Returns False if none exists."""
        if not self._marker.exists():
            return False
        self.vector_store = self._open_store()
        logger.info("Loaded existing index from %s", self.index_dir)
        return True

    def build_index(self, progress: ProgressCallback | None = None, batch_size: int = 32) -> None:
        """Load, chunk and embed the PDF into a fresh Chroma index."""
        report = progress or (lambda fraction, message: logger.info(message))

        report(0.0, "Reading PDF...")
        pages = load_pdf(self.pdf_path)

        report(0.05, "Splitting text into chunks...")
        chunks = split_documents(pages, self.settings.chunk_size, self.settings.chunk_overlap)

        # Start clean: drop any partial index left by an interrupted run
        self._marker.unlink(missing_ok=True)
        if self.index_dir.exists():
            self._open_store().delete_collection()

        store = self._open_store()
        for start in range(0, len(chunks), batch_size):
            store.add_documents(chunks[start : start + batch_size])
            done = min(start + batch_size, len(chunks))
            report(0.05 + 0.95 * done / len(chunks), f"Embedding chunks {done}/{len(chunks)}...")

        self._marker.write_text("ok")
        self.vector_store = store
        report(1.0, f"Indexed {len(chunks)} chunks from {len(pages)} pages.")

    def ensure_index(self, rebuild: bool = False, progress: ProgressCallback | None = None) -> None:
        """Load the index if it exists, otherwise (or when rebuild=True) build it."""
        if rebuild or not self.load_index():
            self.build_index(progress=progress)

    def page_count(self) -> int:
        return count_pages(self.pdf_path)

    def chunk_count(self) -> int:
        return self.vector_store._collection.count() if self.vector_store else 0

    # ------------------------------------------------------------------ querying
    @property
    def llm(self) -> BaseChatModel | Runnable:
        if self._llm is None:
            self._llm = default_llm(self.settings)
        return self._llm

    def reset_llm(self) -> None:
        """Forget the cached LLM client (e.g. after the API key changes)."""
        self._llm = None

    def retrieve(self, question: str) -> Retrieval:
        """Semantic search: return the k chunks closest in meaning to the question."""
        if not self.is_ready:
            raise RuntimeError("Index not loaded. Call ensure_index() first.")
        return Retrieval(self.vector_store.similarity_search(question, k=self.k))

    def _chain(self) -> Runnable:
        return QA_PROMPT | self.llm | StrOutputParser()

    def ask(self, question: str) -> Answer:
        """Retrieve relevant chunks and generate a grounded answer."""
        retrieval = self.retrieve(question)
        text = self._chain().invoke({"context": retrieval.context, "question": question})
        return Answer(text=text, sources=retrieval.documents)

    def stream(self, question: str) -> tuple[Iterator[str], Retrieval]:
        """Like ask(), but returns a token stream (for responsive UIs) plus the sources."""
        retrieval = self.retrieve(question)
        llm_chain = self._chain()  # resolves the LLM now, so a missing key fails early
        tokens = llm_chain.stream({"context": retrieval.context, "question": question})
        return tokens, retrieval
