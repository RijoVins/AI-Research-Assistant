"""RAG pipeline: PDF -> chunks -> embeddings (Chroma) -> retrieval -> LLM answer (Groq)."""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_PDF = PROJECT_ROOT / "data" / "pdfs" / "Notes.pdf"
DEFAULT_DB_DIR = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI research assistant. Answer the user's question using ONLY the "
        "context below, which comes from their PDF notes. If the answer is not in the "
        "context, say you could not find it in the notes. Be clear and concise, and cite "
        "the page numbers you used like (p. 12).\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])


def load_embeddings():
    """Load the sentence-transformers embedding model (slow; reuse the result)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


class AIResearchAssistant:
    def __init__(self, pdf_path=DEFAULT_PDF, persist_directory=None, k: int = 4, embeddings=None):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        if persist_directory is None:
            # One index per PDF (name + content hash), so switching PDFs never
            # answers from a stale index and editing a PDF triggers re-indexing.
            digest = hashlib.sha1(self.pdf_path.read_bytes()).hexdigest()[:10]
            persist_directory = DEFAULT_DB_DIR / f"{self.pdf_path.stem}_{digest}"
        self.persist_directory = str(persist_directory)
        self.k = k
        self.embeddings = embeddings or load_embeddings()
        self.vector_store = None
        self._llm = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def load_pdf_pages(self) -> list[Document]:
        """Read the PDF and return one Document per page (keeps page numbers)."""
        print(f"Reading PDF from {self.pdf_path}...")
        reader = PdfReader(self.pdf_path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={"source": self.pdf_path.name, "page": page_number},
                ))
        print(f"Extracted text from {len(pages)}/{len(reader.pages)} pages.")
        return pages

    def ingest_pdf(self, progress=None, batch_size: int = 32):
        """Chunk + embed the PDF into Chroma.

        progress: optional callback(fraction: float, message: str) for UIs.
        """
        report = progress or (lambda fraction, message: print(message))

        report(0.0, "Reading PDF...")
        pages = self.load_pdf_pages()
        if not pages:
            raise ValueError(
                "No extractable text found in this PDF. It may be a scanned image; "
                "try a PDF with selectable text."
            )

        report(0.05, "Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)

        # Start clean: drop any partial index left by an interrupted run
        self._marker().unlink(missing_ok=True)
        if os.path.isdir(self.persist_directory):
            Chroma(persist_directory=self.persist_directory,
                   embedding_function=self.embeddings).delete_collection()

        # Embed in batches so callers can show real progress
        self.vector_store = Chroma(
            persist_directory=self.persist_directory, embedding_function=self.embeddings
        )
        for i in range(0, len(docs), batch_size):
            self.vector_store.add_documents(docs[i:i + batch_size])
            done = min(i + batch_size, len(docs))
            report(0.05 + 0.95 * done / len(docs), f"Embedding chunks {done}/{len(docs)}...")
        self._marker().write_text("ok")
        report(1.0, f"Indexed {len(docs)} chunks from {len(pages)} pages.")

    def _marker(self) -> Path:
        """File written only after indexing finishes, so partial indexes are never used."""
        return Path(self.persist_directory) / ".index_complete"

    def load_vector_store(self) -> bool:
        """Load an existing, non-empty vector store. Returns False if none exists."""
        if not self._marker().exists():
            return False
        store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        if not store.get(limit=1)["ids"]:
            return False
        self.vector_store = store
        print(f"Loaded existing vector store from {self.persist_directory}")
        return True

    def setup(self, rebuild: bool = False, progress=None):
        """Load the vector store if present, otherwise (or if rebuild) ingest the PDF."""
        if rebuild or not self.load_vector_store():
            self.ingest_pdf(progress=progress)

    @property
    def is_ready(self) -> bool:
        return self.vector_store is not None

    def stats(self) -> dict:
        """Page and chunk counts for display."""
        return {
            "pages": len(PdfReader(self.pdf_path).pages),
            "chunks": self.vector_store._collection.count() if self.vector_store else 0,
        }

    # ------------------------------------------------------------------
    # Retrieval + generation
    # ------------------------------------------------------------------
    def retrieve(self, question: str) -> dict:
        if not self.is_ready:
            raise ValueError("Vector store not initialized. Call setup() first.")
        docs = self.vector_store.similarity_search(question, k=self.k)
        context = "\n\n".join(
            f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
        )
        pages = sorted({d.metadata.get("page") for d in docs} - {None})
        return {"context": context, "source_docs": docs, "pages": pages}

    @property
    def llm(self):
        if self._llm is None:
            if not os.getenv("GROQ_API_KEY"):
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to a .env file in the project root "
                    "(see .env.example) or export it in your shell."
                )
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(model=DEFAULT_LLM_MODEL, temperature=0)
        return self._llm

    def _chain(self):
        return PROMPT | self.llm | StrOutputParser()

    def ask(self, question: str) -> dict:
        """Retrieve relevant chunks and generate an answer with the LLM."""
        retrieved = self.retrieve(question)
        answer = self._chain().invoke({"context": retrieved["context"], "question": question})
        return {"answer": answer, "pages": retrieved["pages"], "source_docs": retrieved["source_docs"]}

    def ask_stream(self, question: str) -> dict:
        """Like ask(), but 'answer' is a generator yielding text as the LLM writes it."""
        retrieved = self.retrieve(question)
        stream = self._chain().stream({"context": retrieved["context"], "question": question})
        return {"answer": stream, "pages": retrieved["pages"], "source_docs": retrieved["source_docs"]}


if __name__ == "__main__":
    assistant = AIResearchAssistant()
    assistant.setup()

    res = assistant.retrieve("What is Machine Learning?")
    print("\n--- Sample Retrieval Test ---")
    for doc in res["source_docs"]:
        print(f"[Page {doc.metadata.get('page')}] {doc.page_content[:150]!r}")
