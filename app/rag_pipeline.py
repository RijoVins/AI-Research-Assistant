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


class AIResearchAssistant:
    def __init__(self, pdf_path=DEFAULT_PDF, persist_directory=None, k: int = 4):
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
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        self.retriever = None
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

    def ingest_pdf(self):
        pages = self.load_pdf_pages()
        if not pages:
            raise ValueError(f"No extractable text found in {self.pdf_path}")

        print("Splitting text into smart chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)
        print(f"Created {len(docs)} chunks.")

        print("Creating vector store (Chroma)...")
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self._make_retriever()
        print("Ingestion completed successfully!")

    def load_vector_store(self) -> bool:
        """Load an existing, non-empty vector store. Returns False if none exists."""
        if not os.path.isdir(self.persist_directory):
            return False
        print(f"Loading existing vector store from {self.persist_directory}...")
        store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        if not store.get(limit=1)["ids"]:
            print("Vector store is empty.")
            return False
        self.vector_store = store
        self._make_retriever()
        print("Vector store loaded successfully!")
        return True

    def setup(self, rebuild: bool = False):
        """Load the vector store if present, otherwise (or if rebuild) ingest the PDF."""
        if rebuild and os.path.isdir(self.persist_directory):
            Chroma(persist_directory=self.persist_directory,
                   embedding_function=self.embeddings).delete_collection()
        if rebuild or not self.load_vector_store():
            self.ingest_pdf()

    def _make_retriever(self):
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )

    # ------------------------------------------------------------------
    # Retrieval + generation
    # ------------------------------------------------------------------
    def retrieve(self, question: str) -> dict:
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call setup() first.")
        docs = self.retriever.invoke(question)
        context = "\n\n".join(
            f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
        )
        return {"context": context, "source_docs": docs}

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

    def ask(self, question: str) -> dict:
        """Retrieve relevant chunks and generate an answer with the LLM."""
        retrieved = self.retrieve(question)
        chain = PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({"context": retrieved["context"], "question": question})
        pages = sorted({d.metadata.get("page") for d in retrieved["source_docs"]} - {None})
        return {"answer": answer, "pages": pages, "source_docs": retrieved["source_docs"]}


if __name__ == "__main__":
    assistant = AIResearchAssistant()
    assistant.setup()

    res = assistant.retrieve("What is Machine Learning?")
    print("\n--- Sample Retrieval Test ---")
    for doc in res["source_docs"]:
        print(f"[Page {doc.metadata.get('page')}] {doc.page_content[:150]!r}")
