"""AI Research Assistant: chat with your PDFs using retrieval-augmented generation (RAG)."""

from .assistant import Answer, MissingAPIKeyError, ResearchAssistant, Retrieval
from .ingestion import NoExtractableTextError

__all__ = [
    "Answer",
    "MissingAPIKeyError",
    "NoExtractableTextError",
    "ResearchAssistant",
    "Retrieval",
]
__version__ = "1.0.0"
