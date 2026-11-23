"""Application settings, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    """All tunable parameters in one place. Override any of them via environment variables."""

    db_dir: Path
    upload_dir: Path
    sample_pdf: Path
    embedding_model: str
    llm_model: str
    llm_base_url: str | None  # set -> use an OpenAI-compatible API instead of Groq
    llm_provider_name: str
    temperature: float
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            db_dir=_env_path("VECTOR_DB_DIR", PROJECT_ROOT / "chroma_db"),
            upload_dir=_env_path("UPLOAD_DIR", PROJECT_ROOT / "data" / "pdfs" / "uploads"),
            sample_pdf=_env_path("SAMPLE_PDF", PROJECT_ROOT / "data" / "pdfs" / "Notes.pdf"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            llm_model=(
                os.getenv("LLM_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
            ),
            llm_base_url=os.getenv("LLM_BASE_URL") or None,
            llm_provider_name=os.getenv("LLM_PROVIDER_NAME", "OpenAI-compatible API"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K", "4")),
        )

    @property
    def uses_openai_compatible(self) -> bool:
        """True when LLM_BASE_URL points at an OpenAI-compatible gateway (OmniRoute, etc.)."""
        return bool(self.llm_base_url)

    @property
    def provider_label(self) -> str:
        return self.llm_provider_name if self.uses_openai_compatible else "Groq"

    @property
    def api_key_env(self) -> str:
        """Name of the environment variable holding the LLM API key."""
        return "LLM_API_KEY" if self.uses_openai_compatible else "GROQ_API_KEY"

    def has_api_key(self) -> bool:
        # Read at call time so a key entered in the UI is picked up immediately.
        return bool(os.getenv(self.api_key_env))


settings = Settings.from_env()
