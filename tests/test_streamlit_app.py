"""End-to-end UI tests using Streamlit's headless AppTest runner."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ai_research_assistant.assistant as assistant_module
import ai_research_assistant.config as config_module

from .conftest import KeywordEmbeddings, fake_llm

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture(autouse=True)
def offline_app(settings, monkeypatch):
    """Point the app at temp dirs and fake models so it runs without downloads or keys."""
    monkeypatch.setattr(config_module, "settings", settings)
    monkeypatch.setattr(assistant_module, "load_embeddings", lambda *a, **k: KeywordEmbeddings())
    monkeypatch.setattr(assistant_module, "default_llm", lambda cfg: fake_llm())


def button(at, text):
    return next(b for b in at.button if text in b.label)


def open_sample(at):
    at.run()
    button(at, "Try the sample notes").click().run()
    assert not at.exception, at.exception
    return at


def test_landing_page_renders():
    at = AppTest.from_file(APP).run()

    assert not at.exception
    assert at.title[0].value == "📚 AI Research Assistant"
    assert len(at.get("file_uploader")) == 1


def test_chat_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    at = open_sample(AppTest.from_file(APP, default_timeout=60))

    assert at.chat_input[0].disabled
    assert at.text_input[0].label == "Groq API key"


def test_suggested_question_produces_cited_answer(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    at = open_sample(AppTest.from_file(APP, default_timeout=60))

    button(at, "Summarize").click().run()

    assert not at.exception, at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "labeled data" in at.chat_message[1].markdown[0].value
    assert "Sources: pages" in at.chat_message[1].caption[0].value
    assert not any("Summarize" in b.label for b in at.button)  # suggestions hidden


def test_change_document_returns_to_landing_with_history(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    at = open_sample(AppTest.from_file(APP, default_timeout=60))
    at.chat_input[0].set_value("What is photosynthesis?").run()

    button(at, "Change document").click().run()

    assert any("1 question(s)" in b.label for b in at.button)
