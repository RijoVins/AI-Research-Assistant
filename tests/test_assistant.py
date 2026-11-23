import pytest
from langchain_core.runnables import RunnableLambda

from ai_research_assistant import MissingAPIKeyError, ResearchAssistant

from .conftest import fake_llm, make_pdf


@pytest.fixture
def assistant(sample_pdf, settings, embeddings):
    a = ResearchAssistant(sample_pdf, settings=settings, embeddings=embeddings, llm=fake_llm())
    a.ensure_index()
    return a


def test_missing_pdf_raises(tmp_path, settings, embeddings):
    with pytest.raises(FileNotFoundError):
        ResearchAssistant(tmp_path / "nope.pdf", settings=settings, embeddings=embeddings)


def test_build_index_reports_progress(sample_pdf, settings, embeddings):
    a = ResearchAssistant(sample_pdf, settings=settings, embeddings=embeddings)
    updates = []

    a.build_index(progress=lambda fraction, msg: updates.append(fraction))

    assert a.is_ready
    assert a.chunk_count() > 0
    assert updates[0] == 0.0 and updates[-1] == 1.0
    assert updates == sorted(updates)


def test_index_is_reused_across_instances(assistant, sample_pdf, settings, embeddings):
    again = ResearchAssistant(sample_pdf, settings=settings, embeddings=embeddings)

    assert again.load_index()
    assert again.chunk_count() == assistant.chunk_count()


def test_interrupted_index_is_not_reused(sample_pdf, settings, embeddings):
    a = ResearchAssistant(sample_pdf, settings=settings, embeddings=embeddings)
    a.build_index()
    (a.index_dir / ".index_complete").unlink()  # simulate a crash before completion

    assert not ResearchAssistant(sample_pdf, settings=settings, embeddings=embeddings).load_index()


def test_rebuild_does_not_duplicate_chunks(assistant):
    before = assistant.chunk_count()

    assistant.ensure_index(rebuild=True)

    assert assistant.chunk_count() == before


def test_each_pdf_gets_its_own_index(tmp_path, sample_pdf, settings, embeddings):
    other = make_pdf(tmp_path / "other.pdf", ["Completely different content"])

    a = ResearchAssistant(sample_pdf, settings=settings, embeddings=embeddings)
    b = ResearchAssistant(other, settings=settings, embeddings=embeddings)

    assert a.index_dir != b.index_dir


def test_retrieve_ranks_relevant_page_first(assistant):
    retrieval = assistant.retrieve("What is supervised learning with labeled examples?")

    assert retrieval.documents[0].metadata["page"] == 2
    assert "[Page 2]" in retrieval.context


def test_retrieve_respects_k(assistant):
    assistant.k = 1

    assert len(assistant.retrieve("photosynthesis").documents) == 1


def test_ask_returns_answer_with_sources(assistant):
    answer = assistant.ask("What is supervised learning?")

    assert answer.text == "Supervised learning uses labeled data (p. 2)."
    assert 2 in answer.pages
    assert answer.sources


def test_prompt_is_grounded_in_retrieved_context(assistant):
    # An "LLM" that echoes its system prompt lets us inspect exactly what the model sees.
    assistant._llm = RunnableLambda(lambda prompt: prompt.to_messages()[0].content)

    answer = assistant.ask("Tell me about photosynthesis and chlorophyll")

    assert "ONLY the context" in answer.text
    assert "chlorophyll" in answer.text
    assert "[Page 1]" in answer.text


def test_stream_yields_tokens(assistant):
    tokens, retrieval = assistant.stream("What is supervised learning?")

    assert "".join(tokens) == "Supervised learning uses labeled data (p. 2)."
    assert retrieval.pages


def test_missing_api_key_gives_clear_error(sample_pdf, settings, embeddings, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    a = ResearchAssistant(sample_pdf, settings=settings, embeddings=embeddings)
    a.ensure_index()

    with pytest.raises(MissingAPIKeyError):
        a.ask("anything")
