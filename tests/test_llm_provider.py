import dataclasses

import pytest

from ai_research_assistant.assistant import MissingAPIKeyError, default_llm


def test_groq_is_used_by_default(settings, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

    llm = default_llm(settings)

    assert type(llm).__name__ == "ChatGroq"
    assert settings.provider_label == "Groq"


def test_openai_compatible_gateway_when_base_url_set(settings, monkeypatch):
    gateway = dataclasses.replace(
        settings,
        llm_base_url="http://localhost:9999/v1",
        llm_model="my-model",
        llm_provider_name="OmniRoute",
    )
    monkeypatch.setenv("LLM_API_KEY", "omni-test-key")

    llm = default_llm(gateway)

    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.openai_api_base == "http://localhost:9999/v1"
    assert llm.model_name == "my-model"
    assert gateway.provider_label == "OmniRoute"
    assert gateway.api_key_env == "LLM_API_KEY"


def test_gateway_requires_its_own_key(settings, monkeypatch):
    gateway = dataclasses.replace(settings, llm_base_url="http://localhost:9999/v1")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")  # a Groq key must not count
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError, match="LLM_API_KEY"):
        default_llm(gateway)
