import pytest
from pydantic import SecretStr

from review_agent.agent.llm import NVIDIA_URL, OPENROUTER_URL, build_chat_model
from review_agent.agent.offline_model import OfflineModel
from review_agent.config import Settings


def make(provider, model, **kw):
    return build_chat_model(Settings(env="test", llm_provider=provider, llm_model=model, _env_file=None,
                                     llm_api_key=SecretStr("test-key"), **kw))


def test_default_is_nemotron_on_nvidia_with_thinking_off(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    s = Settings(env="test", _env_file=None)
    assert (s.llm_provider, s.llm_model) == ("nvidia", "nvidia/nemotron-3-super-120b-a12b")
    m = build_chat_model(s)
    assert str(m.openai_api_base) == NVIDIA_URL and m.max_tokens == 1024
    assert m.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert make("nvidia", "nvidia/nemotron-3-super-120b-a12b", llm_reasoning=True).extra_body == {
        "chat_template_kwargs": {"enable_thinking": True}}


def test_nvidia_needs_a_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        build_chat_model(Settings(env="test", _env_file=None))


def test_openai_and_openrouter_clients_get_timeouts_and_retries():
    m = make("openai", "gpt-4.1-mini", llm_timeout_s=12, llm_max_retries=3)
    assert type(m).__name__ == "ChatOpenAI" and m.max_retries == 3 and m.request_timeout == 12
    r = make("openrouter", "google/gemini-2.5-flash")
    assert str(r.openai_api_base) == OPENROUTER_URL


def test_anthropic_and_google_clients():
    assert type(make("anthropic", "claude-sonnet-4-5")).__name__ == "ChatAnthropic"
    assert type(make("google", "gemini-2.5-flash")).__name__ == "ChatGoogleGenerativeAI"


def test_offline_model_and_openrouter_key_rule():
    assert isinstance(make("offline", "offline"), OfflineModel)
    with pytest.raises(ValueError):
        Settings(env="test", llm_provider="openrouter", _env_file=None)
