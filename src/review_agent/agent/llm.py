"""Model factory. One setting switches between NVIDIA Nemotron (default), OpenAI, Anthropic, Google Gemini
and OpenRouter. Every client gets a timeout, an output cap and automatic retries with backoff for rate
limits and transient errors.
"""
import os

from langchain_core.language_models import BaseChatModel

from ..config import Settings

NVIDIA_URL = "https://integrate.api.nvidia.com/v1"  # OpenAI-compatible; free tier with a build.nvidia.com key
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def build_chat_model(s: Settings) -> BaseChatModel:
    key = s.llm_api_key.get_secret_value() if s.llm_api_key else None
    common = {"temperature": s.llm_temperature, "timeout": s.llm_timeout_s, "max_retries": s.llm_max_retries}
    if s.llm_provider in ("nvidia", "openai", "openrouter"):
        from langchain_openai import ChatOpenAI

        kwargs = {"model": s.llm_model, "max_tokens": s.llm_max_tokens, **common}
        if s.llm_provider == "nvidia":
            key = key or os.getenv("NVIDIA_API_KEY")
            if not key:
                raise ValueError("Set NVIDIA_API_KEY (create one at build.nvidia.com) or RIA_LLM_API_KEY")
            kwargs["base_url"] = NVIDIA_URL
            # Nemotron 3 reasons by default; off keeps answers fast and keeps thinking text out of replies.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": s.llm_reasoning}}
        elif s.llm_provider == "openrouter":
            kwargs["base_url"] = OPENROUTER_URL
        if key:
            kwargs["api_key"] = key
        return ChatOpenAI(**kwargs)
    if s.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # pip install .[anthropic]

        return ChatAnthropic(model=s.llm_model, max_tokens=s.llm_max_tokens, **common,
                             **({"api_key": key} if key else {}))
    if s.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI  # pip install .[google]

        return ChatGoogleGenerativeAI(model=s.llm_model, max_output_tokens=s.llm_max_tokens, **common,
                                      **({"google_api_key": key} if key else {}))
    from .offline_model import OfflineModel

    return OfflineModel()
