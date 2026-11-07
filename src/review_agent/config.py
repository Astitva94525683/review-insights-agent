"""All settings come from environment variables (prefix RIA_) or a .env file.

Nothing secret is hard-coded. In production (RIA_ENV=prod) the app refuses to start
without API keys, so it can never run open to the internet by mistake.
"""
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["nvidia", "openai", "anthropic", "google", "openrouter", "offline"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RIA_", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    service_name: str = "review-insights-agent"

    # Storage: SQLite for local runs, PostgreSQL in production (postgresql+psycopg://...).
    database_url: str = "sqlite:///./data/app.db"
    seed_sample_data: bool = False  # load the sample catalogue on start-up when the database is empty
    sample_data_dir: str = "data/sample"

    # LLM. Default: NVIDIA Nemotron 3 Super on NVIDIA's free hosted API (key from build.nvidia.com).
    # Each provider also reads its standard key variable (NVIDIA_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).
    llm_provider: Provider = "nvidia"
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b"
    llm_api_key: SecretStr | None = None  # optional override of the provider's key variable
    llm_temperature: float = 0.0
    llm_max_tokens: int = Field(1024, ge=64, le=32000)  # cap on each answer's length
    llm_reasoning: bool = False  # Nemotron "thinking" mode: better on hard questions, slower
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3  # retries with backoff on rate limits (the free tier has them) and 5xx
    llm_input_cost_per_mtok: float = 0.0  # USD per 1M input tokens; 0 on NVIDIA's free tier
    llm_output_cost_per_mtok: float = 0.0  # set both when you move to a paid model

    # Agent budgets: hard limits so one request can never loop or spend without bound.
    max_agent_steps: int = Field(6, ge=1, le=20)
    max_tool_calls: int = Field(8, ge=1, le=30)
    max_input_chars: int = Field(2000, ge=50)
    max_output_chars: int = Field(4000, ge=200)
    search_default_limit: int = Field(5, ge=1, le=8)

    # Memory
    history_token_budget: int = Field(2500, ge=200)  # recent turns kept word-for-word
    summary_max_chars: int = Field(1500, ge=200)

    # API security
    api_keys: SecretStr = SecretStr("")  # comma-separated client keys
    admin_api_key: SecretStr = SecretStr("")
    rate_limit_per_minute: int = Field(30, ge=1)
    cors_origins: str = ""  # comma-separated; empty = CORS off

    # Observability
    log_level: str = "INFO"
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    otel_endpoint: str = "http://localhost:4318/v1/traces"

    @property
    def client_keys(self) -> list[str]:
        return [k.strip() for k in self.api_keys.get_secret_value().split(",") if k.strip()]

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _check_production(self) -> "Settings":
        if self.env == "prod":
            if not self.client_keys or not self.admin_api_key.get_secret_value():
                raise ValueError("RIA_API_KEYS and RIA_ADMIN_API_KEY are required when RIA_ENV=prod")
            if self.llm_provider == "offline":
                raise ValueError("The offline model is for local runs and CI only, not production")
        if self.llm_provider == "openrouter" and self.llm_api_key is None:
            raise ValueError("RIA_LLM_API_KEY is required for the openrouter provider")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(override=False)  # also exposes provider keys (NVIDIA_API_KEY, LANGSMITH_*, ...) to the SDKs
    return Settings()
