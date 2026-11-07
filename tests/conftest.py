from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from review_agent.agent import AgentRunner
from review_agent.config import Settings
from review_agent.ingestion.importer import load_directory
from review_agent.observability import Metrics
from review_agent.retrieval.bm25 import BM25Index
from review_agent.storage.db import make_engine, make_session_factory
from review_agent.storage.repository import Repository

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"
SPANS = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(SPANS))
trace.set_tracer_provider(_provider)


class ScriptedModel(BaseChatModel):
    """Returns pre-written replies in order and records what it was sent. For precise agent tests."""

    replies: list = []
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        msg = reply if isinstance(reply, AIMessage) else AIMessage(reply)
        msg.usage_metadata = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{name}", "type": "tool_call"}])


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(env="test", database_url=f"sqlite:///{tmp_path}/test.db", llm_provider="offline",
                    llm_model="offline-heuristic", _env_file=None)


@pytest.fixture
def repo(settings):
    engine = make_engine(settings.database_url)
    r = Repository(make_session_factory(engine))
    load_directory(r, SAMPLE)
    yield r
    engine.dispose()


@pytest.fixture
def index(repo) -> BM25Index:
    idx = BM25Index()
    idx.build(repo.all_reviews())
    return idx


@pytest.fixture
def scripted():
    return ScriptedModel(replies=[], seen=[])


@pytest.fixture
def make_runner(settings, repo, index):
    def _make(model, **overrides):
        s = settings.model_copy(update=overrides) if overrides else settings
        return AgentRunner(s, repo, index, model, Metrics(), canary="ria-test-canary")
    return _make


@pytest.fixture
def spans():
    SPANS.clear()
    return SPANS
