"""Request and response models. FastAPI validates every request against these."""
from typing import Literal

from pydantic import BaseModel, Field

SESSION_ID = r"^[A-Za-z0-9_-]{8,64}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)  # the input guardrail applies the real limit
    session_id: str | None = Field(None, pattern=SESSION_ID,
                                   description="Omit to start a new conversation; reuse to continue it.")


class Citation(BaseModel):
    review_id: str
    product: str
    rating: int
    title: str
    excerpt: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cost_usd: float


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    status: Literal["answered", "blocked_input", "fallback", "step_limit"]
    answer: str
    citations: list[Citation]
    usage: Usage
    latency_ms: float
    guardrail_flags: list[str]


class FeedbackRequest(BaseModel):
    request_id: str = Field(max_length=64)
    score: Literal[1, -1]
    comment: str = Field("", max_length=1000)


class SessionMessage(BaseModel):
    role: str
    content: str
    created_at: str


class SessionView(BaseModel):
    session_id: str
    summary: str
    profile: dict
    messages: list[SessionMessage]


class ImportRequest(BaseModel):
    products: list[dict] = Field(default_factory=list, max_length=1000)
    reviews: list[dict] = Field(default_factory=list, max_length=5000)
