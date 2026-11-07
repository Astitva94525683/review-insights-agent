"""Database tables and engine. Works on SQLite (local) and PostgreSQL (production)."""
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(100), index=True)
    description: Mapped[str] = mapped_column(Text, default="")


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (CheckConstraint("rating BETWEEN 1 AND 5", name="rating_range"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"), index=True)
    rating: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text)
    reviewer_role: Mapped[str] = mapped_column(String(120), default="")
    company_size: Mapped[str] = mapped_column(String(40), default="")
    reviewed_on: Mapped[datetime] = mapped_column(Date)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)  # blocks duplicate imports


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    client_id: Mapped[str] = mapped_column(String(64), index=True)  # which API client owns it
    summary: Mapped[str] = mapped_column(Text, default="")  # rolling summary of older turns
    profile: Mapped[dict] = mapped_column(JSON, default=dict)  # long-term buyer preferences
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # "human" or "ai"
    content: Mapped[str] = mapped_column(Text)
    request_id: Mapped[str] = mapped_column(String(64), default="")
    summarized: Mapped[bool] = mapped_column(Boolean, default=False)  # folded into the session summary
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AgentRun(Base):
    """One row per chat request: the run record used for trace review and online evals."""

    __tablename__ = "agent_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)  # client retries may reuse an ID
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32))  # answered | blocked_input | fallback | error
    model: Mapped[str] = mapped_column(String(120))
    latency_ms: Mapped[float] = mapped_column(Float)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    steps: Mapped[int] = mapped_column(Integer, default=0)
    tools_used: Mapped[list] = mapped_column(JSON, default=list)
    guardrail_flags: Mapped[list] = mapped_column(JSON, default=list)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    citation_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    score: Mapped[int] = mapped_column(Integer)  # +1 helpful, -1 not helpful
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        path = url.split("///", 1)[-1]
        if path not in ("", ":memory:"):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(conn, _record):  # foreign keys + WAL for concurrent readers
            cur = conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)


def make_session_factory(engine: Engine) -> sessionmaker:
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
