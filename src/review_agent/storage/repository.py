"""All database reads and writes live here. Tools, memory and the API never write SQL."""
from collections import Counter
from collections.abc import Iterable

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import sessionmaker

from .db import AgentRun, ChatSession, Feedback, Message, Product, Review


class Repository:
    def __init__(self, session_factory: sessionmaker):
        self._sf = session_factory

    # ---- catalogue -------------------------------------------------------------------
    def upsert_products(self, products: Iterable[dict]) -> int:
        n = 0
        with self._sf.begin() as db:
            for p in products:
                db.merge(Product(**p))
                n += 1
        return n

    def product_ids(self) -> set[str]:
        with self._sf() as db:
            return set(db.scalars(select(Product.id)))

    def list_products(self, category: str | None = None) -> list[dict]:
        with self._sf() as db:
            q = select(Product).order_by(Product.name)
            if category:
                q = q.where(func.lower(Product.category) == category.lower())
            return [{"product_id": p.id, "name": p.name, "category": p.category,
                     "description": p.description} for p in db.scalars(q)]

    def existing_review_keys(self) -> tuple[set[str], set[str]]:
        """(review ids, content hashes) already stored, used to skip duplicates on import."""
        with self._sf() as db:
            rows = db.execute(select(Review.id, Review.content_hash)).all()
        return {r[0] for r in rows}, {r[1] for r in rows}

    def review_count(self) -> int:
        with self._sf() as db:
            return db.scalar(select(func.count()).select_from(Review)) or 0

    def add_reviews(self, reviews: list[dict]) -> int:
        with self._sf.begin() as db:
            db.add_all(Review(**r) for r in reviews)
        return len(reviews)

    def all_reviews(self) -> list[dict]:
        with self._sf() as db:
            rows = db.execute(select(Review, Product.name).join(Product, Product.id == Review.product_id))
            return [{"review_id": r.id, "product_id": r.product_id, "product_name": name, "rating": r.rating,
                     "title": r.title, "body": r.body, "reviewer_role": r.reviewer_role,
                     "company_size": r.company_size, "reviewed_on": r.reviewed_on.isoformat()}
                    for r, name in rows]

    def product_stats(self, product_id: str) -> dict | None:
        with self._sf() as db:
            product = db.get(Product, product_id)
            if product is None:
                return None
            ratings = list(db.scalars(select(Review.rating).where(Review.product_id == product_id)))
            latest = db.scalar(select(func.max(Review.reviewed_on)).where(Review.product_id == product_id))
        dist = Counter(ratings)
        return {"product_id": product.id, "name": product.name, "category": product.category,
                "review_count": len(ratings),
                "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
                "rating_distribution": {str(s): dist.get(s, 0) for s in range(5, 0, -1)},
                "latest_review": latest.isoformat() if latest else None}

    # ---- sessions and memory ----------------------------------------------------------
    def get_or_create_session(self, session_id: str, client_id: str) -> ChatSession:
        with self._sf.begin() as db:
            s = db.get(ChatSession, session_id)
            if s is None:
                s = ChatSession(id=session_id, client_id=client_id, summary="", profile={})
                db.add(s)
            elif s.client_id != client_id:
                raise PermissionError("This session belongs to another client.")
            return s

    def get_session(self, session_id: str, client_id: str) -> ChatSession | None:
        with self._sf() as db:
            s = db.get(ChatSession, session_id)
            return s if s is not None and s.client_id == client_id else None

    def update_session(self, session_id: str, *, summary: str | None = None, profile: dict | None = None):
        values = {k: v for k, v in {"summary": summary, "profile": profile}.items() if v is not None}
        if values:
            with self._sf.begin() as db:
                db.execute(update(ChatSession).where(ChatSession.id == session_id).values(**values))

    def delete_session(self, session_id: str, client_id: str) -> bool:
        with self._sf.begin() as db:
            s = db.get(ChatSession, session_id)
            if s is None or s.client_id != client_id:
                return False
            db.execute(delete(Message).where(Message.session_id == session_id))
            db.delete(s)
            return True

    def add_messages(self, session_id: str, request_id: str, pairs: list[tuple[str, str]]):
        with self._sf.begin() as db:
            db.add_all(Message(session_id=session_id, role=role, content=content, request_id=request_id)
                       for role, content in pairs)

    def messages(self, session_id: str, *, unsummarized_only: bool = False) -> list[Message]:
        with self._sf() as db:
            q = select(Message).where(Message.session_id == session_id).order_by(Message.id)
            if unsummarized_only:
                q = q.where(Message.summarized.is_(False))
            return list(db.scalars(q))

    def mark_summarized(self, message_ids: list[int]):
        if message_ids:
            with self._sf.begin() as db:
                db.execute(update(Message).where(Message.id.in_(message_ids)).values(summarized=True))

    # ---- runs and feedback ------------------------------------------------------------
    def save_run(self, run: dict):
        with self._sf.begin() as db:
            db.add(AgentRun(**run))

    def recent_runs(self, limit: int = 50, flagged_only: bool = False) -> list[dict]:
        with self._sf() as db:
            q = select(AgentRun).order_by(AgentRun.id.desc()).limit(limit)
            if flagged_only:
                q = q.where((AgentRun.status != "answered") | AgentRun.citation_valid.is_(False))
            cols = [c.name for c in AgentRun.__table__.columns]
            return [{c: (getattr(r, c).isoformat() if c == "created_at" else getattr(r, c)) for c in cols}
                    for r in db.scalars(q)]

    def run_session(self, request_id: str) -> str | None:
        with self._sf() as db:
            return db.scalar(select(AgentRun.session_id).where(AgentRun.request_id == request_id)
                             .order_by(AgentRun.id.desc()).limit(1))

    def save_feedback(self, request_id: str, score: int, comment: str):
        with self._sf.begin() as db:
            db.add(Feedback(request_id=request_id, score=score, comment=comment))

    def ping(self) -> bool:
        with self._sf() as db:
            return db.scalar(select(1)) == 1
