"""Review ingestion: validate, clean and de-duplicate reviews before they reach the search index.

Used by the admin API (POST /v1/admin/reviews/import) and from the command line:
    ria-import --products data/sample/products.json --reviews data/sample/reviews.jsonl
"""
import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..guardrails import detect_injection, redact_pii
from ..storage.repository import Repository


class ProductIn(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str = Field(min_length=2, max_length=200)
    category: str = Field(min_length=2, max_length=100)
    description: str = Field("", max_length=2000)


class ReviewIn(BaseModel):
    id: str = Field(pattern=r"^R[0-9A-Za-z_-]{1,63}$")
    product_id: str
    rating: int = Field(ge=1, le=5)
    title: str = Field(min_length=3, max_length=300)
    body: str = Field(min_length=20, max_length=5000)
    reviewer_role: str = Field("", max_length=120)
    company_size: Literal["1-50", "51-200", "201-1000", "1000+", ""] = ""
    reviewed_on: date


class ImportReport(BaseModel):
    received: int = 0
    accepted: int = 0
    duplicates: int = 0
    pii_redacted: int = 0
    injection_flagged: int = 0  # kept (real user text) but neutralised at retrieval time
    rejected: list[dict] = Field(default_factory=list)


def content_hash(product_id: str, body: str) -> str:
    normalized = re.sub(r"\s+", " ", body.lower()).strip()
    return hashlib.sha256(f"{product_id}|{normalized}".encode()).hexdigest()


def import_reviews(repo: Repository, records: list[dict]) -> ImportReport:
    report = ImportReport(received=len(records))
    known_products = repo.product_ids()
    seen_ids, seen_hashes = repo.existing_review_keys()
    accepted: list[dict] = []
    for i, raw in enumerate(records):
        try:
            r = ReviewIn.model_validate(raw)
        except ValidationError as exc:
            errors = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
            report.rejected.append({"index": i, "id": raw.get("id") if isinstance(raw, dict) else None,
                                    "errors": errors})
            continue
        if r.product_id not in known_products:
            report.rejected.append({"index": i, "id": r.id, "errors": [f"unknown product_id {r.product_id}"]})
            continue
        h = content_hash(r.product_id, r.body)
        if r.id in seen_ids or h in seen_hashes:
            report.duplicates += 1
            continue
        title, k1 = redact_pii(r.title)
        body, k2 = redact_pii(r.body)
        report.pii_redacted += int(bool(k1 or k2))
        report.injection_flagged += int(bool(detect_injection(body)))
        accepted.append({**r.model_dump(), "title": title, "body": body, "content_hash": h})
        seen_ids.add(r.id)
        seen_hashes.add(h)
    if accepted:
        repo.add_reviews(accepted)
    report.accepted = len(accepted)
    return report


def import_products(repo: Repository, records: list[dict]) -> int:
    return repo.upsert_products(ProductIn.model_validate(p).model_dump() for p in records)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_directory(repo: Repository, data_dir: Path) -> ImportReport:
    import_products(repo, json.loads((data_dir / "products.json").read_text(encoding="utf-8")))
    return import_reviews(repo, read_jsonl(data_dir / "reviews.jsonl"))


def main() -> None:
    from ..config import get_settings
    from ..storage.db import make_engine, make_session_factory

    parser = argparse.ArgumentParser(description="Import products and reviews into the database.")
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    args = parser.parse_args()
    repo = Repository(make_session_factory(make_engine(get_settings().database_url)))
    import_products(repo, json.loads(args.products.read_text(encoding="utf-8")))
    print(import_reviews(repo, read_jsonl(args.reviews)).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
