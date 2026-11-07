from review_agent.ingestion.importer import import_reviews
from review_agent.retrieval.bm25 import BM25Index, tokenize


def test_sample_import_cleans_pii_and_flags_injection(repo):
    reviews = {r["review_id"]: r for r in repo.all_reviews()}
    assert len(reviews) == 50
    assert "98765" not in reviews["R020"]["body"] and "[PHONE]" in reviews["R020"]["body"]
    assert "[EMAIL]" in reviews["R037"]["body"]


def test_import_rejects_bad_rows_and_skips_duplicates(repo):
    good = {"id": "R900", "product_id": "plannova", "rating": 4, "title": "Solid tool",
            "body": "Works well for our agency and the timeline view is clear.", "reviewed_on": "2026-10-01"}
    report = import_reviews(repo, [
        good,
        dict(good, id="R901"),                                   # same text as R900 -> duplicate
        dict(good, id="R902", rating=9, body="Different text that is long enough."),  # bad rating
        dict(good, id="R903", product_id="nope", body="Another different body text here."),
        {"id": "bad"},
    ])
    assert (report.accepted, report.duplicates, len(report.rejected)) == (1, 1, 3)
    assert import_reviews(repo, [good]).duplicates == 1  # re-import is idempotent


def test_tokenizer_light_stemming():
    assert tokenize("The integrations and Integration") == ["integration", "integration"]


def test_search_ranks_and_filters(index):
    top = index.search("SSO Okta", limit=3)
    assert top[0][0]["review_id"] == "R007"
    only_low = index.search("support", min_rating=1, max_rating=2, limit=8)
    assert only_low and all(d["rating"] <= 2 for d, _ in only_low)
    assert all(d["product_id"] == "plannova" for d, _ in index.search("app", product_id="plannova"))


def test_empty_index_is_safe():
    assert BM25Index().search("anything") == []
