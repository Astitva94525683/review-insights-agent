"""Keyword search over reviews with BM25, the ranking used by Elasticsearch and OpenSearch.

Why BM25 and not embeddings: review questions are full of exact terms (product names, "SSO",
"Salesforce", "pricing") that keyword search matches reliably, it needs no extra API calls, and
the score is easy to explain. The SearchIndex interface stays the same if a vector index is added.
"""
import math
import re
import threading
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOP = frozenset("""a an and are as at be but by for from has have i in is it its of on or our so that the
their them they this to was we were with you your my me can do does how what which who will would
about than then there these those too very just also into if not no""".split())


def tokenize(text: str) -> list[str]:
    tokens = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # light stemming: "integrations" -> "integration"
        tokens.append(t)
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._lock = threading.Lock()
        self._docs: list[dict] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: dict[str, int] = defaultdict(int)
        self._avg = 0.0

    def build(self, docs: list[dict]) -> None:
        """Rebuild from scratch, then swap in atomically so searches never see a half-built index."""
        tf = [Counter(tokenize(f"{d['title']} {d['title']} {d['body']} {d['product_name']}")) for d in docs]
        df: dict[str, int] = defaultdict(int)
        for counts in tf:
            for term in counts:
                df[term] += 1
        lengths = [sum(c.values()) for c in tf]
        with self._lock:
            self._docs, self._tf, self._len, self._df = docs, tf, lengths, df
            self._avg = (sum(lengths) / len(lengths)) if lengths else 0.0

    @property
    def size(self) -> int:
        return len(self._docs)

    def search(self, query: str, *, product_id: str | None = None, min_rating: int | None = None,
               max_rating: int | None = None, limit: int = 5) -> list[tuple[dict, float]]:
        terms = tokenize(query)
        with self._lock:
            docs, tf, lengths, df, avg = self._docs, self._tf, self._len, self._df, self._avg
        n = len(docs)
        scored = []
        for i, d in enumerate(docs):
            if product_id and d["product_id"] != product_id:
                continue
            if min_rating is not None and d["rating"] < min_rating:
                continue
            if max_rating is not None and d["rating"] > max_rating:
                continue
            score = 0.0
            for t in terms:
                f = tf[i].get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * lengths[i] / avg))
            if score > 0 or not terms:
                scored.append((d, score))
        scored.sort(key=lambda x: (-x[1], x[0]["review_id"]))
        return scored[:limit]
