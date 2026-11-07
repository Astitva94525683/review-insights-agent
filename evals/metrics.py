"""Small, dependency-free metric functions used by the eval runner."""
import statistics


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def rate(passed: int, total: int) -> float:
    return round(passed / total, 4) if total else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1], 2)
