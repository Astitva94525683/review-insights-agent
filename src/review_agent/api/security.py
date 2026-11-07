"""API-key authentication and per-client rate limiting."""
import hashlib
import hmac
import threading
import time

from fastapi import HTTPException, Request, status


def client_id_for(key: str) -> str:
    """Stable, non-reversible client ID derived from the key; sessions are bound to it."""
    return "client-" + hashlib.sha256(key.encode()).hexdigest()[:16]


def authenticate(request: Request) -> str:
    settings = request.app.state.services.settings
    keys = settings.client_keys
    presented = request.headers.get("x-api-key", "")
    if not keys:  # local development only: production refuses to start without keys
        return "local-dev"
    for key in keys:
        if hmac.compare_digest(presented.encode(), key.encode()):  # constant time: no timing leaks
            return client_id_for(key)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key.")


def authenticate_admin(request: Request) -> None:
    admin = request.app.state.services.settings.admin_api_key.get_secret_value()
    presented = request.headers.get("x-admin-key", "")
    if not admin or not hmac.compare_digest(presented.encode(), admin.encode()):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin key required.")


class RateLimiter:
    """Token bucket per client: `per_minute` requests, refilled continuously.

    In-process, so each API worker keeps its own buckets. For many replicas, move the buckets to
    Redis; the interface stays the same.
    """

    def __init__(self, per_minute: int):
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, client_id: str) -> float:
        """Return 0 if allowed, otherwise seconds to wait."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(client_id, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1:
                self._buckets[client_id] = (tokens - 1, now)
                return 0.0
            self._buckets[client_id] = (tokens, now)
            return (1 - tokens) / self.rate
