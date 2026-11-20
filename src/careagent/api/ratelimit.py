"""
careagent.api.ratelimit
~~~~~~~~~~~~~~~~~~~~~~~~
Per-client sliding-window rate limit for POST /analyze.

Each /analyze can call the LLM, so an unthrottled public endpoint lets one
client spend the whole OpenAI budget. This keeps the demo usable for
everyone. It is in-memory per instance: with Cloud Run max-instances=2 the
effective ceiling is at most 2x the limit, which is fine for a demo; a
shared store (Redis, Cloud Armor) would be the production version.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable

from fastapi import HTTPException, Request

from careagent.config import get_settings

MAX_TRACKED_CLIENTS = 10_000


class SlidingWindowLimiter:
    def __init__(self, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.window = window_seconds
        self.clock = clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> float:
        """Record a request. Returns 0 if allowed, else seconds until allowed."""
        if limit <= 0:
            return 0.0
        now = self.clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= limit:
                return max(self.window - (now - hits[0]), 0.001)
            hits.append(now)
            if len(self._hits) > MAX_TRACKED_CLIENTS:
                self._evict_idle(now)
            return 0.0

    def _evict_idle(self, now: float) -> None:
        idle = [k for k, h in self._hits.items() if not h or now - h[-1] >= self.window]
        for k in idle:
            del self._hits[k]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


analyze_limiter = SlidingWindowLimiter()


def client_key(request: Request) -> str:
    """Client IP. Behind Cloud Run the real client is the last X-Forwarded-For
    entry (the front end appends it); earlier entries can be set by the caller."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def limit_analyze(request: Request) -> None:
    """FastAPI dependency: 429 with Retry-After when the client is over the limit."""
    wait = analyze_limiter.check(client_key(request), get_settings().rate_limit_per_minute)
    if wait:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a minute and try again.",
            headers={"Retry-After": str(math.ceil(wait))},
        )
