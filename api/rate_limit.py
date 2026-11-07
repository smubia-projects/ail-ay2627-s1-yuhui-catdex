"""Project adapter over the shared AI Lodge rate-limit engine.

``rate_limit_engine.py`` is vendored unmodified from the club's shared engine. This adapter supplies the
project slug and its two buckets; the engine handles X-Forwarded-For, central limits, event mode, the kill
switch (503) and fail-open. Without Upstash keys it is off (local dev).

- ``scan``: one hit per scan (the student's YOLO, CLIP and re-ID models on this server's CPU).
- ``card``: one hit per trading card written by the language model. Template cards cost nothing.
"""

from __future__ import annotations

import os

from fastapi import Request

from api.rate_limit_engine import RateLimiter


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


limiter = RateLimiter(
    # Slug = full project folder name (matches the central control sheet rows).
    project="ail-ay2627-s1-yuhui-catdex",
    buckets={
        "scan": _env_int("RATE_LIMIT_SCAN_MAX", 20),
        "card": _env_int("RATE_LIMIT_CARD_MAX", 5),
    },
    redis_url=os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("UPSTASH_REDIS_URL") or "",
    redis_token=os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("UPSTASH_REDIS_TOKEN") or "",
    default_window=_env_int("RATE_LIMIT_WINDOW_SECONDS", 432000),
)


def check_rate_limit(request: Request, bucket: str) -> None:
    limiter.enforce(request, bucket)
