from __future__ import annotations

import time
from typing import Any

import redis
from fastapi import HTTPException, Request, status

from app.config import settings

_redis: redis.Redis | None = None


def _client() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def rate_limit(key: str, capacity: int, refill_per_sec: float) -> None:
    """Token bucket via Redis. Raises 429 if depleted.
    Degrades gracefully when Redis is unreachable (no-ops with a warning).
    """
    now = time.time()
    bucket_key = f"rl:{key}"
    try:
        client = _client()
        pipe = client.pipeline()
        pipe.hgetall(bucket_key)
        state: dict[str, Any] = pipe.execute()[0] or {}
        tokens = float(state.get("tokens", capacity))
        last = float(state.get("last", now))
        tokens = min(capacity, tokens + (now - last) * refill_per_sec)
        if tokens < 1:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limited")
        tokens -= 1
        client.hset(bucket_key, mapping={"tokens": tokens, "last": now})
        client.expire(bucket_key, 3600)
    except HTTPException:
        raise
    except Exception:
        # Redis down — fail open. In production we want a circuit breaker that
        # falls back to a per-process token bucket; for now, no-op is safer than
        # 500-ing every request.
        return


def per_ip_limiter(capacity: int = 60, refill_per_sec: float = 1.0):
    async def _dep(request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        rate_limit(f"ip:{ip}:{request.url.path}", capacity, refill_per_sec)

    return _dep
