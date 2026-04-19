"""Optional async Redis client for caching (disabled when REDIS_URL is unset)."""

from __future__ import annotations

from typing import Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.core.config import settings
from app.core.logging import logger

_client: Optional[Redis] = None


async def get_redis() -> Optional[Redis]:
    """Return shared Redis client, or None if caching is disabled."""
    global _client
    if not settings.REDIS_URL:
        return None
    if _client is None:
        _client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("redis_client_initialized", redis_configured=True)
    return _client


async def close_redis() -> None:
    """Close Redis connection (e.g. on app shutdown)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("redis_client_closed")
