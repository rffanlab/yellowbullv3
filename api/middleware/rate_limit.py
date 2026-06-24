"""Rate limiting middleware — per-user sliding window."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class _RateLimitEntry:
    timestamps: list[float] = field(default_factory=list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by user identifier.

    Extracts the API key (or IP as fallback) from the request to identify users.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_requests: int = 60,
        window_seconds: int = 60,
        exclude_paths: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._enabled = enabled
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._buckets: dict[str, _RateLimitEntry] = defaultdict(_RateLimitEntry)
        self._exclude_paths = exclude_paths or ["/api/health", "/docs", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable]) -> None:  # type: ignore[override]
        if not self._enabled:
            from starlette.responses import Response

            response = await call_next(request)
            return response

        path = request.url.path
        for excluded in self._exclude_paths:
            if path.startswith(excluded):
                from starlette.responses import Response

                response = await call_next(request)
                return response

        user_id = self._extract_user_id(request)
        if not self._check_limit(user_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
                headers={"Retry-After": str(self._window_seconds)},
            )

        from starlette.responses import Response

        response = await call_next(request)
        return response

    def _extract_user_id(self, request: Request) -> str:
        """Get user identifier from API key header or fall back to IP."""
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
        if api_key:
            return f"key:{api_key}"
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"

    def _check_limit(self, user_id: str) -> bool:
        """Check and record request against sliding window."""
        now = time.monotonic()
        cutoff = now - self._window_seconds
        entry = self._buckets[user_id]

        # Prune old timestamps
        entry.timestamps = [t for t in entry.timestamps if t > cutoff]

        if len(entry.timestamps) >= self._max_requests:
            logger.warning("Rate limit exceeded for user=%s", user_id)
            return False

        entry.timestamps.append(now)
        return True
