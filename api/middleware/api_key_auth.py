"""API Key authentication middleware."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header against a list of allowed keys.

    Skips health check and documentation endpoints.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        api_keys: set[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._enabled = enabled
        self._api_keys = api_keys or set()
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

        api_key = (
            request.headers.get("X-API-Key")
            or request.headers.get("Authorization", "").replace("Bearer ", "")
        )

        if not api_key:
            logger.warning("Missing API key for %s %s", request.method, path)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required. Provide X-API-Key header.",
            )

        if api_key not in self._api_keys:
            logger.warning("Invalid API key for %s %s", request.method, path)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key.",
            )

        from starlette.responses import Response

        response = await call_next(request)
        return response
