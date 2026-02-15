"""Simple sliding-window rate limiter.

In-memory implementation suitable for a single-worker PoC.
For production use Redis-backed counters shared across workers.
"""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Paths exempt from rate limiting.
_EXEMPT_PATHS = frozenset({"/health", "/ready"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding window rate limiter.

    Args:
        app: The ASGI application.
        max_requests: Maximum requests allowed per *window_seconds*.
        window_seconds: Sliding window duration in seconds.
    """

    def __init__(
        self,
        app,  # noqa: ANN001
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self._max = max_requests
        self._window = window_seconds
        # Mapping: client_ip -> list of request timestamps
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start = now - self._window

        # Prune old entries
        hits = self._hits[client_ip]
        self._hits[client_ip] = [t for t in hits if t > window_start]
        hits = self._hits[client_ip]

        if len(hits) >= self._max:
            retry_after = int(hits[0] - window_start) + 1
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": (
                            f"Rate limit exceeded: {self._max} requests "
                            f"per {self._window}s."
                        ),
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        hits.append(now)
        response = await call_next(request)
        remaining = max(0, self._max - len(hits))
        response.headers["X-RateLimit-Limit"] = str(self._max)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
