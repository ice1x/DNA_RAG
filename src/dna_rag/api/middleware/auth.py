"""API-key authentication middleware.

When ``auth_enabled=True`` in :class:`~dna_rag.api.config.APISettings`,
every request must include a valid ``Authorization: Bearer <key>`` header.

Health and readiness endpoints are exempt.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Paths that never require auth.
_PUBLIC_PATHS = frozenset({"/health", "/ready", "/openapi.json", "/docs", "/redoc"})


class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer token API-key auth middleware.

    Args:
        app: The ASGI application.
        api_keys: Set of valid API keys.
        enabled: If ``False`` all requests are allowed.
    """

    def __init__(
        self,
        app,  # noqa: ANN001
        api_keys: set[str] | None = None,
        enabled: bool = False,
    ) -> None:
        super().__init__(app)
        self._api_keys = api_keys or set()
        self._enabled = enabled

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        if not self._enabled:
            return await call_next(request)

        # Exempt public paths
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Missing or invalid Authorization header.",
                    }
                },
            )

        token = auth_header[7:]  # strip "Bearer "
        if token not in self._api_keys:
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "code": "FORBIDDEN",
                        "message": "Invalid API key.",
                    }
                },
            )

        return await call_next(request)
