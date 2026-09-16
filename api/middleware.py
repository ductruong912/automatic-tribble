"""
Custom middleware for the FastAPI application.
Provides security headers, rate limiting, request logging, error handling,
cache control, and streaming-safe GZip compression.
"""

import asyncio
import logging
import time
import uuid
from typing import Callable, Dict, List, Optional, Set

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

logger = logging.getLogger(__name__)

DEFAULT_EXEMPT_PATHS: Set[str] = {
    "/",
    "/detailed",
    "/health",
    "/health/detailed",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}


class StreamingSafeGZipMiddleware(GZipMiddleware):
    """
    GZip every response except streamed routes, which pass through untouched.
    Starlette's gzip never flushes mid-stream, so SSE tokens would reach the client only at the end.
    """

    def __init__(self, app, exclude_paths=("/chat",), **kwargs):
        super().__init__(app, **kwargs)
        self.exclude_paths = tuple(exclude_paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith(self.exclude_paths):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers for enhanced application security.
    Implements common security best practices for web applications.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


class CacheControlMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add appropriate cache control headers.
    Optimizes caching for static content while preventing API response caching.
    """

    def __init__(self, app, static_max_age: int = 3600, api_max_age: int = 0):
        super().__init__(app)
        self.static_max_age = static_max_age
        self.api_max_age = api_max_age

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = f"public, max-age={self.static_max_age}"
        elif request.url.path.startswith("/docs") or request.url.path.startswith("/redoc"):
            response.headers["Cache-Control"] = f"public, max-age={self.static_max_age}"
        else:
            response.headers["Cache-Control"] = f"no-cache, no-store, max-age={self.api_max_age}"
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Enhanced sliding-window rate limiting per client IP address and global concurrent request limiting.
    Exempts health checks, documentation endpoints, and CORS preflight OPTIONS requests.
    """

    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
        max_concurrent: int = 50,
        exempt_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_concurrent = max_concurrent
        self.exempt_paths = exempt_paths or DEFAULT_EXEMPT_PATHS
        self.requests: Dict[str, List[float]] = {}
        self.concurrent_requests = 0
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Exempt preflight OPTIONS and specified paths
        if request.method == "OPTIONS" or request.url.path in self.exempt_paths:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Check rate and concurrency limits under lock
        async with self._lock:
            # Periodic dictionary cleanup to prevent memory leak
            if current_time - self._last_cleanup > 120:
                self._cleanup_stale_ips(current_time)
                self._last_cleanup = current_time

            # Filter old timestamps for this IP
            timestamps = self.requests.get(client_ip, [])
            timestamps = [t for t in timestamps if current_time - t < self.window_seconds]

            # Check rate limit
            if len(timestamps) >= self.max_requests:
                logger.warning(
                    f"🚫 Rate limit exceeded for {client_ip}: {len(timestamps)} requests in {self.window_seconds}s"
                )
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please try again later."},
                    headers={"Retry-After": str(self.window_seconds)},
                )

            # Check global concurrency limit
            if self.concurrent_requests >= self.max_concurrent:
                logger.warning(
                    f"🚫 Concurrent request limit exceeded ({self.concurrent_requests}/{self.max_concurrent})"
                )
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Server busy, too many concurrent requests. Please try again later."},
                    headers={"Retry-After": "5"},
                )

            # Record request
            timestamps.append(current_time)
            self.requests[client_ip] = timestamps
            current_count = len(timestamps)

        # Track concurrency during execution
        async with self.semaphore:
            self.concurrent_requests += 1
            try:
                response = await call_next(request)
                # Add rate limit headers for client information
                response.headers["X-RateLimit-Limit"] = str(self.max_requests)
                response.headers["X-RateLimit-Remaining"] = str(
                    max(0, self.max_requests - current_count)
                )
                response.headers["X-RateLimit-Reset"] = str(
                    int(current_time + self.window_seconds)
                )
                return response
            finally:
                self.concurrent_requests -= 1

    def _cleanup_stale_ips(self, current_time: float):
        """Remove IPs whose requests have all expired outside the window."""
        stale_ips = [
            ip for ip, timestamps in self.requests.items()
            if not timestamps or current_time - timestamps[-1] >= self.window_seconds
        ]
        for ip in stale_ips:
            del self.requests[ip]


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log incoming requests, generate request IDs, and measure response times.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = uuid.uuid4().hex[:8]
        start_time = time.time()
        is_health = request.url.path in ("/", "/health", "/health/detailed", "/favicon.ico")

        if not is_health:
            client_ip = request.client.host if request.client else "unknown"
            logger.info(
                f"📥 [{request_id}] {request.method} {request.url.path} - Client: {client_ip}"
            )

        response = await call_next(request)
        process_time = time.time() - start_time

        if not is_health:
            logger.info(
                f"📤 [{request_id}] {request.method} {request.url.path} - Status: {response.status_code} ({process_time:.3f}s)"
            )

        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Request-ID"] = request_id
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Centralized error handling middleware to catch unhandled exceptions gracefully.
    Prevents application crashes and returns standardized JSON error responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            request_id = uuid.uuid4().hex[:8]
            logger.error(
                f"💥 [{request_id}] Unhandled error in {request.method} {request.url.path}: {e}",
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "request_id": request_id,
                },
                headers={
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "X-Request-ID": request_id,
                },
            )


def setup_middleware(
    app,
    enable_rate_limiting: bool = True,
    enable_request_logging: bool = True,
    max_requests: int = 100,
    window_seconds: int = 60,
    max_concurrent: int = 50,
):
    """
    Configure and attach application middlewares in the correct Starlette execution order.
    In Starlette, the last middleware added wraps earlier ones (runs first on incoming requests).
    """
    # 1. Cache control (innermost custom middleware)
    app.add_middleware(CacheControlMiddleware)
    logger.info(" Cache control middleware enabled")

    # 2. Rate limiting (protects downstream application routes)
    if enable_rate_limiting:
        app.add_middleware(
            RateLimitingMiddleware,
            max_requests=max_requests,
            window_seconds=window_seconds,
            max_concurrent=max_concurrent,
        )
        logger.info(
            f" Rate limiting middleware enabled (limit={max_requests}/{window_seconds}s, max_concurrent={max_concurrent})"
        )

    # 3. Error handling (catches unhandled exceptions from inner layers)
    app.add_middleware(ErrorHandlingMiddleware)
    logger.info(" Error handling middleware enabled")

    # 4. Request logging (logs status and measures duration for all responses including errors)
    if enable_request_logging:
        app.add_middleware(RequestLoggingMiddleware)
        logger.info(" Request logging middleware enabled")

    # 5. Security headers (outermost custom middleware, guarantees security headers on all responses)
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info(" Security headers middleware enabled")


__all__ = [
    "StreamingSafeGZipMiddleware",
    "SecurityHeadersMiddleware",
    "CacheControlMiddleware",
    "RateLimitingMiddleware",
    "RequestLoggingMiddleware",
    "ErrorHandlingMiddleware",
    "setup_middleware",
]
