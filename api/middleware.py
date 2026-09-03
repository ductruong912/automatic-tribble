"""
Custom middleware for the FastAPI application.
Provides request logging, security headers, rate limiting, and error handling.
"""

# Standard library imports
import asyncio
import time
import logging
from typing import Callable

# Third-party imports
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests and responses with timing information.
    Provides comprehensive request tracking for monitoring and debugging.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timing for performance monitoring
        start_time = time.time()

        # Log incoming request with client information
        request_id = hex(hash(request.url.path + str(time.time())))[2:10]
        logger.info(
            f"📥 [Middleware] {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'} - RequestID: {request_id}"
        )

        # Process request through the application
        response = await call_next(request)

        # Calculate processing time for performance analysis
        process_time = time.time() - start_time

        # Log response with status and timing
        logger.info(
            f"📤 [Middleware] {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s - RequestID: {request_id}"
        )

        # Add timing header for client-side monitoring
        response.headers["X-Process-Time"] = str(process_time)

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers for enhanced application security.
    Implements common security best practices for web applications.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add security headers to prevent common attacks
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Enhanced rate limiting middleware to prevent API abuse and connection overload.
    Implements sliding window rate limiting per client IP address and global concurrent request limiting.
    """

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60, max_concurrent: int = 10):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_concurrent = max_concurrent
        self.requests = {}  # {client_ip: [(timestamp, count), ...]}
        self.concurrent_requests = 0
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean old entries outside the time window
        if client_ip in self.requests:
            self.requests[client_ip] = [
                (timestamp, count)
                for timestamp, count in self.requests[client_ip]
                if current_time - timestamp < self.window_seconds
            ]

        # Initialize request tracking for new clients
        if client_ip not in self.requests:
            self.requests[client_ip] = []

        # Count current requests within the time window
        current_requests = sum(count for _, count in self.requests[client_ip])

        # Check global concurrent request limit first
        if self.concurrent_requests >= self.max_concurrent:
            logger.warning(f"🚫 Global concurrent request limit exceeded ({self.concurrent_requests}/{self.max_concurrent})")
            return Response(
                content="Server busy, too many concurrent requests. Please try again later.",
                status_code=503,
                headers={"Retry-After": "5"},
            )

        # Check rate limit and block if exceeded
        if current_requests >= self.max_requests:
            logger.warning(f"🚫 Rate limit exceeded for {client_ip}")
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(self.window_seconds)},
            )

        # Add current request to tracking
        self.requests[client_ip].append((current_time, 1))

        # Acquire semaphore for concurrent request limiting
        async with self.semaphore:
            self.concurrent_requests += 1
            try:
                # Process request normally
                response = await call_next(request)
                
                # Add rate limit headers for client information
                response.headers["X-RateLimit-Limit"] = str(self.max_requests)
                response.headers["X-RateLimit-Remaining"] = str(
                    self.max_requests - current_requests - 1
                )
                response.headers["X-RateLimit-Reset"] = str(
                    int(current_time + self.window_seconds)
                )
                response.headers["X-Concurrent-Requests"] = str(self.concurrent_requests)
                response.headers["X-Max-Concurrent-Requests"] = str(self.max_concurrent)
                
                return response
            finally:
                self.concurrent_requests -= 1


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle and log unhandled errors gracefully.
    Prevents application crashes and provides consistent error responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            request_id = hex(hash(request.url.path + str(time.time())))[2:10]
            logger.error(
                f" [Middleware Error] Unhandled error in {request.method} {request.url.path}: {str(e)} - RequestID: {request_id}"
            )

            # Return generic error response to prevent information leakage
            return Response(
                content="Internal server error",
                status_code=500,
                headers={"Content-Type": "text/plain"},
            )


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

        # Set cache headers based on content type and path
        if request.url.path.startswith("/static/"):
            # Static content - allow caching for performance
            response.headers["Cache-Control"] = f"public, max-age={self.static_max_age}"
        elif request.url.path.startswith("/docs") or request.url.path.startswith(
            "/redoc"
        ):
            # Documentation - allow caching
            response.headers["Cache-Control"] = f"public, max-age={self.static_max_age}"
        else:
            # API endpoints - no cache to ensure fresh data
            response.headers["Cache-Control"] = (
                f"no-cache, no-store, max-age={self.api_max_age}"
            )

        return response


# Middleware configuration helper
def setup_middleware(
    app, enable_rate_limiting: bool = False, enable_request_logging: bool = True
):
    """
    Setup all middleware for the application with configurable options.
    Provides comprehensive request handling, security, and monitoring.
    """

    if enable_request_logging:
        app.add_middleware(RequestLoggingMiddleware)
        logger.info(" Request logging middleware enabled")

    app.add_middleware(SecurityHeadersMiddleware)
    logger.info(" Security headers middleware enabled")

    app.add_middleware(CacheControlMiddleware)
    logger.info(" Cache control middleware enabled")

    if enable_rate_limiting:
        app.add_middleware(RateLimitingMiddleware, max_requests=100, window_seconds=60, max_concurrent=10)
        logger.info(" Rate limiting middleware enabled")

    app.add_middleware(ErrorHandlingMiddleware)
    logger.info(" Error handling middleware enabled")
