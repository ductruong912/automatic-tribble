"""
Custom middleware for the FastAPI application.
Contains StreamingSafeGZipMiddleware for SSE streaming compatibility.
"""

from starlette.middleware.gzip import GZipMiddleware


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
