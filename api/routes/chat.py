"""
Chat endpoint - RAG-powered conversation API with history support.
"""

# Standard library imports
import json
import logging
from typing import Any, Dict, Optional

# Third-party imports
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse

# Local imports
from api.dependencies import get_chat_service
from services import ChatService
from services.chat_service import ChatUnavailableError
from models.responses import ChatRequest

router = APIRouter()
logger = logging.getLogger(__name__)

# Message sent to clients on failure. The real cause is logged server-side only.
GENERIC_STREAM_ERROR = "Unable to generate a response. Please try again."


def _sse_frame(payload: Dict[str, Any], event: Optional[str] = None) -> str:
    """
    Format one Server-Sent Events frame.

    The payload is JSON-encoded, so newlines inside a token travel as the two characters
    \\n instead of ending the frame - a raw token stream loses every line break, which is
    what broke tables and number formatting in answers.
    """
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/")
async def chat(
    request: ChatRequest = Body(...),
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Chat endpoint with configurable mode (query-only or with history).

    Streams the answer as Server-Sent Events: `{"type": "token", "text": ...}` frames
    followed by `{"type": "done"}`, or an `error` event if generation fails. Anything
    that can be checked before the first token is answered with an HTTP status instead,
    since the status is already sent once streaming starts.
    """
    try:
        chat_service.ensure_can_stream(request.query)
    except ChatUnavailableError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    async def event_stream():
        try:
            async for token in chat_service.stream_chat_with_memory(
                request.query, custom_history=request.history
            ):
                yield _sse_frame({"type": "token", "text": token})
        except Exception:
            logger.exception("Streaming chat failed")
            yield _sse_frame(
                {"type": "error", "message": GENERIC_STREAM_ERROR}, event="error"
            )
            return
        yield _sse_frame({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Stops nginx and similar proxies from buffering the stream into one blob
            "X-Accel-Buffering": "no",
        },
    )
