"""
API routes package - Central router configuration for all endpoints.
"""

# Third-party imports
from fastapi import APIRouter

# Local imports
from .chat import router as chat_router
from .files import router as files_router
from .health import router as health_router
from .cleanup import router as cleanup_router

# Create main router to combine all endpoint groups
router = APIRouter()

# Include all sub-routers with appropriate prefixes and tags for API documentation
router.include_router(health_router, tags=["Health"])
router.include_router(chat_router, prefix="/chat", tags=["Chat"])
router.include_router(files_router, prefix="/files", tags=["Files"])
router.include_router(cleanup_router, prefix="/cleanup", tags=["Maintenance"])
