"""
Health check, performance and cache endpoints for monitoring and diagnostics.
"""

# Standard library imports
import time
import logging

# Third-party imports
from fastapi import APIRouter, Depends

# Local imports
from models.responses import HealthResponse, StatusEnum
from utils.performance import get_performance_monitor
from api.dependencies import get_chat_service

router = APIRouter()
logger = logging.getLogger(__name__)


# Track server start time for uptime calculation
_start_time = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint for load balancers and monitoring.
    Returns server status, version, and uptime information.
    """
    # Calculate uptime in human-readable format
    uptime_seconds = time.time() - _start_time
    uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"

    return HealthResponse(
        status=StatusEnum.SUCCESS,
        message="Enhanced RAG Chatbot API is running",
        version="2.0.0",
        uptime=uptime_str,
    )


@router.get("/performance")
async def performance_metrics():
    """
    Get detailed performance metrics for the chatbot.
    Returns response times, cache hit rates, and system resource usage.
    """
    try:
        monitor = get_performance_monitor()
        stats = monitor.get_performance_stats()
        health_status = monitor.get_health_status()
        
        return {
            "status": "success",
            "health_status": health_status,
            "performance_metrics": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {e}", exc_info=True)
        return {
            "status": "error",
            "error": "Failed to retrieve performance metrics.",
            "timestamp": time.time()
        }


@router.get("/cache-stats")
async def cache_statistics(chat_service=Depends(get_chat_service)):
    """
    Get detailed cache statistics for all cache layers.
    Returns information about smart cache, chatbot cache, and vector store cache.
    """
    try:
        cache_stats = chat_service.get_cache_stats()
        return {
            "status": "success",
            "cache_statistics": cache_stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error retrieving cache statistics: {e}", exc_info=True)
        return {
            "status": "error",
            "error": "Failed to retrieve cache statistics.",
            "timestamp": time.time()
        }

