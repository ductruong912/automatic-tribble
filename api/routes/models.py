"""
Model status endpoints for monitoring preloaded ML models.
"""

import time
import logging
from fastapi import APIRouter
from utils.performance import get_model_preloader

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models")
async def get_model_status():
    """
    Get status of preloaded ML models.
    """
    try:
        preloader = get_model_preloader()
        status = preloader.get_status()
        
        return {
            "status": "success",
            "model_status": status,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error retrieving model status: {e}", exc_info=True)
        return {
            "status": "error",
            "error": "Failed to retrieve model status.",
            "timestamp": time.time()
        }

