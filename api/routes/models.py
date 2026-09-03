"""
Model status endpoints for monitoring preloaded ML models.
"""

import time
from fastapi import APIRouter
from utils.performance import get_model_preloader

router = APIRouter()


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
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }
