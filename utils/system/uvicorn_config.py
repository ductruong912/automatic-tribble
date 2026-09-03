"""
Custom uvicorn configuration for Windows to prevent asyncio connection errors.
"""

import asyncio
import sys
import logging

logger = logging.getLogger(__name__)

def configure_uvicorn_for_windows():
    """
    Configure uvicorn to use SelectorEventLoop on Windows to prevent connection errors.
    This should be called before starting uvicorn.
    """
    if sys.platform == "win32":
        try:
            # Force SelectorEventLoop policy
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            logger.info(" Uvicorn configured for Windows with SelectorEventLoop")
            return True
        except Exception as e:
            logger.warning(f" Could not configure uvicorn for Windows: {e}")
            return False
    return True

def get_uvicorn_config():
    """
    Get uvicorn configuration optimized for Windows.
    """
    config = {
        "loop": "asyncio",
        "http": "httptools",
        "ws": "websockets",
    }
    
    if sys.platform == "win32":
        # Use SelectorEventLoop on Windows
        config["loop"] = "asyncio"
        logger.info(" Uvicorn config optimized for Windows")
    
    return config

def get_uvicorn_ssl_config():
    """
    Get uvicorn SSL configuration for HTTPS support.
    """
    import os
    
    ssl_config = {}
    
    # Check if SSL certificates exist
    ssl_cert_file = "./SSL/fullchain.pem"
    ssl_key_file = "./SSL/privkey_converted.pem"
    
    if os.path.exists(ssl_cert_file) and os.path.exists(ssl_key_file):
        ssl_config = {
            "ssl_certfile": ssl_cert_file,
            "ssl_keyfile": ssl_key_file,
        }
        logger.info(" SSL certificates found, HTTPS enabled")
    else:
        logger.warning(" SSL certificates not found, running in HTTP mode")
    
    return ssl_config