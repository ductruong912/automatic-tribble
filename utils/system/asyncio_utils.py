"""
Windows-specific asyncio utilities to prevent socket and event loop issues.
Handles ProactorEventLoop problems that cause WinError 10038.
"""

import asyncio
import logging
import sys
import warnings
from typing import Optional

logger = logging.getLogger(__name__)

def setup_windows_asyncio():
    """
    Setup Windows-specific asyncio configuration to prevent socket errors.
    This fixes WinError 10038, 10054 and other Windows asyncio issues.
    """
    if sys.platform == "win32":
        try:
            # Force set the event loop policy to use SelectorEventLoop on Windows
            # This prevents ProactorEventLoop socket issues
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            logger.info(" Windows asyncio policy set to SelectorEventLoop")
            
            # Also set the default event loop to ensure consistency
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                logger.info(" Default event loop set to SelectorEventLoop")
            except Exception as e:
                logger.warning(f" Could not set default event loop: {e}")
            
            # Set up error handling for asyncio
            setup_asyncio_error_handling()
                
        except Exception as e:
            logger.warning(f" Could not set Windows asyncio policy: {e}")
            # Fallback: try to suppress the specific warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="asyncio")
            warnings.filterwarnings("ignore", message=".*ProactorBasePipeTransport.*")
    else:
        logger.debug("Non-Windows platform, skipping Windows asyncio setup")

def setup_asyncio_error_handling():
    """
    Setup error handling for asyncio to prevent connection errors from crashing the app.
    """
    def handle_asyncio_exception(loop, context):
        """Handle asyncio exceptions gracefully."""
        exception = context.get('exception')
        if exception:
            # Suppress specific Windows connection errors
            if isinstance(exception, (ConnectionResetError, OSError)):
                if any(error_code in str(exception) for error_code in ['10054', '10038', '64']):
                    logger.debug(f"Suppressed Windows connection error: {exception}")
                    return  # Don't log these as errors
            
            # Log other exceptions normally
            logger.error(f"Asyncio exception: {exception}")
        else:
            logger.error(f"Asyncio context error: {context}")
    
    try:
        # Set the exception handler for the current event loop
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(handle_asyncio_exception)
        logger.info(" Asyncio error handling configured")
    except Exception as e:
        logger.warning(f" Could not set asyncio error handler: {e}")


def setup_asyncio_logging():
    """
    Setup asyncio logging to suppress Windows-specific warnings.
    """
    # Suppress specific asyncio warnings that are common on Windows
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Suppress ProactorBasePipeTransport warnings and connection errors
    warnings.filterwarnings("ignore", message=".*ProactorBasePipeTransport.*")
    warnings.filterwarnings("ignore", message=".*_call_connection_lost.*")
    warnings.filterwarnings("ignore", message=".*ConnectionResetError.*")
    warnings.filterwarnings("ignore", message=".*WinError 10054.*")
    warnings.filterwarnings("ignore", message=".*WinError 10038.*")
    warnings.filterwarnings("ignore", message=".*An operation was attempted on something that is not a socket.*")
    warnings.filterwarnings("ignore", message=".*An existing connection was forcibly closed by the remote host.*")
    
    # Suppress asyncio deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="asyncio")
    
    logger.info(" Asyncio logging configured for Windows compatibility")
