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

def get_safe_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """
    Get a safe event loop that handles Windows-specific issues.
    Returns None if no event loop is running.
    """
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        # No event loop running
        return None
    except Exception as e:
        logger.warning(f" Error getting event loop: {e}")
        return None

def create_safe_task(coro, loop: Optional[asyncio.AbstractEventLoop] = None):
    """
    Create a task safely with proper error handling for Windows.
    """
    if loop is None:
        loop = get_safe_event_loop()
    
    if loop is None:
        logger.warning(" No event loop available, running coroutine directly")
        try:
            return asyncio.run(coro)
        except Exception as e:
            logger.error(f" Error running coroutine directly: {e}")
            return None
    
    try:
        task = loop.create_task(coro)
        # Add error handling callback
        task.add_done_callback(lambda t: logger.error(f" Task failed: {t.exception()}") if t.exception() else None)
        return task
    except Exception as e:
        logger.error(f" Error creating task: {e}")
        return None

def safe_close_session(session):
    """
    Safely close a requests session to prevent socket leaks.
    """
    if session is not None:
        try:
            session.close()
            logger.debug(" Session closed successfully")
        except Exception as e:
            logger.warning(f" Error closing session: {e}")

def is_session_closed(session):
    """
    Check if a requests session is closed.
    """
    if session is None:
        return True
    try:
        # Check if the session has a closed attribute
        if hasattr(session, 'closed'):
            return session.closed
        # If no closed attribute, check if the adapter pool is closed
        if hasattr(session, 'adapters'):
            for adapter in session.adapters.values():
                if hasattr(adapter, 'poolmanager') and hasattr(adapter.poolmanager, 'closed'):
                    return adapter.poolmanager.closed
        return False
    except Exception:
        return False

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
