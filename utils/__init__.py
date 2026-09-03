"""
Utilities package for shared helper functions and common operations.
Organized into domain-specific subpackages for better maintainability.
"""

# Import from subpackages for backward compatibility
from .file_operations import FileManager, cleanup_data_folders
from .text_processing import TextUtils, ValidationUtils
from .performance import (
    PerformanceMonitor, get_performance_monitor, log_performance_summary,
    BackgroundTaskManager, start_background_tasks, stop_background_tasks, get_background_manager,
    ModelPreloader, get_model_preloader, preload_all_models
)
from .system import (
    setup_windows_asyncio, setup_asyncio_logging,
    configure_uvicorn_for_windows, get_uvicorn_config, get_uvicorn_ssl_config,
    LogManager
)

# Legacy aliases for backward compatibility
FileUtils = FileManager

# Export all utility classes and functions for convenient access
__all__ = [
    # File operations
    "FileManager", "FileUtils",  # File and directory operations
    "cleanup_data_folders",  # Data directory cleanup
    
    # Text processing
    "TextUtils",  # Text processing and manipulation
    "ValidationUtils",  # Input validation and sanitization
    
    # Performance
    "PerformanceMonitor", "get_performance_monitor", "log_performance_summary",
    "BackgroundTaskManager", "start_background_tasks", "stop_background_tasks", "get_background_manager",
    "ModelPreloader", "get_model_preloader", "preload_all_models",
    
    # System
    "setup_windows_asyncio", "setup_asyncio_logging",
    "configure_uvicorn_for_windows", "get_uvicorn_config", "get_uvicorn_ssl_config",
    "LogManager"
]
