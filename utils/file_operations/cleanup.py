"""
Cleanup utilities for managing temporary files and data directories.
Provides comprehensive cleanup functions for logs and data folders.
"""

# Standard library imports
import logging
from pathlib import Path

# Local imports
from config.settings import Config
from .file_manager import FileManager

# Create aliases for backward compatibility
FileUtils = FileManager
LoggingConfig = Config.Logging

logger = logging.getLogger(__name__)




def cleanup_data_folders():
    """
    Clean up all data folders when server stops to prevent disk space accumulation.
    Removes temporary files, chunks, vectors, and logs while preserving directory structure.
    """
    logger.info("🧹 Cleaning up data folders...")


    # List of directories to clean up for comprehensive maintenance
    cleanup_dirs = [
        Config.Database.CHUNKS_DIR(),
        Config.Database.VECTORS_DIR(),
        Config.Database.TEMP_DIR(),
        Config.Logging.LOG_DIR(),
    ]

    cleaned_count = 0
    for dir_path in cleanup_dirs:
        try:
            if FileManager.directory_exists(dir_path):
                # Use FileManager for consistent cleanup operations
                count = FileManager.clean_directory_contents(
                    dir_path, keep_directory=True
                )
                cleaned_count += count
                logger.info(f" Cleaned: {dir_path} ({count} items)")
            else:
                logger.warning(f" Directory not found: {dir_path}")
        except Exception as e:
            logger.error(f" Error cleaning {dir_path}: {e}")

    logger.info(
        f"🧹 Cleanup completed! Removed {cleaned_count} items from data folders."
    )
