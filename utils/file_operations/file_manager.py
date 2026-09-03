"""
File system utilities for safe file and directory operations.
Provides robust file handling with error handling, validation, and cleanup capabilities.
"""

# Standard library imports
import logging
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class FileManager:
    """
    Utility functions for safe file and directory operations.
    Provides robust file handling with error handling and validation.
    """

    @staticmethod
    def ensure_directory_exists(directory_path: str) -> Path:
        """
        Create directory if it doesn't exist, ensuring parent directories are created.
        Returns the Path object for the created/existing directory.
        """
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def directory_exists(directory_path: str) -> bool:
        """
        Check if a directory exists and is actually a directory.
        Provides safe directory existence validation.
        """
        return Path(directory_path).exists() and Path(directory_path).is_dir()

    @staticmethod
    def safe_delete_file(
        file_path: str, retry_count: int = 3, retry_delay: float = 0.1
    ) -> bool:
        """
        Safely delete a file with error handling, retry logic, and logging.
        Handles Windows-specific file locking issues.
        Returns True if deletion was successful or file didn't exist.
        """
        import time
        import os

        try:
            path = Path(file_path)
            if not path.exists():
                return True  # File doesn't exist, consider it "deleted"

            if not path.is_file():
                logger.warning(f"Path exists but is not a file: {file_path}")
                return False

            # Try to delete with retries for Windows file locking issues
            for attempt in range(retry_count):
                try:
                    path.unlink()
                    logger.debug(f"Successfully deleted file: {file_path}")
                    return True
                except PermissionError as e:
                    if (
                        "being used by another process" in str(e)
                        and attempt < retry_count - 1
                    ):
                        logger.debug(
                            f"File {file_path} is in use, retrying in {retry_delay}s (attempt {attempt + 1}/{retry_count})"
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.warning(
                            f"Permission error deleting file {file_path}: {e}"
                        )
                        return False
                except OSError as e:
                    if attempt < retry_count - 1:
                        logger.debug(
                            f"OS error deleting file {file_path}, retrying in {retry_delay}s (attempt {attempt + 1}/{retry_count}): {e}"
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        logger.error(f"Error deleting file {file_path}: {e}")
                        return False

            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting file {file_path}: {e}")
            return False

    @staticmethod
    def safe_delete_directory(directory_path: str) -> bool:
        """
        Safely delete a directory and all its contents recursively.
        Handles errors gracefully and logs any issues encountered.
        """
        try:
            path = Path(directory_path)
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting directory {directory_path}: {e}")
            return False

    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """
        Get file size in megabytes for size validation and monitoring.
        Returns 0.0 if file doesn't exist or error occurs.
        """
        try:
            size_bytes = Path(file_path).stat().st_size
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0

    @staticmethod
    def get_file_extension(filename: str) -> str:
        """
        Get file extension in lowercase for format validation.
        Useful for determining file type and processing requirements.
        """
        return Path(filename).suffix.lower()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename by removing invalid characters for safe file operations.
        Uses ValidationUtils for validation and sanitization.
        """
        from utils.text_processing.validation import ValidationUtils
        return ValidationUtils.sanitize_filename(filename)

    @staticmethod
    def list_files_in_directory(
        directory_path: str, extensions: Optional[List[str]] = None
    ) -> List[str]:
        """
        List files in a directory with optional extension filtering.
        Returns sorted list of file paths matching the specified extensions.
        """
        try:
            path = Path(directory_path)
            if not path.exists() or not path.is_dir():
                return []

            files = []
            for file_path in path.iterdir():
                if file_path.is_file():
                    if extensions is None or file_path.suffix.lower() in extensions:
                        files.append(str(file_path))

            return sorted(files)
        except Exception as e:
            logger.error(f"Error listing files in {directory_path}: {e}")
            return []

    @staticmethod
    def clean_directory_contents(
        directory_path: str, keep_directory: bool = True
    ) -> int:
        """
        Clean all contents of a directory while optionally preserving the directory structure.
        Returns the number of items successfully cleaned.
        """
        try:
            path = Path(directory_path)
            if not path.exists():
                return 0
            cleaned_count = 0
            from utils.file_operations.file_manager import FileManager as FileUtils

            for item in path.iterdir():
                try:
                    if item.is_file():
                        if FileUtils.safe_delete_file(str(item)):
                            cleaned_count += 1
                    elif item.is_dir():
                        if FileUtils.safe_delete_directory(str(item)):
                            cleaned_count += 1
                except Exception as e:
                    logger.error(f"Error cleaning {item}: {e}")
            return cleaned_count
        except Exception as e:
            logger.error(f"Error cleaning directory {directory_path}: {e}")
            return 0
