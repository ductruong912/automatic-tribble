# Standard library imports
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import os

# Local imports
from config.settings import Config
from .text_utils import TextUtils


class ValidationUtils:
    """Utility functions for input validation."""

    @staticmethod
    def validate_file_size(
        file_size: int, max_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate file size against maximum allowed size."""
        max_size = max_size or Config.File.MAX_FILE_SIZE

        if file_size > max_size:
            return {
                "valid": False,
                "error": f"File too large. Maximum {max_size / (1024*1024):.1f}MB allowed.",
            }

        return {"valid": True, "error": None}

    @staticmethod
    def validate_file_extension(
        filename: str, allowed_extensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate file extension against allowed extensions."""
        allowed_extensions = allowed_extensions or Config.File.ALLOWED_EXTENSIONS

        if not filename:
            return {"valid": False, "error": "No filename provided"}

        file_ext = Path(filename).suffix.lower()

        if file_ext not in allowed_extensions:
            return {
                "valid": False,
                "error": f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}",
            }

        return {"valid": True, "error": None}

    @staticmethod
    def validate_batch_size(file_count: int, total_size: int) -> Dict[str, Any]:
        """Validate batch constraints."""
        # Check file count
        if file_count > Config.File.MAX_FILES_PER_BATCH:
            return {
                "valid": False,
                "error": f"Too many files. Maximum {Config.File.MAX_FILES_PER_BATCH} files per batch.",
            }

        # Check total size
        if total_size > Config.File.MAX_TOTAL_BATCH_SIZE:
            return {
                "valid": False,
                "error": f"Total batch size too large. Maximum {Config.File.MAX_TOTAL_BATCH_SIZE / (1024*1024):.1f}MB allowed.",
            }

        return {"valid": True, "error": None}

    @staticmethod
    def validate_url(url: str) -> Dict[str, Any]:
        """Validate URL format and accessibility."""
        if not url or not url.strip():
            return {"valid": False, "error": "Empty URL provided"}

        url = url.strip()

        # Basic URL format validation
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return {"valid": False, "error": "Invalid URL format"}

            if parsed.scheme not in ["http", "https"]:
                return {"valid": False, "error": "URL must use HTTP or HTTPS protocol"}

        except Exception as e:
            return {"valid": False, "error": f"Invalid URL: {str(e)}"}

        return {"valid": True, "error": None}

    @staticmethod
    def validate_query(
        query: str, min_length: int = 1, max_length: int = 2000
    ) -> Dict[str, Any]:
        """Validate user query."""
        if not query:
            return {"valid": False, "error": "Query cannot be empty"}

        query = query.strip()

        if len(query) < min_length:
            return {
                "valid": False,
                "error": f"Query too short. Minimum {min_length} characters.",
            }

        if len(query) > max_length:
            return {
                "valid": False,
                "error": f"Query too long. Maximum {max_length} characters.",
            }

        return {"valid": True, "error": None}

    @staticmethod
    def validate_filename(filename: str) -> Dict[str, Any]:
        """Validate filename for safety and compatibility."""
        if not filename or not filename.strip():
            return {"valid": False, "error": "Filename cannot be empty"}

        filename = filename.strip()

        # Check for invalid characters
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            return {
                "valid": False,
                "error": f"Filename contains invalid characters: {invalid_chars}",
            }

        # Check length
        if len(filename) > 255:
            return {"valid": False, "error": "Filename too long (max 255 characters)"}

        # Check for reserved names (Windows)
        reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }

        name_without_ext = Path(filename).stem.upper()
        if name_without_ext in reserved_names:
            return {"valid": False, "error": f"Filename '{filename}' is reserved"}

        return {"valid": True, "error": None}

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename by removing invalid characters for safe file operations.
        Replaces problematic characters with underscores for cross-platform compatibility.
        """
        if not filename:
            return "unnamed_file"
        validation = ValidationUtils.validate_filename(filename)
        if not validation["valid"]:
            return "unnamed_file"
        # Replace invalid characters with underscores for safety
        invalid_chars = '<>:"/\\|?*'
        sanitized = filename
        for char in invalid_chars:
            sanitized = sanitized.replace(char, "_")
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        sanitized = sanitized.strip("_").strip()
        return sanitized if sanitized else "unnamed_file"

    @staticmethod
    def validate_email(email: str) -> Dict[str, Any]:
        """Validate email format (basic validation)."""
        if not email or not email.strip():
            return {"valid": False, "error": "Email cannot be empty"}

        email = email.strip()

        # Basic email pattern
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        if not re.match(email_pattern, email):
            return {"valid": False, "error": "Invalid email format"}

        return {"valid": True, "error": None}

    @staticmethod
    def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
        """Sanitize user input by removing potentially harmful content."""
        if not text:
            return ""

        # Remove control characters except common whitespace
        sanitized = "".join(
            char for char in text if ord(char) >= 32 or char in "\n\r\t"
        )

        # Normalize whitespace using consolidated TextUtils
        sanitized = TextUtils.normalize_whitespace(sanitized)

        # Truncate if needed using consolidated TextUtils
        if max_length:
            sanitized = TextUtils.truncate_text(sanitized, max_length, suffix="")

        return sanitized
