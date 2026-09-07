# Standard library imports
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import os

# Local imports
from config.settings import Config
from .text_utils import TextUtils

# Windows device names, unusable as a file or directory name even with an extension
RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


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
        name_without_ext = Path(filename).stem.upper()
        if name_without_ext in RESERVED_WINDOWS_NAMES:
            return {"valid": False, "error": f"Filename '{filename}' is reserved"}

        return {"valid": True, "error": None}

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename by replacing invalid characters for safe file operations.

        Repairs the name instead of rejecting it: returning a constant for every name
        that fails validation would map unrelated files onto the same storage location.
        """
        if not filename or not filename.strip():
            return "unnamed_file"

        # Drop any directory component so the result cannot escape its parent
        sanitized = Path(filename.strip()).name

        # Replace characters Windows rejects, plus control characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            sanitized = sanitized.replace(char, "_")
        sanitized = "".join(char for char in sanitized if ord(char) >= 32)

        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        # Windows silently drops trailing dots and spaces
        sanitized = sanitized.strip("_").strip().rstrip(". ")

        # "." and ".." are directory references, not names
        if not sanitized or set(sanitized) == {"."}:
            return "unnamed_file"

        if Path(sanitized).stem.upper() in RESERVED_WINDOWS_NAMES:
            sanitized = f"file_{sanitized}"

        # Leave room for the chunk filenames stored under this name
        if len(sanitized) > 200:
            suffix = Path(sanitized).suffix[:20]
            sanitized = sanitized[: 200 - len(suffix)] + suffix

        return sanitized

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
