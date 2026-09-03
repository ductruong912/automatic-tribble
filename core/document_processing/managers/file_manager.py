"""
File manager for handling document chunk storage and file operations.
Provides safe file management with chunk organization and cleanup capabilities.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import List

# Third-party imports
import aiofiles

# Local imports
from config.settings import Config
from utils.file_operations.file_manager import FileManager as FileUtils
from utils.text_processing.text_utils import TextUtils

logger = logging.getLogger(__name__)


class FileManager:
    """
    Manages file storage and organization for document processing.
    Handles chunk storage, file cleanup, and safe file operations.
    """

    def __init__(self):
        """
        Initialize file manager without upload directory dependency.
        Files are processed directly in memory for better performance.
        """
        # No upload directory needed - we process files directly

    async def save_chunks_to_files(self, documents: List[str], filename: str) -> str:
        """
        Save document chunks to individual files for persistence and debugging.
        Creates organized directory structure for chunk storage.

        Args:
            documents: List of document chunks to save
            filename: Original filename for directory naming

        Returns:
            Path to the created chunks directory
        """
        # Create chunks directory for this file with safe naming
        safe_filename = FileUtils.sanitize_filename(
            Path(filename).stem
        )  # Remove extension and sanitize
        chunks_dir = Path(Config.File.CHUNKS_DIR()) / safe_filename
        FileUtils.ensure_directory_exists(str(chunks_dir))

        # Save each chunk as a separate file for easy access
        chunk_idx = 1
        for i, chunk in enumerate(documents):
            cleaned_chunk = TextUtils.clean_chunk_text(chunk)
            if not cleaned_chunk.strip():
                continue  # Skip empty cleaned chunks
            chunk_filename = chunks_dir / f"chunk_{chunk_idx:03d}.txt"
            async with aiofiles.open(chunk_filename, "w", encoding="utf-8") as f:
                await f.write(cleaned_chunk)
            chunk_idx += 1

        logger.info(f"💾 Saved {len(documents)} chunks to {chunks_dir}")
        return str(chunks_dir)

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file safely using consolidated FileUtils.
        Provides error handling and logging for file deletion operations.
        """
        return FileUtils.safe_delete_file(file_path)

    def delete_directory(self, dir_path: str) -> bool:
        """
        Delete a directory and all its contents safely using consolidated FileUtils.
        Provides error handling and logging for directory deletion operations.
        """
        return FileUtils.safe_delete_directory(dir_path)
