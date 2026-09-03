"""
File upload service for handling file uploads and processing.
Provides validation, batch processing, and error handling for file uploads.
"""

# Standard library imports
import logging
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
from fastapi import UploadFile, HTTPException

# Local imports
from config.settings import Config
from models.responses import MultipleFileUploadResponse
from utils.text_processing import ValidationUtils
from .base_service import BaseProcessingService

logger = logging.getLogger(__name__)


class UploadService(BaseProcessingService):
    """
    Service for handling file uploads and processing with validation.
    Extends base processing service with file-specific validation and batch processing.
    """

    async def process_file_uploads(
        self, files: List[UploadFile]
    ) -> MultipleFileUploadResponse:
        """
        Process multiple file uploads with validation and batch processing.
        Validates files, processes them in batches, and returns comprehensive results.

        Args:
            files: List of uploaded files from FastAPI

        Returns:
            MultipleFileUploadResponse with processing results and metadata
        """
        return await self.process_items(files, "files")

    async def _validate_and_prepare_items(self, files: List[UploadFile]) -> List[tuple]:
        """
        Validate files and prepare them for processing with size and type checks.
        Uses concurrent file reading for better performance.
        """
        total_size = 0
        valid_files = []

        # Process files concurrently for better performance
        async def process_single_file(file: UploadFile) -> tuple | None:
            if not file.filename:
                return None  # Skip files without names

            # Check if file type is supported by document service
            if not self.document_service.is_supported_file(file.filename):
                logger.warning(f"Unsupported file type: {file.filename}")
                return None

            # Read file content for size validation
            try:
                file_content = await file.read()
                if not file_content:
                    return None  # Skip empty files

                # Reset file position for potential re-reading
                await file.seek(0)
                return (file, file_content)
            except Exception as e:
                logger.error(f"Error reading file {file.filename}: {e}")
                return None

        # Process all files concurrently
        tasks = [process_single_file(file) for file in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect valid results and calculate total size
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing file: {result}")
                continue
            if result is not None:
                try:
                    file, file_content = result
                    total_size += len(file_content)
                    valid_files.append((file, file_content))
                except (ValueError, TypeError) as e:
                    logger.error(f"Error unpacking file result: {e}")
                    continue

        # Check total batch size against configured limits
        if total_size > Config.File.MAX_TOTAL_BATCH_SIZE():
            raise HTTPException(
                status_code=400,
                detail=f"Total batch size too large. Maximum {Config.File.MAX_TOTAL_BATCH_SIZE() / (1024*1024):.1f}MB allowed.",
            )

        return valid_files

    async def _process_items_concurrently(
        self, valid_files: List[tuple]
    ) -> List[Dict[str, Any]]:
        """
        Process files using batch processing for optimal performance.
        Uses document service batch processing to handle multiple files efficiently.
        """
        # Prepare file data for batch processing
        file_data_list = [
            (file_content, file.filename) for file, file_content in valid_files
        ]

        # Use batch processing instead of individual processing for efficiency
        batch_result = await self.document_service.process_multiple_documents(
            file_data_list, rebuild_at_end=True  # Rebuild vector store once at the end
        )

        # Convert batch results to individual file results using base service method
        return self._convert_batch_results_to_file_results(
            batch_result, valid_files, "files"
        )
