"""
Base service for handling batch processing with validation and concurrent operations.
Provides common functionality for file upload, URL processing, and document management.
"""

# Standard library imports
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

# Third-party imports
from fastapi import HTTPException

# Local imports
from config.settings import Config
from .document_service import DocumentService
from models.responses import (
    StatusEnum,
    MultipleFileUploadResponse,
    FileProcessResult,
    ErrorResponse,
)
from models.metadata import normalize_legacy_metadata, SourceType

logger = logging.getLogger(__name__)


class BaseProcessingService(ABC):
    """
    Abstract base service for handling batch processing with validation and concurrent operations.
    Provides common functionality for file uploads, URL processing, and document management.
    """

    def __init__(self):
        # Initialize document service for processing capabilities
        self.document_service = DocumentService()

    async def process_items(
        self, items: List[Union[str, Any]], item_type: str = "items"
    ) -> MultipleFileUploadResponse:
        """
        Process multiple items with validation and batch processing constraints.

        Args:
            items: List of items to process (files, URLs, etc.)
            item_type: Type of items for error messages and validation

        Returns:
            MultipleFileUploadResponse with comprehensive processing results
        """
        try:
            # Validate batch size
            if len(items) > Config.File.MAX_FILES_PER_BATCH():
                raise HTTPException(
                    status_code=400,
                    detail=f"Too many {item_type}. Maximum {Config.File.MAX_FILES_PER_BATCH()} {item_type} per batch.",
                )

            # Check if any items are provided
            if not items:
                raise HTTPException(status_code=400, detail=f"No {item_type} provided")

            # Pre-validate items and prepare them for processing
            valid_items = await self._validate_and_prepare_items(items)

            if not valid_items:
                raise HTTPException(
                    status_code=400, detail=f"No valid {item_type} found"
                )

            # Process items concurrently for improved performance
            results = await self._process_items_concurrently(valid_items)

            # Generate comprehensive response with statistics
            return self._generate_upload_response(results, len(valid_items), item_type)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Multiple {item_type} processing error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error processing {item_type}: {str(e)}"
            )

    @abstractmethod
    async def _validate_and_prepare_items(self, items: List[Any]) -> List[Any]:
        """
        Validate items and prepare them for processing.
        Must be implemented by subclasses for specific item types.
        """
        pass

    @abstractmethod
    async def _process_items_concurrently(
        self, valid_items: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Process items concurrently and return results.
        Must be implemented by subclasses for specific processing logic.
        """
        pass

    def _convert_batch_results_to_file_results(
        self,
        batch_result: Dict[str, Any],
        valid_items: List[Any],
        item_type: str = "item",
    ) -> List[Dict[str, Any]]:
        """
        Convert batch processing results to individual file results for consistent response format.
        Handles different item types (files, URLs) with appropriate metadata extraction.
        """
        results = []

        for i, item_result in enumerate(batch_result["results"]):
            try:
                # Get file size based on item type for accurate reporting
                file_size = self._get_item_size(valid_items[i], item_type)

                # Extract and normalize metadata
                raw_metadata = item_result.get("metadata", {}) if item_result["success"] else {}
                
                # Normalize metadata to ensure consistent structure
                if raw_metadata and item_result["success"]:
                    try:
                        normalized_metadata = normalize_legacy_metadata(
                            raw_metadata,
                            source_id=raw_metadata.get("source_id", item_result["filename"]),
                            source_name=item_result["filename"],
                            source_type=SourceType.FILE
                        )
                        metadata_dict = normalized_metadata.dict()
                    except Exception as e:
                        logger.warning(f"Failed to normalize metadata for {item_result['filename']}: {e}")
                        metadata_dict = raw_metadata
                else:
                    metadata_dict = raw_metadata
                
                results.append(
                    {
                        "success": item_result["success"],
                        "result": FileProcessResult(
                            filename=item_result["filename"],
                            file_size=file_size,
                            document_count=item_result["document_count"],
                            source_id=metadata_dict.get("source_id", "unknown"),
                            status=(
                                StatusEnum.SUCCESS
                                if item_result["success"]
                                else StatusEnum.ERROR
                            ),
                            error_message=(
                                item_result.get("error")
                                if not item_result["success"]
                                else None
                            ),
                            # Use normalized metadata fields
                            file_type=metadata_dict.get("file_extension", "unknown"),
                            processing_method=metadata_dict.get("processing_method", "unknown"),
                            processing_time=f"{metadata_dict.get('processing_time_seconds', 0):.2f}s" if metadata_dict.get("processing_time_seconds") else None,
                            # Optional debug info (only if needed)
                            debug_info=(
                                {
                                    "chunks_directory": metadata_dict.get("chunks_directory"),
                                    "ocr_enabled": metadata_dict.get("ocr_enabled"),
                                    "ocr_used": metadata_dict.get("ocr_used"),
                                    "conversion_success": metadata_dict.get("conversion_success"),
                                    "processing_status": metadata_dict.get("processing_status"),
                                    "total_chunks": metadata_dict.get("total_chunks"),
                                }
                                if item_result["success"] and metadata_dict
                                else None
                            ),
                        ),
                    }
                )

            except Exception as e:
                # Get item name for error case with fallback handling
                item_name = self._get_item_name(
                    valid_items[i] if i < len(valid_items) else None, i, item_type
                )

                logger.warning(f"Error processing {item_type} {item_name}: {str(e)}")
                results.append(
                    {
                        "success": False,
                        "result": FileProcessResult(
                            filename=item_name,
                            file_size=0,
                            document_count=0,
                            source_id="",
                            status=StatusEnum.ERROR,
                            error_message=str(e),
                            file_type="unknown",
                            processing_method="unknown",
                            processing_time=None,
                            debug_info=None,
                        ),
                    }
                )

        return results

    def _create_error_response(
        self, error_msg: str, error_code: str = "PROCESSING_ERROR", details: dict = None
    ) -> ErrorResponse:
        """Create standardized error response using ErrorResponse model."""
        return ErrorResponse(
            status=StatusEnum.ERROR,
            message=error_msg,
            error_code=error_code,
            details=details or {},
        )

    def _get_item_size(self, item: Any, item_type: str) -> int:
        """
        Get size of an item based on its type for accurate reporting.
        Handles files, URLs, and other item types appropriately.
        """
        if item_type == "files":
            # For files, item is (file, file_content) tuple
            _, file_content = item
            return len(file_content)
        elif item_type == "URLs":
            # For URLs, use a reasonable estimate since we don't have actual content size
            return len(str(item)) if item else 0
        else:
            return 0

    def _get_item_name(self, item: Any, index: int, item_type: str) -> str:
        """
        Get name of an item based on its type for error reporting and identification.
        Provides fallback names for error cases.
        """
        if item_type == "files" and item:
            file, _ = item
            return file.filename if file else f"file_{index}"
        elif item_type == "URLs" and item:
            return str(item)
        else:
            return f"{item_type.lower()}_{index}"

    def _generate_upload_response(
        self,
        processing_results: List[Dict[str, Any]],
        total_items: int,
        item_type: str = "items",
    ) -> MultipleFileUploadResponse:
        """
        Generate the final upload response with comprehensive statistics and status.
        Determines overall success/failure status based on individual results.
        """
        successful_count = 0
        failed_count = 0
        total_documents = 0
        results = []

        # Aggregate results and calculate statistics
        for proc_result in processing_results:
            results.append(proc_result["result"])
            if proc_result["success"]:
                successful_count += 1
                total_documents += proc_result["result"].document_count
            else:
                failed_count += 1

        # Determine overall status based on success/failure ratios
        if successful_count == 0:
            overall_status = StatusEnum.ERROR
            message = f"All {total_items} {item_type} failed to process"
        elif failed_count == 0:
            overall_status = StatusEnum.SUCCESS
            message = f"Successfully processed all {successful_count} {item_type}"
        else:
            overall_status = StatusEnum.WARNING
            message = f"Processed {total_items} {item_type}: {successful_count} successful, {failed_count} failed"

        return MultipleFileUploadResponse(
            status=overall_status,
            message=message,
            total_files=total_items,
            successful_files=successful_count,
            failed_files=failed_count,
            total_documents=total_documents,
            results=results,
        )
