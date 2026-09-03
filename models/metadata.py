"""
Normalized metadata models for consistent data structure across all processing methods.
Ensures uniform metadata reporting regardless of processing method (Docling, OCR, URL, etc.).
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ProcessingMethod(str, Enum):
    """Enumeration of supported processing methods."""
    DOCLING = "docling"
    OCR_FALLBACK = "ocr_fallback"
    EXISTING_PROCESSOR = "existing_processor"
    URL_PROCESSOR = "url_processor"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    """Enumeration of source types."""
    FILE = "file"
    URL = "url"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Enumeration of processing statuses."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    UNKNOWN = "unknown"


class NormalizedMetadata(BaseModel):
    """
    Normalized metadata structure that all processing methods must follow.
    This ensures consistent data reporting across the entire system.
    """
    
    # === CORE IDENTIFICATION ===
    source_id: str = Field(..., description="Unique identifier for the source")
    source_name: str = Field(..., description="Human-readable name of the source")
    source_type: SourceType = Field(..., description="Type of source (file, url, etc.)")
    
    # === FILE INFORMATION ===
    file_extension: str = Field(..., description="File extension (e.g., '.pdf', '.docx')")
    file_size_bytes: int = Field(..., description="Size of the original file in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")
    
    # === PROCESSING INFORMATION ===
    processing_method: ProcessingMethod = Field(..., description="Method used to process the content")
    processing_status: ProcessingStatus = Field(..., description="Status of the processing operation")
    processing_timestamp: datetime = Field(default_factory=datetime.now, description="When processing occurred")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken to process in seconds")
    
    # === CONTENT STATISTICS ===
    total_chunks: int = Field(..., description="Number of text chunks created")
    total_characters: Optional[int] = Field(None, description="Total characters in processed content")
    total_words: Optional[int] = Field(None, description="Total words in processed content")
    
    # === PROCESSING CAPABILITIES ===
    ocr_enabled: bool = Field(default=False, description="Whether OCR was enabled during processing")
    ocr_used: bool = Field(default=False, description="Whether OCR was actually used")
    ocr_time_seconds: Optional[float] = Field(None, description="Time spent on OCR processing")
    
    # === CONTENT FEATURES ===
    has_tables: bool = Field(default=False, description="Whether content contains tables")
    has_images: bool = Field(default=False, description="Whether content contains images")
    has_links: bool = Field(default=False, description="Whether content contains links")
    has_formulas: bool = Field(default=False, description="Whether content contains mathematical formulas")
    
    # === QUALITY METRICS ===
    conversion_success: bool = Field(default=True, description="Whether content conversion was successful")
    confidence_score: Optional[float] = Field(None, description="Confidence score for processing quality (0-1)")
    error_count: int = Field(default=0, description="Number of errors encountered during processing")
    
    # === SYSTEM INFORMATION ===
    processor_version: str = Field(default="1.0.0", description="Version of the processor used")
    chunk_size: int = Field(default=1000, description="Size of text chunks created")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    
    # === STORAGE INFORMATION ===
    chunks_directory: Optional[str] = Field(None, description="Directory where chunks are stored")
    vector_store_updated: bool = Field(default=False, description="Whether vector store was updated")
    
    # === EXTENSIBLE METADATA ===
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_id": "document_20240101_120000",
                "source_name": "example.pdf",
                "source_type": "file",
                "file_extension": ".pdf",
                "file_size_bytes": 1024000,
                "mime_type": "application/pdf",
                "processing_method": "docling",
                "processing_status": "success",
                "processing_timestamp": "2024-01-01T12:00:00Z",
                "processing_time_seconds": 2.5,
                "total_chunks": 15,
                "total_characters": 45000,
                "total_words": 7500,
                "ocr_enabled": True,
                "ocr_used": False,
                "ocr_time_seconds": None,
                "has_tables": True,
                "has_images": False,
                "has_links": True,
                "has_formulas": False,
                "conversion_success": True,
                "confidence_score": 0.95,
                "error_count": 0,
                "processor_version": "1.0.0",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "chunks_directory": "data/chunks/example.pdf",
                "vector_store_updated": True,
                "custom_metadata": {
                    "language": "en",
                    "author": "Unknown",
                    "creation_date": "2024-01-01"
                }
            }
        }


class MetadataBuilder:
    """
    Builder class for creating normalized metadata from various processing methods.
    Provides a consistent interface for creating metadata regardless of source.
    """
    
    def __init__(self):
        self._metadata = {}
    
    def set_source_info(self, source_id: str, source_name: str, source_type: SourceType) -> 'MetadataBuilder':
        """Set basic source information."""
        self._metadata.update({
            "source_id": source_id,
            "source_name": source_name,
            "source_type": source_type
        })
        return self
    
    def set_file_info(self, file_extension: str, file_size_bytes: int, mime_type: str = None) -> 'MetadataBuilder':
        """Set file-specific information."""
        self._metadata.update({
            "file_extension": file_extension,
            "file_size_bytes": file_size_bytes,
            "mime_type": mime_type
        })
        return self
    
    def set_processing_info(self, method: ProcessingMethod, status: ProcessingStatus, 
                          processing_time: float = None) -> 'MetadataBuilder':
        """Set processing information."""
        self._metadata.update({
            "processing_method": method,
            "processing_status": status,
            "processing_time_seconds": processing_time
        })
        return self
    
    def set_content_stats(self, total_chunks: int, total_characters: int = None, 
                         total_words: int = None) -> 'MetadataBuilder':
        """Set content statistics."""
        self._metadata.update({
            "total_chunks": total_chunks,
            "total_characters": total_characters,
            "total_words": total_words
        })
        return self
    
    def set_ocr_info(self, ocr_enabled: bool = False, ocr_used: bool = False, 
                    ocr_time: float = None) -> 'MetadataBuilder':
        """Set OCR-related information."""
        self._metadata.update({
            "ocr_enabled": ocr_enabled,
            "ocr_used": ocr_used,
            "ocr_time_seconds": ocr_time
        })
        return self
    
    def set_content_features(self, has_tables: bool = False, has_images: bool = False,
                           has_links: bool = False, has_formulas: bool = False) -> 'MetadataBuilder':
        """Set content feature flags."""
        self._metadata.update({
            "has_tables": has_tables,
            "has_images": has_images,
            "has_links": has_links,
            "has_formulas": has_formulas
        })
        return self
    
    def set_quality_metrics(self, conversion_success: bool = True, confidence_score: float = None,
                          error_count: int = 0) -> 'MetadataBuilder':
        """Set quality and error metrics."""
        self._metadata.update({
            "conversion_success": conversion_success,
            "confidence_score": confidence_score,
            "error_count": error_count
        })
        return self
    
    def set_system_info(self, processor_version: str = "1.0.0", chunk_size: int = 1000,
                       chunk_overlap: int = 200) -> 'MetadataBuilder':
        """Set system and configuration information."""
        self._metadata.update({
            "processor_version": processor_version,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        })
        return self
    
    def set_storage_info(self, chunks_directory: str = None, vector_store_updated: bool = False) -> 'MetadataBuilder':
        """Set storage-related information."""
        self._metadata.update({
            "chunks_directory": chunks_directory,
            "vector_store_updated": vector_store_updated
        })
        return self
    
    def add_custom_metadata(self, key: str, value: Any) -> 'MetadataBuilder':
        """Add custom metadata."""
        if "custom_metadata" not in self._metadata:
            self._metadata["custom_metadata"] = {}
        self._metadata["custom_metadata"][key] = value
        return self
    
    def build(self) -> NormalizedMetadata:
        """Build and return the normalized metadata object."""
        return NormalizedMetadata(**self._metadata)


def normalize_legacy_metadata(legacy_metadata: Dict[str, Any], 
                            source_id: str, 
                            source_name: str,
                            source_type: SourceType = SourceType.FILE) -> NormalizedMetadata:
    """
    Convert legacy metadata to normalized format.
    Handles backward compatibility with existing metadata structures.
    """
    builder = MetadataBuilder()
    
    # Set basic source info
    builder.set_source_info(
        source_id=source_id,
        source_name=source_name,
        source_type=source_type
    )
    
    # Map legacy fields to normalized fields
    file_ext = legacy_metadata.get("file_extension") or legacy_metadata.get("file_type", "unknown")
    if not file_ext.startswith("."):
        file_ext = f".{file_ext}" if file_ext != "unknown" else "unknown"
    
    builder.set_file_info(
        file_extension=file_ext,
        file_size_bytes=legacy_metadata.get("original_size_bytes", 0),
        mime_type=legacy_metadata.get("mime_type")
    )
    
    # Map processing method
    method_str = legacy_metadata.get("processing_method", "unknown")
    try:
        method = ProcessingMethod(method_str)
    except ValueError:
        method = ProcessingMethod.UNKNOWN
    
    # Determine processing status
    if legacy_metadata.get("conversion_success", True):
        status = ProcessingStatus.SUCCESS
    elif legacy_metadata.get("error"):
        status = ProcessingStatus.FAILED
    else:
        status = ProcessingStatus.PARTIAL_SUCCESS
    
    builder.set_processing_info(
        method=method,
        status=status,
        processing_time=legacy_metadata.get("process_time")
    )
    
    # Set content stats
    builder.set_content_stats(
        total_chunks=legacy_metadata.get("total_chunks", 0),
        total_characters=legacy_metadata.get("total_characters"),
        total_words=legacy_metadata.get("total_words")
    )
    
    # Set OCR info
    builder.set_ocr_info(
        ocr_enabled=legacy_metadata.get("ocr_enabled", False),
        ocr_used=legacy_metadata.get("ocr_used", False),
        ocr_time=legacy_metadata.get("ocr_time")
    )
    
    # Set content features
    builder.set_content_features(
        has_tables=legacy_metadata.get("has_tables", False),
        has_images=legacy_metadata.get("has_images", False),
        has_links=legacy_metadata.get("has_links", False),
        has_formulas=legacy_metadata.get("has_formulas", False)
    )
    
    # Set quality metrics
    builder.set_quality_metrics(
        conversion_success=legacy_metadata.get("conversion_success", True),
        confidence_score=legacy_metadata.get("confidence_score"),
        error_count=legacy_metadata.get("error_count", 0)
    )
    
    # Set system info
    builder.set_system_info(
        processor_version=legacy_metadata.get("processor_version", "1.0.0"),
        chunk_size=legacy_metadata.get("chunk_size", 1000),
        chunk_overlap=legacy_metadata.get("chunk_overlap", 200)
    )
    
    # Set storage info
    builder.set_storage_info(
        chunks_directory=legacy_metadata.get("chunks_directory"),
        vector_store_updated=legacy_metadata.get("vector_store_updated", False)
    )
    
    # Add any remaining fields as custom metadata
    known_fields = {
        "source_id", "source_name", "source_type", "file_extension", "file_type",
        "original_size_bytes", "file_size_bytes", "mime_type", "processing_method",
        "processing_status", "processing_timestamp", "processing_time_seconds",
        "process_time", "total_chunks", "total_characters", "total_words",
        "ocr_enabled", "ocr_used", "ocr_time_seconds", "ocr_time", "has_tables",
        "has_images", "has_links", "has_formulas", "conversion_success",
        "confidence_score", "error_count", "processor_version", "chunk_size",
        "chunk_overlap", "chunks_directory", "vector_store_updated"
    }
    
    for key, value in legacy_metadata.items():
        if key not in known_fields and value is not None:
            builder.add_custom_metadata(key, value)
    
    return builder.build()
