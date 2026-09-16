"""
Pydantic models for request and response validation.
"""

# Local imports
from .responses import (
    StatusEnum,
    BaseResponse,
    HealthResponse,
    ChatResponse,
    FileProcessResult,
    MultipleFileUploadResponse,
    ErrorResponse,
)

from .metadata import (
    NormalizedMetadata,
    MetadataBuilder,
    ProcessingMethod,
    SourceType,
    ProcessingStatus,
    normalize_legacy_metadata,
)
