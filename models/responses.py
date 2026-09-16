"""
Pydantic models for API responses - Structured data models for consistent API communication.
Defines response schemas for chat, file uploads, health checks, and error handling.
"""

# Standard library imports
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union

# Third-party imports
from pydantic import BaseModel, Field, PrivateAttr


class StatusEnum(str, Enum):
    """Status enumeration for API responses - standardizes response status values."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class BaseResponse(BaseModel):
    """Base response model with common fields for all API responses."""

    status: StatusEnum = Field(..., description="Response status")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )


class HealthResponse(BaseResponse):
    """Health check response model for system monitoring and status endpoints."""

    version: str = Field(..., description="API version")
    uptime: Optional[str] = Field(None, description="Server uptime")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "RAG Chatbot API is running",
                "version": "2.0.0",
                "timestamp": "2024-01-01T12:00:00Z",
            }
        }


class ChatResponse(BaseResponse):
    """Enhanced chat response model with confidence scoring and search metadata."""

    response: str = Field(..., description="AI response to the query")
    query: str = Field(..., description="Original query")
    confidence: Optional[Dict[str, Any]] = Field(
        None, description="Confidence scoring details"
    )
    search_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Search and retrieval metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "response": "Based on the available information, we have several products...",
                "query": "What products are available?",
                "confidence": {
                    "score": 0.85,
                    "level": "High",
                    "details": {
                        "context_alignment": 0.9,
                        "response_length_appropriateness": 0.8,
                        "semantic_coherence": 0.85,
                        "source_citation": 0.7,
                        "uncertainty_indicators": 0.9,
                        "reasoning": "Response well-aligned with provided context. Response length appropriate for query complexity.",
                    },
                },
                "search_metadata": {
                    "results_count": 3,
                    "top_scores": [0.92, 0.85, 0.78],
                    "cached_response": False,
                },
                "timestamp": "2024-01-01T12:00:00Z",
            }
        }


class FileProcessResult(BaseModel):
    """Individual file processing result for batch upload responses."""

    filename: str = Field(..., description="Processed filename")
    file_size: int = Field(..., description="File size in bytes")
    document_count: int = Field(..., description="Number of documents extracted")
    source_id: str = Field(..., description="Generated source ID")
    status: StatusEnum = Field(..., description="Processing status")
    error_message: Optional[str] = Field(
        None, description="Error message if processing failed"
    )
    # Streamlined metadata - only essential info
    file_type: str = Field(..., description="File extension")
    processing_method: str = Field(..., description="Processing method used")
    processing_time: Optional[str] = Field(None, description="Processing time")
    # Optional detailed metadata for debugging
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Additional debug information")


class MultipleFileUploadResponse(BaseResponse):
    """Multiple file upload response model for batch processing results."""

    total_files: int = Field(..., description="Total number of files processed")
    successful_files: int = Field(
        ..., description="Number of successfully processed files"
    )
    failed_files: int = Field(..., description="Number of failed files")
    total_documents: int = Field(
        ..., description="Total documents extracted from all files"
    )
    results: List[FileProcessResult] = Field(
        ..., description="Individual file processing results"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Processed 3 files: 2 successful, 1 failed",
                "total_files": 3,
                "successful_files": 2,
                "failed_files": 1,
                "total_documents": 25,
                "results": [
                    {
                        "filename": "doc1.pdf",
                        "file_size": 1024000,
                        "document_count": 15,
                        "source_id": "uuid-123",
                        "status": "success",
                        "file_type": ".pdf",
                        "processing_method": "docling",
                        "processing_time": "2.5s",
                        "debug_info": {
                            "chunks_directory": "data/chunks/doc1",
                            "ocr_enabled": True,
                            "conversion_success": True
                        }
                    },
                    {
                        "filename": "doc2.txt",
                        "file_size": 5000,
                        "document_count": 0,
                        "source_id": "",
                        "status": "error",
                        "error_message": "File too small",
                        "file_type": ".txt",
                        "processing_method": "unknown",
                        "processing_time": None,
                        "debug_info": None
                    },
                ],
                "timestamp": "2024-01-01T12:00:00Z",
            }
        }


class ErrorResponse(BaseResponse):
    """Error response model for standardized error handling across the API."""

    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "File processing failed",
                "error_code": "PROCESSING_ERROR",
                "details": {"reason": "Unsupported file format"},
                "timestamp": "2024-01-01T12:00:00Z",
            }
        }


class VectorRebuildResponse(BaseResponse):
    """Vector rebuild response model for force rebuilding vectors with current chunks."""

    documents_processed: int = Field(..., description="Number of documents processed")
    vectors_created: int = Field(..., description="Number of vectors created")
    processing_time: str = Field(..., description="Time taken to rebuild vectors")
    vector_store_path: str = Field(..., description="Path to the vector store file")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional rebuild details")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Vector store rebuilt successfully",
                "documents_processed": 150,
                "vectors_created": 150,
                "processing_time": "2.5s",
                "vector_store_path": "data/vectors/vector_store.pkl",
                "details": {
                    "chunk_files_loaded": 25,
                    "embedding_model": "text-embedding-3-small",
                    "vector_dimensions": 1536
                },
                "timestamp": "2024-01-01T12:00:00Z",
            }
        }


class ChatRequest(BaseModel):
    """Chat request model for context-aware responses with conversation history support."""

    query: str = Field(..., description="User query", min_length=1, max_length=2000)
    history: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="Conversation history for context-aware responses"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Hi?",
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help you?"}
                ]
            }
        }