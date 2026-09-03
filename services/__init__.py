"""
Services package for high-level business logic and API integration.
Provides service layer abstraction for document processing, chat, and file management.
"""

# Import all services for easy access and dependency injection
from .document_service import DocumentService
from .chat_service import ChatService
from .upload_service import UploadService

# Export all service classes for convenient access
__all__ = [
    "DocumentService",  # Document processing and management
    "ChatService",  # RAG-powered conversation service
    "UploadService",  # File upload and processing
]
