"""
FastAPI dependencies for dependency injection and service management.
Provides service instances for API endpoints with proper dependency injection.
"""

# Local imports
from services import DocumentService, ChatService, UploadService
from core.ai_services import ChatbotService
from core.retrieval import ContextRetriever
from core.storage.vector_stores import VectorStore
import os


def get_document_service() -> DocumentService:
    """Get document service instance for document processing and management."""
    # Get preprocessing configuration from environment variable
    preprocessing_config = os.getenv("PREPROCESSING_CONFIG", "ocr_optimized")
    return DocumentService(preprocessing_config=preprocessing_config)


def get_chat_service() -> ChatService:
    """Get chat service instance for RAG-powered conversation handling."""
    # Wire up dependencies explicitly
    context_retriever = ContextRetriever()
    chatbot_service = ChatbotService(context_retriever=context_retriever)
    return ChatService(chatbot_service=chatbot_service)


def get_upload_service() -> UploadService:
    """Get upload service instance for file upload and processing."""
    return UploadService()


def get_vector_store() -> VectorStore:
    """Get vector store instance for document retrieval and management."""
    from core.storage.vector_stores.vector_store_optimized import OptimizedVectorStore
    return OptimizedVectorStore()
