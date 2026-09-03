"""
Core functionality package for the RAG chatbot system.
Organized by domain for better maintainability and clarity.
"""

# Import main components from domain modules
from .ai_services import ChatbotService, PromptManager, EmbeddingService
from .document_processing import MainDocumentProcessor
from .retrieval import ContextRetriever
from .storage import VectorStore, DocumentStore

# Export all main components for convenient access
__all__ = [
    "ChatbotService",  # RAG-powered conversation engine
    "PromptManager",  # System prompt management
    "MainDocumentProcessor",  # Document processing pipeline
    "EmbeddingService",  # Text embedding generation
    "ContextRetriever",  # Document retrieval system
    "VectorStore",  # Vector storage management
    "DocumentStore",  # Document metadata storage
]
