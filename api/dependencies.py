"""
FastAPI dependencies for dependency injection and service management.

Services are process-wide singletons built once at startup. Creating them per
request reloads the reranker from disk, re-probes the OpenAI API and throws away
every cache, which makes the whole caching layer pointless.
"""

# Standard library imports
import logging
import os
import threading

# Local imports
from services import DocumentService, ChatService, UploadService
from core.ai_services import ChatbotService
from core.retrieval import ContextRetriever
from core.storage.vector_stores import VectorStore

logger = logging.getLogger(__name__)

# Reentrant: get_upload_service() asks for the document service while holding it
_services_lock = threading.RLock()
_document_service = None
_chat_service = None
_upload_service = None
_vector_store = None


def get_document_service() -> DocumentService:
    """Get the shared document service for document processing and management."""
    global _document_service
    if _document_service is None:
        with _services_lock:
            if _document_service is None:
                _document_service = DocumentService()
                logger.info("DocumentService ready")
    return _document_service


def get_chat_service() -> ChatService:
    """Get the shared chat service for RAG-powered conversation handling."""
    global _chat_service
    if _chat_service is None:
        with _services_lock:
            if _chat_service is None:
                from utils.performance import get_model_preloader

                preloader = get_model_preloader()

                def shared_vector_store():
                    """Current shared store and its version, reloaded first if an upload marked it stale."""
                    preloader.get_vector_store_data()
                    return preloader.get_model("vector_store"), preloader.get_vector_store_version()

                # Reuse the reranker already loaded by the preloader; follow its vector store
                context_retriever = ContextRetriever(
                    reranker=preloader.get_model("reranker"),
                    vector_store_provider=shared_vector_store,
                )
                chatbot_service = ChatbotService(context_retriever=context_retriever)
                _chat_service = ChatService(chatbot_service=chatbot_service)
                logger.info("ChatService ready")
    return _chat_service


def get_upload_service() -> UploadService:
    """Get the shared upload service for file upload and processing."""
    global _upload_service
    if _upload_service is None:
        with _services_lock:
            if _upload_service is None:
                _upload_service = UploadService(document_service=get_document_service())
                logger.info("UploadService ready")
    return _upload_service


def get_vector_store() -> VectorStore:
    """Get the shared vector store instance for document retrieval and management."""
    global _vector_store
    if _vector_store is None:
        with _services_lock:
            if _vector_store is None:
                from core.storage.vector_stores.vector_store_optimized import OptimizedVectorStore

                _vector_store = OptimizedVectorStore()
                logger.info("Vector store ready")
    return _vector_store


def init_services():
    """
    Build every shared service once, at startup.
    Keeps model loading and the OpenAI probe off the request path.
    """
    get_document_service()
    get_vector_store()
    chat_service = get_chat_service()

    # One API probe per process instead of one per request
    try:
        available = chat_service.chatbot_service.probe_api()
        logger.info(f"OpenAI API available: {available}")
    except Exception as e:
        logger.warning(f"OpenAI API probe failed: {e}")

    get_upload_service()
    logger.info("All shared services initialized")


async def shutdown_services():
    """Release shared resources at shutdown."""
    global _document_service, _chat_service, _upload_service, _vector_store

    if _chat_service is not None:
        try:
            await _chat_service.chatbot_service.aclose()
        except Exception as e:
            logger.warning(f"Error closing chatbot service: {e}")

    _document_service = None
    _chat_service = None
    _upload_service = None
    _vector_store = None
    logger.info("Shared services released")
