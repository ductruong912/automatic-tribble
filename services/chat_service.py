# Standard library imports
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
import time
from datetime import datetime
import numpy as np

# Local imports
from core.ai_services import ChatbotService
from core.ai_services.llm.chatbot import _build_context
from core.storage.vector_stores import VectorStore
from config.settings import Config
from models.responses import ErrorResponse, BaseResponse, StatusEnum
from utils.performance import get_performance_monitor
from utils.performance import get_model_preloader
from core.infrastructure.caching.cache_service import get_cache_service

logger = logging.getLogger(__name__)


class ChatUnavailableError(RuntimeError):
    """
    Chat cannot start, carrying the HTTP status the API should answer with.

    Raised instead of yielded as a token: once a streaming response begins, the status
    line has already been sent and can no longer report the failure.
    """

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class ChatService:
    """
    Enhanced chat service with RAG integration, performance monitoring, and robust error handling.
    Provides comprehensive chat functionality with knowledge base integration and service health tracking.
    """

    def __init__(self, chatbot_service=None):
        """
        Initialize chat service with core components and monitoring infrastructure.
        Sets up chatbot service, request tracking, and performance metrics.
        """
        # Initialize core components and monitoring
        if chatbot_service is None:
            self.chatbot_service = ChatbotService()
        else:
            self.chatbot_service = chatbot_service
        # No process-wide conversation history: the service is shared by every
        # caller, so a single list would mix one user's conversation into another's
        # prompt. History must come from the request.
        self.service_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "service_start_time": datetime.now().isoformat(),
        }
        
        # Performance optimizations
        self._vector_store_cache = None
        self._vector_store_last_loaded = 0
        self._vector_store_ttl = 300  # 5 minutes cache
        self._vector_store_version = -1  # Preloader version this cache was built from
        
        # Use existing smart cache service
        self._smart_cache = get_cache_service()
        
        # Performance monitoring
        self._performance_monitor = get_performance_monitor()

    def _get_cached_vector_store(self):
        """Get the vector store, reloading whenever an upload has changed it."""
        # Preloader owns the shared copy and reloads it after every rebuild
        preloader = get_model_preloader()
        preloaded_data = preloader.get_vector_store_data()
        if preloaded_data is not None:
            logger.debug("⚡ Using preloaded vector store")
            return preloaded_data

        # Fallback: load here, expiring on TTL or on any vector store change
        current_time = time.time()
        version = preloader.get_vector_store_version()
        if (self._vector_store_cache is None
                or version != self._vector_store_version
                or current_time - self._vector_store_last_loaded > self._vector_store_ttl):
            try:
                # Use OptimizedVectorStore for better performance
                from core.storage.vector_stores.vector_store_optimized import OptimizedVectorStore
                vs = OptimizedVectorStore()
                self._vector_store_cache = vs.load_vector_store()
                self._vector_store_last_loaded = current_time
                self._vector_store_version = version
                logger.debug("🔄 Vector store cache refreshed")
            except Exception as e:
                logger.error(f" Error loading vector store: {e}")
                return None
        return self._vector_store_cache

    def _get_cached_search_results(self, query: str, embeddings, documents, k: int, semantic_weight: float):
        """Get cached search results using the existing smart cache service."""
        import hashlib
        context_fingerprint = hashlib.sha256()
        for document in documents:
            encoded_document = str(document).encode("utf-8")
            context_fingerprint.update(len(encoded_document).to_bytes(8, "big"))
            context_fingerprint.update(encoded_document)
        context_hash = hashlib.sha256(
            f"{context_fingerprint.hexdigest()}|{k}|{semantic_weight}".encode()
        ).hexdigest()
        
        # Try to get from smart cache first
        cached_results = self._smart_cache.get(query, context_hash)
        if cached_results is not None:
            logger.debug("⚡ Using cached search results from smart cache")
            return cached_results
        
        # Perform search and cache result
        try:
            results = self.chatbot_service.context_retriever.hybrid_search(
                query=query,
                embeddings=embeddings,
                documents=documents,
                k=k,
                semantic_weight=semantic_weight
            )
            
            # Cache using smart cache service
            self._smart_cache.set(query, results, context_hash)
            logger.debug("💾 Cached search results in smart cache")
            return results
        except Exception as e:
            logger.warning(f" Search failed: {e}")
            return []

    def _update_average_response_time(self, response_time: float):
        """Update rolling average response time."""
        current_avg = self.service_metrics["average_response_time"]
        successful_requests = self.service_metrics["successful_requests"]

        if successful_requests == 1:
            self.service_metrics["average_response_time"] = response_time
        else:
            self.service_metrics["average_response_time"] = (
                current_avg * (successful_requests - 1) + response_time
            ) / successful_requests

    def get_knowledge_base_status(self) -> Dict[str, Any]:
        """
        Get comprehensive knowledge base status with OCR-level detail.

        Returns:
            Dict containing detailed knowledge base information
        """
        try:
            logger.debug("Checking knowledge base status...")
            vs = getattr(self.chatbot_service.context_retriever, "vector_store", None)
            vector_store_result = self._get_cached_vector_store()
            if vector_store_result is not None:
                _, current_embeddings, current_documents = vector_store_result
            elif vs is not None:
                _, current_embeddings, current_documents = vs.load_vector_store()
            else:
                current_embeddings, current_documents = None, None

            # Calculate additional metrics
            embedding_dimensions = 0
            if current_embeddings is not None and len(current_embeddings) > 0:
                embedding_dimensions = (
                    current_embeddings.shape[1]
                    if hasattr(current_embeddings, "shape")
                    else 0
                )

            vs_path = getattr(vs, "vector_store_path", "data/vector_stores")

            status_info = {
                "available": bool(current_documents),
                "document_count": len(current_documents) if current_documents else 0,
                "embedding_count": (
                    len(current_embeddings) if current_embeddings is not None else 0
                ),
                "embedding_dimensions": embedding_dimensions,
                "status": "ready" if current_documents else "empty",
                "last_updated": datetime.now().isoformat(),
                "vector_store_path": str(vs_path),
                "health": "healthy" if current_documents else "no_data",
            }

            logger.debug(
                f" Knowledge base status: {status_info['status']} ({status_info['document_count']} docs)"
            )
            return status_info

        except Exception as e:
            logger.error(f" Error getting knowledge base status: {str(e)}")
            return {
                "available": False,
                "document_count": 0,
                "embedding_count": 0,
                "embedding_dimensions": 0,
                "status": "error",
                "health": "unhealthy",
                "last_updated": datetime.now().isoformat(),
                "error": str(e),
            }

    def get_comprehensive_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status like OCR's service monitoring.

        Returns:
            Dict containing detailed service health and performance metrics
        """
        try:
            # Get knowledge base status
            kb_status = self.get_knowledge_base_status()

            # Get chatbot service status
            chatbot_status = self.chatbot_service.get_service_status()

            # Calculate success rate
            total_requests = self.service_metrics["total_requests"]
            success_rate = 0.0
            if total_requests > 0:
                success_rate = (
                    self.service_metrics["successful_requests"] / total_requests
                ) * 100

            # Determine overall health
            overall_health = "healthy"
            if not chatbot_status["service_available"]:
                overall_health = "unhealthy"
            elif not kb_status["available"]:
                overall_health = "no_data"
            elif (
                success_rate < Config.Health.SERVICE_SUCCESS_RATE_THRESHOLD()
                and total_requests > Config.Health.SERVICE_MIN_REQUESTS_FOR_HEALTH()
            ):
                overall_health = "degraded"

            return {
                "service_name": "ChatService",
                "overall_health": overall_health,
                "service_available": chatbot_status["service_available"],
                "knowledge_base": kb_status,
                "chatbot_service": chatbot_status,
                "service_metrics": {
                    **self.service_metrics,
                    "success_rate_percent": round(success_rate, 2),
                    "failure_rate_percent": round(100 - success_rate, 2),
                },
                "timestamp": datetime.now().isoformat(),
                "uptime_info": {
                    "service_start_time": self.service_metrics["service_start_time"],
                    "current_time": datetime.now().isoformat(),
                },
            }

        except Exception as e:
            logger.error(f" Error getting comprehensive service status: {e}")
            return {
                "service_name": "ChatService",
                "overall_health": "error",
                "service_available": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def reset_metrics(self):
        """Reset service metrics for monitoring purposes."""
        logger.info("🔄 Resetting ChatService metrics")
        self.service_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "service_start_time": datetime.now().isoformat(),
        }

    def clear_cache(self) -> Dict[str, Any]:
        """Clear all caches including smart cache and local caches."""
        try:
            # Clear chatbot service cache
            chatbot_result = self.chatbot_service.clear_cache()
            
            # Clear smart cache
            self._smart_cache.clear()
            smart_cache_stats = self._smart_cache.get_stats()
            
            # Clear local vector store cache
            self._vector_store_cache = None
            self._vector_store_last_loaded = 0
            self._vector_store_version = -1

            logger.info("🧹 All caches cleared successfully")
            return {
                "message": "All caches cleared successfully",
                "chatbot_cache": chatbot_result,
                "smart_cache_cleared": True,
                "smart_cache_stats": smart_cache_stats,
                "vector_store_cache_cleared": True,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error("Failed to clear cache: %s", e)
            return {
                "message": "Failed to clear cache",
                "error": str(e),
                "timestamp": time.time()
            }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics from all cache layers."""
        try:
            # Get smart cache stats
            smart_cache_stats = self._smart_cache.get_stats()
            
            # Get chatbot cache stats
            chatbot_status = self.chatbot_service.get_service_status()
            
            return {
                "smart_cache": smart_cache_stats,
                "chatbot_cache": {
                    "size": chatbot_status.get("cache_size", 0),
                    "model": chatbot_status.get("model", "unknown"),
                    "available": chatbot_status.get("service_available", False)
                },
                "vector_store_cache": {
                    "cached": self._vector_store_cache is not None,
                    "last_loaded": self._vector_store_last_loaded,
                    "ttl_seconds": self._vector_store_ttl
                },
                "performance_metrics": self._performance_monitor.get_performance_stats(),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error("Failed to get cache stats: %s", e)
            return {
                "error": str(e),
                "timestamp": time.time()
            }

    def ensure_can_stream(self, query: Optional[str]) -> None:
        """
        Check everything knowable before the first token, raising ChatUnavailableError.

        Call this before opening the streaming response so these failures come back as a
        real HTTP status instead of as an error message inside a successful stream.
        """
        if not query or not query.strip():
            raise ChatUnavailableError("Query cannot be empty.", 400)
        if not self.chatbot_service.api_available:
            raise ChatUnavailableError("Chat service is currently unavailable.", 503)
        if self._get_cached_vector_store() is None:
            raise ChatUnavailableError("Knowledge base is unavailable.", 503)

    async def stream_chat_with_memory(
        self, query: str, custom_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Async generator for streaming chat responses with memory/history support.
        Calls the streaming method on ChatbotService and yields tokens.

        Callers should run ensure_can_stream() first; the checks repeated here only
        cover the vector store cache expiring between that call and this one.
        """
        # Load vector store with caching
        vector_store_result = self._get_cached_vector_store()
        if vector_store_result is None:
            raise ChatUnavailableError("Knowledge base is unavailable.", 503)

        _, current_embeddings, current_documents = vector_store_result
        if not current_documents:
            current_embeddings = np.array([])
            current_documents = []
        else:
            if current_embeddings is None:
                current_embeddings = np.array([])
            elif isinstance(current_embeddings, list):
                current_embeddings = np.array(current_embeddings)
        # History comes from the caller only. The service is shared across every
        # request, so falling back to a process-wide list would mix conversations
        # between users.
        history = custom_history[-Config.LLM.LLM_HISTORY_LENGTH():] if custom_history else []
        # Note: Don't add current query to history here - it will be added in stream_response_with_history
        # Stream response from chatbot_service
        async for token in self.chatbot_service.stream_response_with_history(
            query,
            embeddings=current_embeddings,
            documents=current_documents,
            history=history,
        ):
            yield token
        # Optionally update global history (not done here to avoid race conditions in streaming)
