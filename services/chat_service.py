# Standard library imports
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
import time
from datetime import datetime
import numpy as np
import asyncio

# Local imports
from core.ai_services import ChatbotService
from core.storage.vector_stores import VectorStore
from config.settings import Config
from models.responses import ErrorResponse, BaseResponse, StatusEnum
from utils.performance import get_performance_monitor
from utils.performance import get_model_preloader
from core.infrastructure.caching.cache_service import get_cache_service

logger = logging.getLogger(__name__)


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
        self.request_history = []
        self._history_lock = asyncio.Lock()
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
        
        # Use existing smart cache service
        self._smart_cache = get_cache_service()
        
        # Performance monitoring
        self._performance_monitor = get_performance_monitor()

    def _get_cached_vector_store(self):
        """Get cached vector store or load if expired."""
        # Try to get preloaded vector store first
        preloader = get_model_preloader()
        preloaded_data = preloader.get_vector_store_data()
        if preloaded_data is not None:
            logger.debug("⚡ Using preloaded vector store")
            return preloaded_data
        
        # Fallback to cached loading
        current_time = time.time()
        if (self._vector_store_cache is None or 
            current_time - self._vector_store_last_loaded > self._vector_store_ttl):
            try:
                # Use OptimizedVectorStore for better performance
                from core.storage.vector_stores.vector_store_optimized import OptimizedVectorStore
                vs = OptimizedVectorStore()
                self._vector_store_cache = vs.load_vector_store()
                self._vector_store_last_loaded = current_time
                logger.debug("🔄 Vector store cache refreshed")
            except Exception as e:
                logger.error(f" Error loading vector store: {e}")
                return None
        return self._vector_store_cache

    def _get_cached_search_results(self, query: str, embeddings, documents, k: int, semantic_weight: float):
        """Get cached search results using the existing smart cache service."""
        import hashlib
        context_hash = f"{len(documents)}_{k}_{semantic_weight}"
        
        # Try to get from smart cache first
        cached_results = self._smart_cache.get(query, context_hash, use_similarity=True)
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

    async def chat_with_memory(
        self, query: str, custom_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Unified chat method: always uses history memory, configurable length.
        If custom_history is provided, use it; otherwise, use global history.
        """
        start_time = time.time()
        request_id = f"chat_{int(time.time() * 1000)}"
        if logger.isEnabledFor(logging.INFO):
            logger.info("Processing chat request %s: %s...", request_id, query[:50])
        try:
            # Update request metrics for monitoring
            self.service_metrics["total_requests"] += 1

            # Validate input query for processing
            if not query or not query.strip():
                return self._create_error_response(
                    query, "Empty query provided", request_id, start_time
                ).dict()

            # Check chatbot service availability before processing
            if not self.chatbot_service.api_available:
                logger.error(" ChatbotService not available")
                return self._create_error_response(
                    query,
                    "Chat service is currently unavailable",
                    request_id,
                    start_time,
                ).dict()

            # Load vector store with caching for performance
            vector_store_result = self._get_cached_vector_store()
            if vector_store_result is None:
                return self._create_error_response(
                    query,
                    "Knowledge base is currently unavailable",
                    request_id,
                    start_time,
                ).dict()
            
            _, current_embeddings, current_documents = vector_store_result
            is_cached = False  # Initialize is_cached variable

            # Handle empty knowledge base gracefully with universal prompt
            if not current_documents:
                logger.info(
                    "💬 No documents in knowledge base - using universal prompt"
                )
                current_embeddings = np.array([])
                current_documents = []
            else:
                logger.info(
                    f" Knowledge base loaded: {len(current_documents)} documents"
                )
                if current_embeddings is None:
                    current_embeddings = np.array([])
                elif isinstance(current_embeddings, list):
                    current_embeddings = np.array(current_embeddings)

            # Perform RAG search with caching for performance
            context = ""
            search_results = []
            if current_documents and len(current_documents) > 0 and current_embeddings is not None:
                try:
                    # Use cached search results for better performance
                    search_start_time = time.time()
                    search_results = self._get_cached_search_results(
                        query=query,
                        embeddings=current_embeddings,
                        documents=current_documents,
                        k=Config.RAG.RETRIEVAL_TOP_K(),
                        semantic_weight=Config.RAG.SEMANTIC_WEIGHT()
                    )
                    search_time = time.time() - search_start_time
                    logger.info(f"⏱️ RAG search time: {search_time:.3f}s")
                    
                    # Build context from search results (same as test)
                    context_chunks = []
                    for result in search_results:
                        context_chunks.append(f"[Chunk {result['index']}]\n{result['document']}\n")
                    context = "\n".join(context_chunks)
                    
                    # Truncate context if too long (same as test)
                    max_context_length = Config.LLM.MAX_CONTEXT_LENGTH()
                    if len(context) > max_context_length:
                        context = context[:max_context_length]
                        logger.debug(f" Context truncated to {max_context_length} characters")
                        
                except Exception as e:
                    logger.warning(f" RAG search failed: {e}")
                    context = ""
                    search_results = []

            # Prepare history (lock only around shared history access)
            if custom_history is not None:
                history = custom_history[-Config.LLM.LLM_HISTORY_LENGTH() :]
            else:
                async with self._history_lock:
                    history = self.request_history[-Config.LLM.LLM_HISTORY_LENGTH() :]

            # Generate response using the same context for both modes
            try:
                llm_start_time = time.time()
                if history and len(history) > 0:
                    # History mode: use get_response_with_history but with pre-built context
                    logger.debug(f"🔄 Calling LLM with history (context: {len(context)} chars)")
                    result = self.chatbot_service.get_response_with_history_and_context(
                        query=query,
                        context=context,
                        search_results=search_results,
                        history=history,
                    )
                else:
                    # Query-only mode: use get_response_with_context
                    logger.debug(f"🔄 Calling LLM without history (context: {len(context)} chars)")
                    result = self.chatbot_service.get_response_with_context(
                        query=query,
                        context=context,
                        search_results=search_results,
                    )
                llm_time = time.time() - llm_start_time
                logger.info(f"⏱️ LLM response time: {llm_time:.3f}s")
                # Calculate total processing time for performance monitoring
                processing_time = time.time() - start_time
                # Update service metrics for health monitoring
                self.service_metrics["successful_requests"] += 1
                self._update_average_response_time(processing_time)
                
                # Record performance metrics
                if hasattr(self, '_performance_monitor'):
                    self._performance_monitor.record_request(processing_time, is_cached)
                    self._performance_monitor.record_system_metrics()
                
                if logger.isEnabledFor(logging.INFO):
                    logger.info(" Request %s completed successfully in %.2fs", request_id, processing_time)
                # Extract response and metadata from result for comprehensive response
                if hasattr(result, 'response'):
                    # ChatResponse object
                    response_text = result.response
                    confidence_score = result.confidence.get("score", 0.0) if result.confidence else 0.0
                    confidence_level = result.confidence.get("level", "Unknown") if result.confidence else "Unknown"
                    confidence_details = result.confidence.get("details", {}) if result.confidence else {}
                    search_results = result.search_metadata if hasattr(result, 'search_metadata') else {}
                    is_cached = result.search_metadata.get("cached_response", False) if hasattr(result, 'search_metadata') else False
                else:
                    # Dictionary format
                    response_text = result.get("response", "")
                    confidence_score = result.get("confidence", 0.0)
                    confidence_level = result.get("confidence_level", "Unknown")
                    confidence_details = result.get("confidence_details", {})
                    search_results = result.get("search_results", {})
                    is_cached = result.get("cached", False)
                # Update global history (append user and assistant turns)
                if custom_history is None:
                    async with self._history_lock:
                        self.request_history.append({"role": "user", "content": query})
                        self.request_history.append(
                            {"role": "assistant", "content": response_text}
                        )
                        # Trim to max history length
                        self.request_history = self.request_history[
                            -(Config.LLM.LLM_HISTORY_LENGTH() * 2) :
                        ]
                return BaseResponse(
                    status=StatusEnum.SUCCESS,
                    response=response_text,
                    query=query,
                    request_id=request_id,
                    document_count=len(current_documents),
                    processing_time=processing_time,
                    confidence={
                        "score": confidence_score,
                        "level": confidence_level,
                        "details": confidence_details,
                    },
                    search_metadata={
                        "results_count": search_results.get("count", 0),
                        "top_scores": search_results.get("top_scores", []),
                        "cached_response": is_cached,
                    },
                    error=None,
                ).dict()
            except Exception as e:
                logger.error(" Error in chatbot service for request %s: %s", request_id, e)
                self.service_metrics["failed_requests"] += 1
                return self._create_error_response(
                    query,
                    f"Error generating response: {str(e)}",
                    request_id,
                    start_time,
                    document_count=len(current_documents),
                ).dict()
        finally:
            pass

    def _create_error_response(
        self,
        query: str,
        error_msg: str,
        request_id: str,
        start_time: float,
        document_count: int = 0,
    ) -> ErrorResponse:
        """Create standardized error response using ErrorResponse model."""
        processing_time = time.time() - start_time
        return ErrorResponse(
            status=StatusEnum.ERROR,
            message=error_msg,
            error_code="CHAT_ERROR",
            details={
                "query": query,
                "request_id": request_id,
                "document_count": document_count,
                "processing_time": processing_time,
            },
        )

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
            _, current_embeddings, current_documents = vector_store.load_vector_store()

            # Calculate additional metrics
            embedding_dimensions = 0
            if current_embeddings is not None and len(current_embeddings) > 0:
                embedding_dimensions = (
                    current_embeddings.shape[1]
                    if hasattr(current_embeddings, "shape")
                    else 0
                )

            status_info = {
                "available": bool(current_documents),
                "document_count": len(current_documents) if current_documents else 0,
                "embedding_count": (
                    len(current_embeddings) if current_embeddings is not None else 0
                ),
                "embedding_dimensions": embedding_dimensions,
                "status": "ready" if current_documents else "empty",
                "last_updated": datetime.now().isoformat(),
                "vector_store_path": vector_store.vector_store_path,
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
        self.request_history.clear()

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

    async def stream_chat_with_memory(
        self, query: str, custom_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Async generator for streaming chat responses with memory/history support.
        Calls the streaming method on ChatbotService and yields tokens.
        """
        # Validate input query
        if not query or not query.strip():
            yield "[ERROR] Empty query provided."
            return
        if not self.chatbot_service.api_available:
            yield "[ERROR] Chat service is currently unavailable."
            return
        # Load vector store with caching
        vector_store_result = self._get_cached_vector_store()
        if vector_store_result is None:
            yield "[ERROR] Knowledge base unavailable"
            return
        
        _, current_embeddings, current_documents = vector_store_result
        if not current_documents:
            current_embeddings = np.array([])
            current_documents = []
        else:
            if current_embeddings is None:
                current_embeddings = np.array([])
            elif isinstance(current_embeddings, list):
                current_embeddings = np.array(current_embeddings)
        # Prepare history
        if custom_history is not None:
            history = custom_history[-Config.LLM.LLM_HISTORY_LENGTH() :]
        else:
            async with self._history_lock:
                history = self.request_history[-Config.LLM.LLM_HISTORY_LENGTH() :]
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
