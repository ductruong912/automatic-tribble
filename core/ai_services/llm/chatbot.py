"""
Simple, powerful chatbot service - RAG-powered conversation engine.
Handles query processing, context retrieval, and response generation with confidence scoring.
"""

import logging
import time
import asyncio
import hashlib
import threading
from typing import Dict, Any, List, Optional, AsyncGenerator, Union, cast
from ...infrastructure.caching.cache_service import get_cache_service
import openai
from openai import AsyncOpenAI, APIStatusError, APIConnectionError
from openai.types.chat import ChatCompletionMessageParam

from config.settings import Config
from .prompts import PromptManager, SystemPrompts
from ..confidence.confidence import ConfidenceScorer
from models.responses import ChatResponse, ErrorResponse, BaseResponse, StatusEnum

logger = logging.getLogger(__name__)

# Roles a client may supply. "system" is excluded on purpose: accepting it would
# let the request body replace the bot's own instructions.
_ALLOWED_HISTORY_ROLES = {"user", "assistant"}


def _sanitize_history(
    history: Optional[List[Dict[str, str]]]
) -> List[Dict[str, str]]:
    """
    Keep only well-formed user/assistant turns from client history and cap each one.
    Rebuilds every message so no extra client-supplied key reaches the API.
    """
    if not history:
        return []

    max_length = Config.LLM.MAX_HISTORY_MESSAGE_LENGTH()
    clean: List[Dict[str, str]] = []

    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        role = role.strip().lower()
        if role not in _ALLOWED_HISTORY_ROLES or not content.strip():
            continue
        clean.append({"role": role, "content": content[:max_length]})

    dropped = len(history) - len(clean)
    if dropped:
        logger.warning(f"Dropped {dropped} invalid message(s) from client history")

    return clean


def _build_context(search_results: List[Dict[str, Any]], max_length: int) -> str:
    """Pack retrieved chunks into the context whole, highest scoring first."""
    if not search_results:
        return ""

    ordered = sorted(
        search_results, key=lambda r: r.get("combined_score", 0.0), reverse=True
    )

    parts: List[str] = []
    used = 0
    for result in ordered:
        block = f"[Chunk {result['index']}]\n{result['document']}\n"
        cost = len(block) + (1 if parts else 0)  # 1 for the joining newline
        if used + cost > max_length:
            if parts:
                continue
            block = block[:max_length]  # Top chunk alone exceeds the budget
            cost = len(block)
        parts.append(block)
        used += cost

    dropped = len(ordered) - len(parts)
    if dropped:
        logger.debug(f"Context budget {max_length} chars: kept {len(parts)}, dropped {dropped}")

    return "\n".join(parts)


class ChatbotService:
    """
    RAG-powered chatbot service with smart caching, confidence scoring, and batch processing.
    Handles document retrieval, context building, and AI response generation.
    """

    def __init__(self, context_retriever=None):
        """
        Initialize ChatbotService with optional context retriever.
        
        Args:
            context_retriever: Optional ContextRetriever instance for RAG operations
        """
        from core.retrieval.search.retriever import ContextRetriever
        
        if context_retriever is None:
            self.context_retriever = ContextRetriever()
        else:
            self.context_retriever = context_retriever
        
        # Initialize missing components
        self.prompt_manager = PromptManager()
        self.confidence_scorer = ConfidenceScorer()
        
        # Initialize cache with LRU strategy
        self._cache = {}
        self._cache_ttl = Config.LLM.LLM_CACHE_TTL()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_access_times = {}  # Track access times for LRU
        self._cache_lock = threading.Lock()  # The service is shared across requests

        # Initialize async OpenAI client
        self.async_openai_client = AsyncOpenAI(api_key=Config.LLM.OPENAI_API_KEY())

        # Initialize thread pool workers
        self._max_workers = Config.LLM.LLM_MAX_WORKERS()

        # Assume usable when a key is configured; probe_api() confirms it at startup
        self._api_available = bool(Config.LLM.OPENAI_API_KEY())
        if not self._api_available:
            logger.warning(" OpenAI API key not configured")

    @property
    def api_available(self) -> bool:
        """Check if the OpenAI API is available."""
        return self._api_available

    def probe_api(self) -> bool:
        """
        Check the OpenAI API once and record the result.
        Called at startup - never from __init__, so building the service costs no network round trip.
        """
        self._api_available = self._test_api()
        return self._api_available

    def _test_api(self) -> bool:
        """
        Test OpenAI API availability.
        Treats HTTP errors (4xx) as "reachable" to allow startup,
        while catching genuine connection errors.
        """
        if not Config.LLM.OPENAI_API_KEY():
            logger.warning(" OpenAI API key not configured")
            return False
        try:
            client = openai.OpenAI(api_key=Config.LLM.OPENAI_API_KEY())
            client.chat.completions.create(
                model=Config.LLM.OPENAI_MODEL(),
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            logger.info(" OpenAI API available and responding.")
            return True
        except APIStatusError as e:
            # The API is reachable but returned an error status code (e.g., 401, 429).
            # This is treated as "available" for warm-up purposes.
            logger.warning(
                f" OpenAI API is reachable but returned status {e.status_code}. "
                f"This may indicate an API key or permission issue. "
                f"Treating as available for startup. Details: {e.message}"
            )
            return True
        except APIConnectionError as e:
            # A genuine network-level error occurred.
            logger.error(f" OpenAI API connection failed. The service is unreachable. Details: {e.__cause__}")
            return False
        except Exception as e:
            # Catch any other unexpected errors.
            logger.error(f" An unexpected error occurred while testing OpenAI API: {e}")
            return False

    def _create_error_response(self, query: str, error_msg: str) -> ErrorResponse:
        """Create standardized error response using ErrorResponse model."""
        return ErrorResponse(
            status=StatusEnum.ERROR,
            message=error_msg,
            error_code="LLM_ERROR",
            details={"query": query},
        )

    def get_response(self, query: str, embeddings=None, documents=None) -> Union[ChatResponse, ErrorResponse]:
        """
        Generate AI response using RAG system with context retrieval and confidence scoring.
        
        Args:
            query: User query to process
            embeddings: Document embeddings for semantic search
            documents: Document collection for context retrieval

        Returns:
            Dict containing response, confidence metrics, and search metadata
        """
        if not self.api_available:
            error_response = self._create_error_response(query, "I apologize, but the chat service is currently unavailable. Please try again later.")
            return error_response

        try:
            logger.info(f" [Request] Received query: '{query[:50]}...' (id={hashlib.md5(query.encode()).hexdigest()[:8]})")
            
            # Retrieve relevant context from documents using hybrid search
            context = ""
            search_results = []
            if documents is not None and embeddings is not None and len(documents) > 0:
                logger.info(f"\U0001f4da RAG enabled: {len(documents)} documents available")
                try:
                    # Clear cache to ensure consistent search results
                    self.context_retriever.clear_cache()
                    
                    # Use enhanced hybrid search with query expansion
                    search_results = self.context_retriever.hybrid_search(
                        query=query,
                        embeddings=embeddings,
                        documents=documents,
                        k=Config.RAG.RETRIEVAL_TOP_K(),
                        semantic_weight=Config.RAG.SEMANTIC_WEIGHT()
                    )
                    
                    # Assemble context from top-ranked chunks for precision
                    context = _build_context(search_results, Config.LLM.MAX_CONTEXT_LENGTH())

                    logger.info(f"📄 Retrieved context length: {len(context)} characters")

                    # Log context preview for debugging
                    if context:
                        context_preview = context[:200].replace('\n', ' ').replace('\r', ' ')
                        logger.info(f"📄 Context preview: {context_preview}...")
                    else:
                        logger.warning(" No context retrieved from documents")
                except Exception as e:
                    logger.warning(f" Context retrieval failed: {e}")
                    context = ""
            else:
                logger.info(" RAG disabled: No documents or embeddings provided")

            logger.debug(f"[Context] Retrieved {len(search_results)} context chunks for query (id={hashlib.md5(query.encode()).hexdigest()[:8]})")

            # Build OpenAI messages with system prompt and user query
            messages = [
                {"role": "system", "content": str(self.prompt_manager.get_system_prompt(context=context))}
            ]
            messages.append({"role": "user", "content": str(query)})

            # Check cache for performance optimization
            cache_key = self._get_cache_key(query, context)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                logger.debug(f"⚡ [Cache] Hit for query (id={hashlib.md5(query.encode()).hexdigest()[:8]})")
                # Calculate confidence for cached response
                confidence = self.confidence_scorer.calculate_confidence(
                    cached_response, query, context, search_results
                )
                return ChatResponse(
                    status=StatusEnum.SUCCESS,
                    message="Response generated successfully",
                    response=cached_response,
                    query=query,
                    confidence={
                        "score": confidence.overall_score,
                        "level": self.confidence_scorer.get_confidence_level(confidence.overall_score),
                        "details": {
                            "context_alignment": confidence.context_alignment,
                            "response_length_appropriateness": confidence.response_length_appropriateness,
                            "semantic_coherence": confidence.semantic_coherence,
                            "source_citation": confidence.source_citation,
                            "uncertainty_indicators": confidence.uncertainty_indicators,
                            "reasoning": confidence.reasoning
                        }
                    },
                    search_metadata={
                        "results_count": len(search_results),
                        "top_scores": [r.get("combined_score", 0) for r in search_results[:3] if r.get("combined_score", 0) > 0],
                        "cached_response": True
                    }
                )
            else:
                logger.debug(f"⚡ [Cache] Miss for query (id={hashlib.md5(query.encode()).hexdigest()[:8]})")

            # Get system prompt for context-aware responses
            system_prompt = self.prompt_manager.get_system_prompt(context=context)
            logger.debug(f"📝 System prompt length: {len(system_prompt)} characters")

            # Make optimized API call to OpenAI with better error handling
            logger.info(f" [LLM] Calling OpenAI for query (id={hashlib.md5(query.encode()).hexdigest()[:8]})")
            logger.debug(f"📝 [LLM Payload] {messages}")
            
            try:
                client = openai.OpenAI(
                    api_key=Config.LLM.OPENAI_API_KEY(),
                    timeout=Config.LLM.OPENAI_TIMEOUT(),
                    max_retries=2  # Add retry logic
                )
                response = client.chat.completions.create(
                    model=Config.LLM.OPENAI_MODEL(),
                    messages=messages,  # type: ignore
                    max_tokens=Config.LLM.OPENAI_MAX_TOKENS(),
                    temperature=Config.LLM.OPENAI_TEMPERATURE(),
                    timeout=Config.LLM.OPENAI_TIMEOUT()
                )
            except openai.APITimeoutError:
                logger.error("⏰ OpenAI API timeout - request took too long")
                error_response = self._create_error_response(query, "The AI service is taking too long to respond. Please try again with a simpler question.")
                return error_response
            except openai.RateLimitError:
                logger.error("🚫 OpenAI API rate limit exceeded")
                error_response = self._create_error_response(query, "The AI service is currently busy. Please try again in a moment.")
                return error_response
            except Exception as e:
                logger.error(f" OpenAI API error: {e}")
                error_response = self._create_error_response(query, f"AI service error: {str(e)}")
                return error_response

            content = response.choices[0].message.content
            response_text = content.strip() if content else ""
            logger.info(f" Response generated: {len(response_text)} characters")
            logger.debug(f"📝 [LLM Response] {response_text[:100]}... (id={hashlib.md5(query.encode()).hexdigest()[:8]})")

            # Calculate comprehensive confidence score
            confidence = self.confidence_scorer.calculate_confidence(
                response_text, query, context, search_results
            )
            
            logger.debug(f" [Confidence] Score for query (id={hashlib.md5(query.encode()).hexdigest()[:8]}): {confidence.overall_score}")

            # Cache the response for future requests
            self._cache_response(cache_key, response_text)

            logger.debug(" Response generated successfully")
            return ChatResponse(
                status=StatusEnum.SUCCESS,
                message="Response generated successfully",
                response=response_text,
                query=query,
                confidence={
                    "score": confidence.overall_score,
                    "level": self.confidence_scorer.get_confidence_level(confidence.overall_score),
                    "details": {
                        "context_alignment": confidence.context_alignment,
                        "response_length_appropriateness": confidence.response_length_appropriateness,
                        "semantic_coherence": confidence.semantic_coherence,
                        "source_citation": confidence.source_citation,
                        "uncertainty_indicators": confidence.uncertainty_indicators,
                        "reasoning": confidence.reasoning
                    }
                },
                search_metadata={
                    "results_count": len(search_results),
                    "top_scores": [r.get("combined_score", 0) for r in search_results[:3] if r.get("combined_score", 0) > 0],
                    "cached_response": False
                }
            )

        except Exception as e:
            logger.error(f" [Error] {str(e)} (id={hashlib.md5(query.encode()).hexdigest()[:8]})", exc_info=True)
            return self._create_error_response(query, "I apologize, but I encountered an error while processing your request. Please try again later.")

    def _get_cache_key(self, query: str, context: str) -> str:
        """Generate MD5-based cache key from query and context for efficient caching."""
        # Use only query for cache key to enable better caching
        # Context varies but similar queries should be cached
        content = f"{query.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Retrieve cached response if available and not expired based on TTL configuration."""
        with self._cache_lock:
            if cache_key in self._cache:
                cached_data = self._cache[cache_key]
                if time.time() - cached_data['timestamp'] < self._cache_ttl:
                    # Update access time for LRU
                    self._cache_access_times[cache_key] = time.time()
                    self._cache_hits += 1
                    return cached_data['response']
                else:
                    self._cache.pop(cache_key, None)  # Remove expired
                    self._cache_access_times.pop(cache_key, None)
            self._cache_misses += 1
            return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache response with LRU eviction policy."""
        with self._cache_lock:
            # LRU: remove oldest if cache is full
            if len(self._cache) >= Config.LLM.LLM_CACHE_MAX_ENTRIES():
                if self._cache_access_times:
                    oldest_key = min(self._cache_access_times, key=self._cache_access_times.get)
                else:
                    # Fallback to timestamp-based removal
                    oldest_key = min(self._cache, key=lambda k: self._cache[k]['timestamp'])
                self._cache.pop(oldest_key, None)
                self._cache_access_times.pop(oldest_key, None)

            current_time = time.time()
            self._cache[cache_key] = {
                'response': response,
                'timestamp': current_time
            }
            self._cache_access_times[cache_key] = current_time

    def get_service_status(self) -> Dict[str, Any]:
        """Get simple service status."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "service_available": self.api_available,
            "model": Config.LLM.OPENAI_MODEL(),
            "max_tokens": Config.LLM.OPENAI_MAX_TOKENS(),
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": f"{hit_rate:.1f}%"
        }
    
    async def aclose(self):
        """Close the OpenAI client. Called once at shutdown."""
        try:
            await self.async_openai_client.close()
        except Exception as e:
            logger.warning(f"Error closing OpenAI client: {e}")
        logger.info("ChatbotService cleanup completed")
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear the response cache."""
        with self._cache_lock:
            cache_size = len(self._cache)
            self._cache.clear()
            self._cache_access_times.clear()
        logger.info(f"🧹 Cache cleared: {cache_size} entries removed")
        return {
            "message": "Cache cleared successfully",
            "cleared_entries": cache_size,
            "current_cache_size": 0
        }

    def get_response_with_context(self, query: str, context: str, search_results: List[Dict[str, Any]] = None) -> Union[ChatResponse, ErrorResponse]:
        """
        Generate AI response using pre-built context (no RAG search).
        This ensures consistency with test RAG process.
        
        Args:
            query: User query to process
            context: Pre-built context from RAG process
            search_results: Search results metadata for confidence scoring

        Returns:
            Dict containing response, confidence metrics, and search metadata
        """
        if not self.api_available:
            error_response = self._create_error_response(query, "I apologize, but the chat service is currently unavailable. Please try again later.")
            return error_response

        try:
            logger.info(f" [Request] Received query: '{query[:50]}...' (id={hashlib.md5(query.encode()).hexdigest()[:8]})")
            logger.info(f"📄 Using pre-built context: {len(context)} characters")
            
            # Log context preview for debugging
            if context:
                context_preview = context[:200].replace('\n', ' ').replace('\r', ' ')
                logger.info(f"📄 Context preview: {context_preview}...")
            else:
                logger.warning(" No context provided")
            
            # Build OpenAI messages
            messages = [
                {"role": "system", "content": self.prompt_manager.get_system_prompt(context=context)}
            ]
            
            # Add current query
            messages.append({"role": "user", "content": query})

            # Check cache for speed
            cache_key = self._get_cache_key(query, context)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                logger.debug("⚡ Using cached response")
                # Calculate confidence for cached response
                confidence = self.confidence_scorer.calculate_confidence(
                    cached_response, query, context, search_results or []
                )
                return ChatResponse(
                    status=StatusEnum.SUCCESS,
                    message="Response generated successfully",
                    response=cached_response,
                    query=query,
                    confidence={
                        "score": confidence.overall_score,
                        "level": self.confidence_scorer.get_confidence_level(confidence.overall_score),
                        "details": {
                            "context_alignment": confidence.context_alignment,
                            "response_length_appropriateness": confidence.response_length_appropriateness,
                            "semantic_coherence": confidence.semantic_coherence,
                            "source_citation": confidence.source_citation,
                            "uncertainty_indicators": confidence.uncertainty_indicators,
                            "reasoning": confidence.reasoning
                        }
                    },
                    search_metadata={
                        "results_count": len(search_results) if search_results else 0,
                        "top_scores": [r.get("combined_score", 0) for r in (search_results or [])[:3] if r.get("combined_score", 0) > 0],
                        "cached_response": True
                    }
                )

            # Make optimized API call with better error handling
            logger.info(" Making API call to OpenAI...")
            try:
                client = openai.OpenAI(api_key=Config.LLM.OPENAI_API_KEY())
                response = client.chat.completions.create(
                    model=Config.LLM.OPENAI_MODEL(),
                    messages=messages,  # type: ignore
                    max_tokens=Config.LLM.OPENAI_MAX_TOKENS(),
                    temperature=Config.LLM.OPENAI_TEMPERATURE(),
                    timeout=Config.LLM.OPENAI_TIMEOUT()
                )
            except openai.APITimeoutError:
                logger.error("⏰ OpenAI API timeout - request took too long")
                error_response = self._create_error_response(query, "The AI service is taking too long to respond. Please try again with a simpler question.")
                return error_response
            except openai.RateLimitError:
                logger.error("🚫 OpenAI API rate limit exceeded")
                error_response = self._create_error_response(query, "The AI service is currently busy. Please try again in a moment.")
                return error_response
            except Exception as e:
                logger.error(f" OpenAI API error: {e}")
                error_response = self._create_error_response(query, f"AI service error: {str(e)}")
                return error_response

            content = response.choices[0].message.content
            response_text = content.strip() if content else ""
            logger.info(f" Response generated: {len(response_text)} characters")

            # Calculate confidence score
            confidence = self.confidence_scorer.calculate_confidence(
                response_text, query, context, search_results or []
            )
            
            logger.info(f" Confidence score: {confidence.overall_score:.3f} ({self.confidence_scorer.get_confidence_level(confidence.overall_score)})")

            # Cache the response for speed
            self._cache_response(cache_key, response_text)

            logger.debug(" Response generated successfully")
            return ChatResponse(
                status=StatusEnum.SUCCESS,
                message="Response generated successfully",
                response=response_text,
                query=query,
                confidence={
                    "score": confidence.overall_score,
                    "level": self.confidence_scorer.get_confidence_level(confidence.overall_score),
                    "details": {
                        "context_alignment": confidence.context_alignment,
                        "response_length_appropriateness": confidence.response_length_appropriateness,
                        "semantic_coherence": confidence.semantic_coherence,
                        "source_citation": confidence.source_citation,
                        "uncertainty_indicators": confidence.uncertainty_indicators,
                        "reasoning": confidence.reasoning
                    }
                },
                search_metadata={
                    "results_count": len(search_results) if search_results else 0,
                    "top_scores": [r.get("combined_score", 0) for r in (search_results or [])[:3] if r.get("combined_score", 0) > 0],
                    "cached_response": False
                }
            )

        except Exception as e:
            logger.error(f" Error generating response: {e}")
            return self._create_error_response(query, f"An error occurred while generating the response: {str(e)}")

    def get_response_with_history_and_context(self, query: str, context: str, search_results: List[Dict[str, Any]] = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate AI response using pre-built context with conversation history.
        This ensures consistency with test RAG process while supporting history.
        
        Args:
            query: User query to process
            context: Pre-built context from RAG process
            search_results: Search results metadata for confidence scoring
            history: Conversation history

        Returns:
            Dict containing response, confidence metrics, and search metadata
        """
        # History comes straight from the request body - strip anything that is
        # not a plain user/assistant turn before it reaches the model
        history = _sanitize_history(history)

        if not self.api_available:
            error_response = self._create_error_response(query, "I apologize, but the chat service is currently unavailable. Please try again later.")
            return error_response

        try:
            logger.info(f" Processing query with history: '{query[:50]}...'")
            logger.info(f" History length: {len(history) if history else 0} messages")
            logger.info(f"📄 Using pre-built context: {len(context)} characters")
            
            # Log context preview for debugging
            if context:
                context_preview = context[:200].replace('\n', ' ').replace('\r', ' ')
                logger.info(f"📄 Context preview: {context_preview}...")
            else:
                logger.warning(" No context provided")
            
            # Build OpenAI messages with history
            messages = [
                {"role": "system", "content": self.prompt_manager.get_system_prompt(context=context)}
            ]
            
            # Add conversation history if provided
            if history:
                # Limit history to prevent token overflow
                max_history_messages = min(len(history), 20)  # Limit to last 20 messages
                history_to_include = history[-max_history_messages:]
                messages.extend(history_to_include)
                logger.info(f" Included {len(history_to_include)} history messages")
            
            # Add current query
            messages.append({"role": "user", "content": query})

            # Check cache for speed (cache key includes history for uniqueness)
            cache_key = self._get_cache_key_with_history(query, context, history)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                logger.debug("⚡ Using cached response")
                # Calculate confidence for cached response
                confidence = self.confidence_scorer.calculate_confidence(
                    cached_response, query, context, search_results or []
                )
                return {
                    "response": cached_response,
                    "confidence": confidence.overall_score,
                    "confidence_level": self.confidence_scorer.get_confidence_level(confidence.overall_score),
                    "confidence_details": {
                        "context_alignment": confidence.context_alignment,
                        "response_length_appropriateness": confidence.response_length_appropriateness,
                        "semantic_coherence": confidence.semantic_coherence,
                        "source_citation": confidence.source_citation,
                        "uncertainty_indicators": confidence.uncertainty_indicators,
                        "reasoning": confidence.reasoning
                    },
                    "search_results": {
                        "count": len(search_results) if search_results else 0,
                        "top_scores": [r.get("combined_score", 0) for r in (search_results or [])[:3] if r.get("combined_score", 0) > 0]
                    },
                    "success": True,
                    "cached": True
                }

            # Make optimized API call
            logger.info(" Making API call to OpenAI...")
            client = openai.OpenAI(api_key=Config.LLM.OPENAI_API_KEY())
            response = client.chat.completions.create(
                model=Config.LLM.OPENAI_MODEL(),
                messages=messages,  # type: ignore
                max_tokens=Config.LLM.OPENAI_MAX_TOKENS(),
                temperature=Config.LLM.OPENAI_TEMPERATURE(),
                timeout=Config.LLM.OPENAI_TIMEOUT()
            )

            content = response.choices[0].message.content
            response_text = content.strip() if content else ""
            logger.info(f" Response generated: {len(response_text)} characters")

            # Calculate confidence score
            confidence = self.confidence_scorer.calculate_confidence(
                response_text, query, context, search_results or []
            )
            
            logger.info(f" Confidence score: {confidence.overall_score:.3f} ({self.confidence_scorer.get_confidence_level(confidence.overall_score)})")

            # Cache the response for speed
            self._cache_response(cache_key, response_text)

            logger.debug(" Response generated successfully")
            return {
                "response": response_text,
                "confidence": confidence.overall_score,
                "confidence_level": self.confidence_scorer.get_confidence_level(confidence.overall_score),
                "confidence_details": {
                    "context_alignment": confidence.context_alignment,
                    "response_length_appropriateness": confidence.response_length_appropriateness,
                    "semantic_coherence": confidence.semantic_coherence,
                    "source_citation": confidence.source_citation,
                    "uncertainty_indicators": confidence.uncertainty_indicators,
                    "reasoning": confidence.reasoning
                },
                "search_results": {
                    "count": len(search_results) if search_results else 0,
                    "top_scores": [r.get("combined_score", 0) for r in (search_results or [])[:3] if r.get("combined_score", 0) > 0]
                },
                "success": True,
                "cached": False
            }

        except Exception as e:
            logger.error(f" Error generating response: {e}")
            return {
                "response": f"An error occurred while generating the response: {str(e)}",
                "confidence": 0.0,
                "confidence_level": "Low",
                "confidence_details": {},
                "search_results": {"count": 0, "top_scores": []},
                "success": False,
                "cached": False
            }

    def _get_cache_key_with_history(self, query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate cache key including conversation history."""
        history_str = ""
        if history:
            # Create a hash of the history for the cache key
            history_content = "".join([msg.get("content", "") for msg in history])
            history_str = hashlib.md5(history_content.encode()).hexdigest()[:8]
        
        return hashlib.md5(f"{query}:{context}:{history_str}".encode()).hexdigest()

    async def stream_response_with_history(self, query: str, embeddings=None, documents=None, history: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[str, None]:
        """
        Async generator that streams response tokens from OpenAI with conversation history.
        Uses AsyncOpenAI for better performance and async context retrieval.
        Yields each token as it is generated.

        Failures are raised, not yielded: a token is indistinguishable from an answer to
        the caller, so an error text yielded here would be rendered as if the bot said it.
        """
        # Same filtering as the non-streaming path
        history = _sanitize_history(history)

        if not self.api_available:
            raise RuntimeError("Chat service unavailable: OpenAI client is not configured.")

        try:
            # Enrich query with conversation context if history provided
            search_query = query
            if history and len(history) > 0:
                enriched = self._enrich_query_with_history(query, history)
                if enriched != query:
                    search_query = enriched
                    logger.info(f"🔍 Using enriched query for streaming search: '{search_query[:100]}...'")
            
            # Async context retrieval if RAG enabled
            context = ""
            search_results: List[Dict[str, Any]] = []
            if documents is not None and embeddings is not None and len(documents) > 0:
                try:
                    # Use hybrid search with enriched query for better results
                    search_results = self.context_retriever.hybrid_search(
                        query=search_query,  # Use enriched query for better search
                        embeddings=embeddings,
                        documents=documents,
                        k=Config.RAG.RETRIEVAL_TOP_K(),
                        semantic_weight=Config.RAG.SEMANTIC_WEIGHT()
                    )
                    
                    # Build context from hybrid search results
                    context = _build_context(search_results, Config.LLM.MAX_CONTEXT_LENGTH())
                except Exception as e:
                    logger.warning(f" Hybrid search failed (streaming): {e}")
                    context = ""
            
            # Build messages with proper OpenAI types
            openai_messages: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": str(self.prompt_manager.get_system_prompt(context=context))}
            ]
            if history:
                max_history_messages = min(len(history), 20)
                history_to_include = history[-max_history_messages:]
                for msg in history_to_include:
                    # Already validated by _sanitize_history
                    openai_messages.append(cast(ChatCompletionMessageParam, msg))
            
            # Use enriched query for user message to provide better context to LLM
            # This ensures LLM understands what the user is asking about
            user_message_query = search_query if search_query != query else query
            openai_messages.append({"role": "user", "content": str(user_message_query)})

            # Use the shared AsyncOpenAI client for streaming
            response_stream = await self.async_openai_client.chat.completions.create(
                model=Config.LLM.OPENAI_MODEL(),
                messages=openai_messages,  # type: ignore
                max_tokens=Config.LLM.OPENAI_MAX_TOKENS(),
                temperature=Config.LLM.OPENAI_TEMPERATURE(),
                stream=True
            )
            async for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
        except Exception:
            # Log here, where the traceback still exists; the API turns this into a
            # generic message so exception details never reach the client.
            logger.exception(f"Streaming failed for query: {query[:100]}")
            raise

    def _enrich_query_with_history(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Intelligently enrich query with conversation context using semantic understanding.
        Analyzes query intent and conversation context to provide relevant enrichment.
        
        Args:
            query: Original user query
            history: Conversation history
            
        Returns:
            Semantically enriched query that maintains conversation context
        """
        if not history or len(history) == 0:
            return query
            
        try:
            # Analyze query intent
            query_intent = self._analyze_query_intent(query)
            
            # Extract conversation context
            context_info = self._extract_conversation_context(history)
            
            # RULE 1: Skip enrichment for new questions (specific entity queries)
            # E.g., "ai là CEO của TNT", "địa chỉ công ty ở đâu"
            if query_intent.get("type") == "new_question":
                logger.info(f"🆕 New question detected - skipping enrichment: '{query}'")
                logger.info(f"   Has entity: {query_intent.get('has_entity')}, Has WH-word: {query_intent.get('has_wh_word')}")
                return query
            
            # RULE 2: Check semantic similarity for follow-up queries
            # This catches topic changes even when intent detection is uncertain
            is_semantically_related = self._check_semantic_relevance(query, context_info)
            
            # RULE 3: For follow-up queries, require EITHER:
            # - Strong follow-up indicators (pronouns: nó, đó, này OR detail requests: cho tôi, biết thêm), OR
            # - Semantic similarity with context
            if query_intent.get("type") == "follow_up":
                strong_follow_up_indicators = [
                    # Pronouns (strongest signal)
                    "nó", "đó", "này", "chúng",
                    # Detail requests (strong signal)
                    "cho tôi", "cho biết", "biết thêm", "chi tiết hơn", 
                    "thông tin chi tiết", "cụ thể hơn"
                ]
                has_strong_follow_up = any(indicator in query.lower() for indicator in strong_follow_up_indicators)
                
                # If has strong indicator, always enrich (regardless of semantic similarity)
                if has_strong_follow_up:
                    logger.info(f"🔗 Strong follow-up indicator detected - enriching: '{query}'")
                elif not is_semantically_related:
                    # No strong indicator AND low semantic similarity = likely topic change
                    logger.info(f"🔀 Topic change detected (no strong indicator + low semantic) - skipping enrichment: '{query}'")
                    logger.info(f"   Query lacks strong follow-up indicators and low semantic similarity")
                    return query
            
            # RULE 4: Enrich if it's a follow-up AND (has pronoun OR semantically related)
            if query_intent.get("needs_context", False) and context_info.get("has_relevant_context", False):
                enriched_query = self._enrich_based_on_intent(query, query_intent, context_info)
                
                if enriched_query != query:
                    logger.info(f"🎯 Intent-based enrichment: '{query}' -> '{enriched_query}'")
                    logger.info(f"   Intent: {query_intent.get('type', 'unknown')}, Context: {context_info.get('main_topic', 'none')}")
                    return enriched_query
            
            return query
            
        except Exception as e:
            logger.warning(f" Query enrichment failed: {e}")
            return query
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze the intent of a user query to determine if it needs context enrichment.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dict containing intent analysis
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Specific product/service mentions make query specific
        specific_products = [
            "tn bidmate", "tnm auto", "tnm fin", "tn edu", "tnm edu",
            "phần mềm quản lý tour", "quản lý tour du lịch"
        ]
        
        has_specific_product = any(product in query_lower for product in specific_products)
        
        # Check if query contains WH-questions with specific entity
        # E.g., "ai là CEO", "địa chỉ ở đâu", "sản phẩm nào"
        wh_words = ["ai", "gì", "đâu", "nào", "bao nhiêu", "khi nào", "thế nào"]
        has_wh_word = any(wh in query_lower for wh in wh_words)
        
        # Check for specific nouns/entities that make query self-contained
        specific_entities = [
            # People & organization
            "ceo", "giám đốc", "sáng lập", "founder", "nhân viên", "team",
            "tnt", "công ty", "tringhiatech",
            # Places & contact
            "địa chỉ", "văn phòng", "office", "liên hệ", "contact",
            "email", "phone", "điện thoại", "hotline",
            # Products & services  
            "sản phẩm", "product", "dịch vụ", "service", "giải pháp", "solution",
            "phần mềm", "software", "ứng dụng", "app",
            # General info requests
            "giới thiệu", "introduction", "về", "about",
            "lịch sử", "history", "thành lập", "established"
        ]
        
        has_specific_entity = any(entity in query_lower for entity in specific_entities)
        
        # If query has WH-word + specific entity, it's likely a NEW question
        # E.g., "ai là CEO", "địa chỉ ở đâu", "sản phẩm nào"
        is_new_question = has_wh_word and has_specific_entity
        
        # Pure follow-up indicators (vague requests without specific entity)
        follow_up_indicators = [
            # Detail requests (vague)
            "thông tin chi tiết", "chi tiết hơn", "giải thích thêm",
            "thêm thông tin", "nói rõ hơn", "cụ thể hơn",
            "tất cả thông tin", "đầy đủ", "toàn bộ", "hoàn chỉnh",
            "biết thêm", "cho biết", "cho tôi",
            # Pronoun references (definitely follow-up)
            "nó", "chúng", "đó", "này",
            # Question phrases without entity
            "làm thế nào", "cách nào", "tại sao",
            "details", "more information", "explain"
        ]
        
        has_follow_up_indicator = any(indicator in query_lower for indicator in follow_up_indicators)
        
        # Very short queries without specific entity need context
        is_very_short = len(words) <= 3 and not has_specific_entity
        
        # Determine intent type
        intent_type = "unknown"
        if has_follow_up_indicator:
            intent_type = "follow_up"
        elif is_new_question:
            intent_type = "new_question"  # New question with specific entity
        elif has_specific_entity and not has_wh_word:
            # Query like "giới thiệu về TNT" - has entity but no WH word
            intent_type = "new_question"
        elif is_very_short:
            intent_type = "brief"
        
        return {
            "type": intent_type,
            "is_specific": has_specific_product or has_specific_entity,
            "needs_context": has_follow_up_indicator or is_very_short,
            "has_follow_up": has_follow_up_indicator,
            "has_wh_word": has_wh_word,
            "has_entity": has_specific_entity,
            "is_new_question": is_new_question,
            "is_brief": is_very_short,
            "word_count": len(words)
        }
    
    def _extract_conversation_context(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Extract relevant context information from conversation history.
        
        Args:
            history: Conversation history
            
        Returns:
            Dict containing context information
        """
        recent_history = history[-10:]  # Last 10 messages
        
        products = set()
        topics = set()
        last_assistant_response = ""
        
        # Product/service keywords
        product_keywords = {
            "tn bidmate": ["tn bidmate", "bidmate", "đấu thầu", "dược phẩm"],
            "tnm auto": ["tnm auto", "ô tô", "bán hàng", "hậu mãi"],
            "tnm fin": ["tnm fin", "tài chính", "kế toán", "quản lý tài chính"],
            "tn edu": ["tn edu", "tnm edu", "trường học", "giáo dục", "học sinh"]
        }
        
        # Topic keywords
        topic_keywords = [
            "thành công", "câu chuyện", "triển khai", "case study",
            "chuyển đổi số", "quản lý", "tự động hóa", "hiệu quả"
        ]
        
        for msg in recent_history:
            content = msg.get("content", "").lower()
            
            # Extract products
            for product, aliases in product_keywords.items():
                if any(alias in content for alias in aliases):
                    products.add(product)
            
            # Extract topics
            for topic in topic_keywords:
                if topic in content:
                    topics.add(topic)
            
            # Get last assistant response
            if msg.get("role") == "assistant":
                last_assistant_response = msg.get("content", "")
        
        # Determine main topic from last assistant response
        main_topic = "general"
        if last_assistant_response:
            for product, aliases in product_keywords.items():
                if any(alias in last_assistant_response.lower() for alias in aliases):
                    main_topic = product
                    break
        
        return {
            "products": products,
            "topics": topics,
            "main_topic": main_topic,
            "has_relevant_context": bool(products or topics),
            "last_assistant_response_length": len(last_assistant_response),
            "last_assistant_response": last_assistant_response[:500]  # Keep recent context for semantic check
        }
    
    def _check_semantic_relevance(self, query: str, context_info: Dict[str, Any]) -> bool:
        """
        Check if query is semantically related to conversation context using embeddings.
        
        Args:
            query: User query
            context_info: Extracted conversation context
            
        Returns:
            True if query is semantically related to context, False otherwise
        """
        try:
            # If no context, can't check relevance
            last_response = context_info.get("last_assistant_response", "")
            if not last_response or len(last_response) < 10:
                return True  # No context to compare, assume related
            
            # Get embeddings for query and last response
            from core.ai_services.embeddings.embeddings import EmbeddingService
            embedder = EmbeddingService()
            
            query_embedding = embedder.encode(query, convert_to_numpy=True)
            context_embedding = embedder.encode(last_response, convert_to_numpy=True)
            
            # Calculate cosine similarity
            import numpy as np
            
            # Flatten embeddings if they are 2D
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            if context_embedding.ndim > 1:
                context_embedding = context_embedding.flatten()
            
            similarity = np.dot(query_embedding, context_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding)
            )
            
            # Threshold for semantic relevance
            # Higher threshold (0.6) = stricter matching
            # Lower threshold (0.4) = more lenient
            SEMANTIC_THRESHOLD = 0.5
            
            is_related = similarity >= SEMANTIC_THRESHOLD
            
            logger.info(f"🔍 Semantic similarity: {similarity:.3f} (threshold: {SEMANTIC_THRESHOLD})")
            logger.info(f"   Query: '{query}'")
            logger.info(f"   Context: '{last_response[:100]}...'")
            logger.info(f"   Related: {'✅ YES' if is_related else '❌ NO'}")
            
            return is_related
            
        except Exception as e:
            logger.warning(f"⚠️ Semantic relevance check failed: {e}")
            # On error, default to True (be conservative)
            return True
    
    def _enrich_based_on_intent(self, query: str, intent: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Enrich query based on analyzed intent and conversation context.
        
        Args:
            query: Original query
            intent: Intent analysis result
            context: Context extraction result
            
        Returns:
            Enriched query
        """
        try:
            query_lower = query.lower().strip()
            
            # Build context string
            context_parts = []
            
            # Add main topic/product first
            if context.get("main_topic") and context["main_topic"] != "general":
                context_parts.append(context["main_topic"])
            
            # Add other mentioned products
            other_products = context.get("products", set()) - {context.get("main_topic")}
            if other_products:
                context_parts.extend(list(other_products))
            
            # Add topics
            if context.get("topics"):
                context_parts.extend(list(context["topics"]))
            
            context_str = " ".join(context_parts)
            
            # Enrich based on intent type
            if intent.get("type") == "follow_up":
                if "thông tin chi tiết" in query_lower or "chi tiết" in query_lower:
                    if context_str:
                        return f"thông tin chi tiết về {context_str}"
                        
                elif "giải thích" in query_lower:
                    if context_str:
                        return f"giải thích về {context_str}"
                        
                else:
                    # Generic follow-up
                    if context_str:
                        return f"{query} về {context_str}"
                        
            elif intent.get("type") == "question":
                # For questions, append context
                if context_str:
                    return f"{query} về {context_str}"
                    
            elif intent.get("type") == "brief":
                # For brief queries, intelligently combine
                if context_str:
                    return f"{query} {context_str}"
            
            # Enhanced fallback for queries seeking information
            if context_str:
                # Check for information-seeking patterns
                info_seeking_patterns = [
                    "tất cả", "toàn bộ", "đầy đủ", "hoàn chỉnh",
                    "thông tin", "chi tiết", "tính năng", "chức năng",
                    "all", "full", "complete", "information", "details"
                ]
                
                has_info_seeking = any(pattern in query_lower for pattern in info_seeking_patterns)
                
                # Check for pronouns (indicating reference to previous topic)
                pronouns = ["nó", "chúng", "đó", "này", "it", "they", "that", "this"]
                has_pronoun = any(pronoun in query_lower for pronoun in pronouns)
                
                # Enrich if: seeking info OR has pronoun OR query is short
                if has_info_seeking or has_pronoun or intent.get("word_count", 10) <= 5:
                    return f"{query} về {context_str}"
                
            return query
            
        except Exception as e:
            logger.warning(f" Intent-based enrichment failed: {e}")
            return query
