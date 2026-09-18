"""
Simple, powerful chatbot service - RAG-powered conversation engine.
Handles query processing, context retrieval, and response generation with confidence scoring.
"""

import logging
import time
import asyncio
import hashlib
import re
import threading
from typing import Dict, Any, List, Optional, AsyncGenerator, Union, cast
from ...infrastructure.caching.cache_service import get_cache_service
import openai
from openai import AsyncOpenAI, APIStatusError, APIConnectionError
from openai.types.chat import ChatCompletionMessageParam

from config.settings import Config
from .prompts import PromptManager, SystemPrompts
from .token_budget import TokenBudgetManager
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
    Keep only well-formed user/assistant turns from client history.
    Rebuilds every message so no extra client-supplied key reaches the API.
    (Length caps and token budgeting are handled by TokenBudgetManager.)
    """
    if not history:
        return []

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
        clean.append({"role": role, "content": content})

    dropped = len(history) - len(clean)
    if dropped:
        logger.warning(f"Dropped {dropped} invalid message(s) from client history")

    return clean


# Follow-up phrasing and function words: what is left after removing them is the query's own topic
_FOLLOW_UP_PHRASES = [
    "giải thích", "tại sao", "vì sao", "làm thế nào", "làm sao", "như thế nào", "thế nào",
    "bằng cách nào", "cách nào", "thông tin", "chi tiết", "cụ thể", "nói rõ", "được không",
    "sử dụng", "đầy đủ", "toàn bộ", "tất cả", "hoàn chỉnh", "cho tôi", "cho biết", "biết thêm",
    "vui lòng",
]
_FILLER_WORDS = {
    "thêm", "hơn", "rõ", "không", "đi", "để", "về", "cho", "tôi", "mình", "biết", "với", "vậy",
    "thì", "là", "có", "được", "ạ", "nhé", "nữa", "gì", "dùng", "ra", "nào", "sao",
    "bạn", "ơi", "nhỉ", "thế", "à", "cần",
    "more", "details", "explain", "information", "how", "why", "what", "please", "the", "it",
    "this", "that", "about", "to", "use",
}


def _content_words(query: str) -> List[str]:
    """Words of the query that name its own subject, follow-up phrasing removed."""
    text = query.lower()
    for phrase in _FOLLOW_UP_PHRASES:
        text = re.sub(rf"(?<!\w){phrase}(?!\w)", " ", text)
    return [w for w in re.findall(r"\w+", text) if w not in _FILLER_WORDS]


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
        self.budget_manager = TokenBudgetManager()

        # Initialize async OpenAI client with optional base_url (for local LLM e.g. llama.cpp)
        base_url = Config.LLM.OPENAI_BASE_URL()
        api_key = Config.LLM.OPENAI_API_KEY() or ("local" if base_url else None)
        
        if api_key == "local" and not base_url:
            raise ValueError("Local LLM mode requires OPENAI_BASE_URL to be configured.")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.async_openai_client = AsyncOpenAI(**client_kwargs) if api_key else None

        # Assume usable when a key or local base_url is configured; probe_api() confirms it at startup
        self._api_available = bool(Config.LLM.OPENAI_API_KEY() or Config.LLM.OPENAI_BASE_URL())
        if not self._api_available:
            logger.warning(" Neither OpenAI API key nor OPENAI_BASE_URL configured")

    def _get_sync_client(self) -> openai.OpenAI:
        """Get synchronous OpenAI client configured for Cloud or Local LLM."""
        base_url = Config.LLM.OPENAI_BASE_URL()
        api_key = Config.LLM.OPENAI_API_KEY() or ("local" if base_url else None)
        
        if api_key == "local" and not base_url:
            raise ValueError("Local LLM mode requires OPENAI_BASE_URL to be configured.")
            
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        return openai.OpenAI(**client_kwargs)

    @property
    def api_available(self) -> bool:
        """Check if the LLM API is available."""
        return self._api_available

    def probe_api(self) -> bool:
        """
        Check the LLM API once and record the result.
        Called at startup - never from __init__, so building the service costs no network round trip.
        """
        self._api_available = self._test_api()
        return self._api_available

    def _test_api(self) -> bool:
        """
        Test LLM API availability (Cloud OpenAI or Local OpenAI-compatible server).
        Treats HTTP errors (4xx) as "reachable" to allow startup,
        while catching genuine connection errors.
        """
        if not Config.LLM.OPENAI_API_KEY() and not Config.LLM.OPENAI_BASE_URL():
            logger.warning(" Neither OpenAI API key nor OPENAI_BASE_URL configured")
            return False
        try:
            client = self._get_sync_client()
            client.chat.completions.create(
                model=Config.LLM.OPENAI_MODEL(),
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            provider_desc = f"Local LLM ({Config.LLM.OPENAI_BASE_URL()})" if Config.LLM.OPENAI_BASE_URL() else "OpenAI Cloud"
            logger.info(f" {provider_desc} API available and responding.")
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

    def _get_cache_key(self, query: str, context: str) -> str:
        """Generate MD5-based cache key from query and context for efficient caching."""
        content = f"query:{query.lower().strip()}\ncontext:{context}"
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
            if self.async_openai_client:
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
                client = self._get_sync_client()
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
                logger.error(f" OpenAI API error: {e}", exc_info=True)
                error_response = self._create_error_response(
                    query,
                    "An error occurred while processing your request with the AI service. Please try again later."
                )
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
            logger.error(f" Error generating response: {e}", exc_info=True)
            return self._create_error_response(query, "An error occurred while generating the response. Please try again later.")

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
        # We only do structural sanitization here.
        # Budgeting and truncation are handled by TokenBudgetManager.
        history = _sanitize_history(history)

        if not self.api_available:
            error_response = self._create_error_response(query, "I apologize, but the chat service is currently unavailable. Please try again later.")
            return error_response

        try:
            logger.info(f" Processing query with history: '{query[:50]}...'")
            
            # Centralized budgeting calculation
            base_system = self.prompt_manager.get_system_prompt(context="")
            try:
                formatted_rag_context, retained_history, safe_max_tokens = self.budget_manager.build_prompt_and_budget(
                    base_system_prompt=base_system,
                    query=query,
                    search_results=search_results or [],
                    history=history or [],
                    context_str=context
                )
            except ValueError as ve:
                logger.error(f" Budgeting error: {ve}")
                return self._create_error_response(query, str(ve))
                
            # Log context preview for debugging
            if formatted_rag_context:
                context_preview = formatted_rag_context[:200].replace('\n', ' ').replace('\r', ' ')
                logger.info(f"📄 Context preview: {context_preview}...")
            else:
                logger.warning(" No context provided or context dropped due to budget")
            
            # Build final messages
            messages = [
                {"role": "system", "content": self.prompt_manager.get_system_prompt(context=formatted_rag_context)}
            ]
            
            if retained_history:
                messages.extend(retained_history)
                logger.info(f" Included {len(retained_history)} history messages (after budget)")
            
            messages.append({"role": "user", "content": query})

            # Check cache for speed (cache key includes history for uniqueness)
            cache_key = self._get_cache_key_with_history(query, formatted_rag_context, retained_history)
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
            logger.info(f" Making API call to OpenAI... (safe_max_tokens: {safe_max_tokens})")
            client = self._get_sync_client()
            response = client.chat.completions.create(
                model=Config.LLM.OPENAI_MODEL(),
                messages=messages,  # type: ignore
                max_tokens=safe_max_tokens,
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
            logger.error(f" Error generating response: {e}", exc_info=True)
            return {
                "response": "An error occurred while generating the response. Please try again later.",
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
                except Exception as e:
                    logger.warning(f" Hybrid search failed (streaming): {e}")
            
            user_message_query = search_query if search_query != query else query
            
            # Use TokenBudgetManager to construct prompt safely
            base_system = self.prompt_manager.get_system_prompt(context="")
            try:
                formatted_rag_context, retained_history, safe_max_tokens = self.budget_manager.build_prompt_and_budget(
                    base_system_prompt=base_system,
                    query=user_message_query,
                    search_results=search_results,
                    history=history or []
                )
            except ValueError as ve:
                logger.error(f" Budgeting error in streaming: {ve}")
                yield f"Error: {str(ve)}"
                return

            # Build messages with proper OpenAI types
            openai_messages: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": str(self.prompt_manager.get_system_prompt(context=formatted_rag_context))}
            ]
            if retained_history:
                for msg in retained_history:
                    openai_messages.append(cast(ChatCompletionMessageParam, msg))
            
            openai_messages.append({"role": "user", "content": str(user_message_query)})

            # Use the shared AsyncOpenAI client for streaming
            response_stream = await self.async_openai_client.chat.completions.create(
                model=Config.LLM.OPENAI_MODEL(),
                messages=openai_messages,  # type: ignore
                max_tokens=safe_max_tokens,
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
        Domain-agnostic query enrichment using conversation history.
        Enriches follow-ups and brief queries with the preceding topic without
        overwriting the user's keywords or relying on hardcoded business entities.
        
        Args:
            query: Original user query
            history: Conversation history
            
        Returns:
            Enriched query that maintains conversation context
        """
        if not history or len(history) == 0:
            return query
            
        try:
            intent = self._analyze_query_intent(query)
            
            # If self-contained question with its own subject, search as asked
            if intent.get("is_self_contained", False):
                logger.info(f"🆕 Self-contained query detected - skipping enrichment: '{query}'")
                return query
                
            # If query needs context (follow-up, pronoun, or brief)
            if intent.get("needs_context", False):
                subject = self._last_subject_question(history)
                if subject:
                    clean_subject = subject.strip()
                    if len(clean_subject) > 120:
                        clean_subject = clean_subject[:120].strip() + "..."
                    enriched_query = f"{query} ({clean_subject})"
                    logger.info(f"🔗 Contextual enrichment: '{query}' -> '{enriched_query}'")
                    return enriched_query
            
            return query
            
        except Exception as e:
            logger.warning(f" Query enrichment failed: {e}")
            return query

    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query intent purely through linguistic patterns (domain-agnostic).
        Identifies follow-up signals, pronouns, and substantive content words.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dict containing linguistic intent analysis
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        content_words = _content_words(query)
        
        # Referring pronouns in Vietnamese & English
        pronouns = [
            "nó", "đó", "này", "chúng", "kia", "ấy", "đây",
            "it", "they", "them", "this", "that", "these", "those"
        ]
        has_pronoun = any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", query_lower) for p in pronouns)
        
        # Common follow-up and detail request phrasing
        follow_up_phrases = [
            "chi tiết hơn", "cụ thể hơn", "rõ hơn", "nói rõ", "giải thích thêm",
            "thêm thông tin", "cho tôi biết", "cho biết thêm", "thế nào", "như thế nào",
            "bao nhiêu", "làm sao", "tại sao", "được không", "có được không",
            "more details", "details", "explain more", "how much", "tell me more"
        ]
        has_follow_up = any(phrase in query_lower for phrase in follow_up_phrases)
        
        # Greetings / small talk
        greetings = ["chào", "xin chào", "hello", "hi", "cảm ơn", "thank", "tạm biệt", "bye"]
        is_greeting = len(words) <= 4 and any(re.search(rf"(?<!\w){re.escape(g)}(?!\w)", query_lower) for g in greetings)
        
        # Short / brief query
        is_brief = len(content_words) <= 2
        
        # Self-contained if has 3+ content words, no referring pronouns, and not purely a follow-up phrase
        is_self_contained = len(content_words) >= 3 and not has_pronoun and not has_follow_up
        
        # Needs context if it has referring pronoun, follow-up phrase, or is very brief (and not just greeting)
        needs_context = (has_pronoun or has_follow_up or is_brief) and not is_greeting
        
        intent_type = "self_contained" if is_self_contained else ("follow_up" if needs_context else "general")
        
        return {
            "type": intent_type,
            "has_pronoun": has_pronoun,
            "has_follow_up": has_follow_up,
            "is_brief": is_brief,
            "is_greeting": is_greeting,
            "is_self_contained": is_self_contained,
            "needs_context": needs_context,
            "content_words_count": len(content_words),
            "word_count": len(words)
        }

    @staticmethod
    def _last_subject_question(history: List[Dict[str, str]]) -> Optional[str]:
        """Most recent user question that names a subject, skipping earlier vague follow-ups."""
        for msg in reversed(history):
            if msg.get("role") == "user" and _content_words(msg.get("content", "")):
                return msg["content"].strip()[:300]
        return None
