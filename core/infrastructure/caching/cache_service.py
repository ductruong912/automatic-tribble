"""
Smart caching service for responses and embeddings to improve performance.
Provides intelligent caching with TTL, LRU eviction, and query similarity matching.
"""

import time
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import json
import pickle
import os
from pathlib import Path

# Local imports
from config.settings import Config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent eviction."""
    value: Any
    timestamp: float
    access_count: int
    last_accessed: float
    query_embedding: Optional[Any] = None
    confidence_score: Optional[float] = None


class SmartCacheService:
    """
    Intelligent caching service with multiple cache layers and smart eviction.
    Provides semantic similarity matching for chat responses and embedding caching.
    """

    def __init__(self, max_size: int = None, ttl_seconds: int = None):
        """
        Initialize smart cache service.
        
        Args:
            max_size: Maximum number of entries (default from config)
            ttl_seconds: Time to live in seconds (default from config)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size or Config.LLM.LLM_CACHE_MAX_SIZE()
        self._ttl_seconds = ttl_seconds or Config.LLM.LLM_CACHE_TTL()
        self._hits = 0
        self._misses = 0
        
        # Embedding cache for faster similarity matching
        self._embedding_cache: Dict[str, Any] = {}
        
        # Persistent cache file
        self._cache_file = os.path.join(Config.File.TEMP_DIR(), "smart_cache.pkl")
        self._load_persistent_cache()
        
        logger.info(f" SmartCacheService initialized: max_size={self._max_size}, ttl={self._ttl_seconds}s")

    def _generate_cache_key(self, query: str, context_hash: str = "") -> str:
        """Generate a deterministic cache key for query and context."""
        combined = f"{query}|{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - entry.timestamp) > self._ttl_seconds

    def _evict_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if (current_time - entry.timestamp) > self._ttl_seconds
        ]
        
        for key in expired_keys:
            del self._cache[key]
            
        if expired_keys:
            logger.debug(f"⏰ Evicted {len(expired_keys)} expired cache entries")

    def _evict_lru(self):
        """Evict least recently used entries when cache is full."""
        if len(self._cache) <= self._max_size:
            return
            
        # Sort by last accessed time and access count
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].last_accessed, x[1].access_count)
        )
        
        # Remove oldest entries
        entries_to_remove = len(self._cache) - self._max_size + 1
        for i in range(entries_to_remove):
            key, _ = sorted_entries[i]
            del self._cache[key]
            
        logger.debug(f" Evicted {entries_to_remove} LRU cache entries")

    def _find_similar_query(self, query: str, similarity_threshold: float = 0.9) -> Optional[str]:
        """
        Find cached queries that are semantically similar to the current query.
        Returns cache key if similar query found, None otherwise.
        """
        try:
            from core.ai_services.embeddings.embeddings import get_embedding_service
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Get embedding for current query
            query_key = f"query_emb_{hashlib.md5(query.encode()).hexdigest()}"
            if query_key not in self._embedding_cache:
                embedding_service = get_embedding_service()
                query_embedding = embedding_service.encode([query], convert_to_numpy=True)
                self._embedding_cache[query_key] = query_embedding[0]
            else:
                query_embedding = self._embedding_cache[query_key]
            
            # Check similarity with cached queries
            for cache_key, entry in self._cache.items():
                if entry.query_embedding is not None:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        entry.query_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity >= similarity_threshold:
                        logger.debug(f" Found similar cached query (similarity: {similarity:.3f})")
                        return cache_key
                        
        except Exception as e:
            logger.debug(f" Similarity search failed: {e}")
            
        return None

    def set(self, query: str, response: Any, context_hash: str = "", 
            confidence_score: float = None) -> str:
        """
        Store response in cache with intelligent metadata.
        
        Args:
            query: User query
            response: Response to cache
            context_hash: Hash of context used for response
            confidence_score: Confidence score of the response
            
        Returns:
            Cache key for the stored entry
        """
        cache_key = self._generate_cache_key(query, context_hash)
        current_time = time.time()
        
        # Generate query embedding for similarity matching
        query_embedding = None
        try:
            from core.ai_services.embeddings.embeddings import get_embedding_service
            embedding_service = get_embedding_service()
            query_embedding = embedding_service.encode([query], convert_to_numpy=True)[0]
        except Exception as e:
            logger.debug(f" Failed to generate query embedding: {e}")
        
        # Create cache entry
        entry = CacheEntry(
            value=response,
            timestamp=current_time,
            access_count=0,
            last_accessed=current_time,
            query_embedding=query_embedding,
            confidence_score=confidence_score
        )
        
        # Clean up before adding
        self._evict_expired()
        self._evict_lru()
        
        # Store in cache
        self._cache[cache_key] = entry
        
        logger.debug(f"💾 Cached response for query: {query[:50]}... (key: {cache_key[:8]}...)")
        return cache_key

    def get(self, query: str, context_hash: str = "", 
            use_similarity: bool = True) -> Optional[Any]:
        """
        Retrieve response from cache with similarity matching.
        
        Args:
            query: User query
            context_hash: Hash of context
            use_similarity: Whether to use semantic similarity matching
            
        Returns:
            Cached response if found, None otherwise
        """
        # First try exact match
        cache_key = self._generate_cache_key(query, context_hash)
        
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            
            if self._is_expired(entry):
                del self._cache[cache_key]
                self._misses += 1
                return None
            
            # Update access metadata
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            self._hits += 1
            logger.debug(f" Cache hit (exact): {query[:50]}...")
            return entry.value
        
        # Try similarity matching if enabled
        if use_similarity:
            similar_key = self._find_similar_query(query)
            if similar_key and similar_key in self._cache:
                entry = self._cache[similar_key]
                
                if self._is_expired(entry):
                    del self._cache[similar_key]
                    self._misses += 1
                    return None
                
                # Update access metadata
                entry.access_count += 1
                entry.last_accessed = time.time()
                
                self._hits += 1
                logger.debug(f" Cache hit (similar): {query[:50]}...")
                return entry.value
        
        self._misses += 1
        return None

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._embedding_cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("🧹 Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "entries": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "embedding_cache_size": len(self._embedding_cache)
        }

    def _load_persistent_cache(self):
        """Load cache from persistent storage."""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self._cache = data.get('cache', {})
                    self._embedding_cache = data.get('embedding_cache', {})
                    
                # Clean expired entries on load
                self._evict_expired()
                
                logger.info(f"📂 Loaded {len(self._cache)} cache entries from persistent storage")
        except Exception as e:
            logger.warning(f" Failed to load persistent cache: {e}")
            self._cache = {}
            self._embedding_cache = {}

    def save_persistent_cache(self):
        """Save cache to persistent storage."""
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            
            # Only save recent, high-confidence entries
            filtered_cache = {}
            current_time = time.time()
            
            for key, entry in self._cache.items():
                # Keep entries that are recent or frequently accessed
                age_hours = (current_time - entry.timestamp) / 3600
                if (age_hours < 24 and entry.access_count > 0) or entry.access_count > 3:
                    filtered_cache[key] = entry
            
            data = {
                'cache': filtered_cache,
                'embedding_cache': self._embedding_cache
            }
            
            with open(self._cache_file, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"💾 Saved {len(filtered_cache)} cache entries to persistent storage")
            
        except Exception as e:
            logger.warning(f" Failed to save persistent cache: {e}")

    def cleanup_on_shutdown(self):
        """Cleanup method to be called on application shutdown."""
        self.save_persistent_cache()
        logger.info("🛑 Cache service shutdown cleanup completed")


# Global cache service instance
_cache_service_instance = None


def get_cache_service() -> SmartCacheService:
    """Get global cache service instance."""
    global _cache_service_instance
    if _cache_service_instance is None:
        _cache_service_instance = SmartCacheService()
    return _cache_service_instance


def clear_cache():
    """Clear global cache."""
    service = get_cache_service()
    service.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    service = get_cache_service()
    return service.get_stats()
