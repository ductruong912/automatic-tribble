"""
Infrastructure Domain
Handles caching, performance monitoring, and system infrastructure.
"""

from .caching.cache_service import SmartCacheService, get_cache_service

__all__ = [
    "SmartCacheService",
    "get_cache_service",
]
