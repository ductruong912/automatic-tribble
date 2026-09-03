"""
Embedding Services
Handles text embedding generation and management.
"""

from .embeddings import EmbeddingService, get_embedding_service

__all__ = [
    "EmbeddingService",
    "get_embedding_service",
]
