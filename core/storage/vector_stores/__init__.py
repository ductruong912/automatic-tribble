"""
Vector Storage Services
Handles vector embeddings storage and management.
"""

from .vector_store import VectorStore
from .vector_store_optimized import OptimizedVectorStore

__all__ = [
    "VectorStore",
    "OptimizedVectorStore",
]
