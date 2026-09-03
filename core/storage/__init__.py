"""
Storage Domain
Handles vector storage, document metadata, and data persistence.
"""

from .vector_stores.vector_store import VectorStore
from .vector_stores.vector_store_optimized import OptimizedVectorStore
from .metadata_stores.document_store import DocumentStore
from .metadata_stores.metadata_store import MetadataStore

__all__ = [
    "VectorStore",
    "OptimizedVectorStore",
    "DocumentStore", 
    "MetadataStore",
]