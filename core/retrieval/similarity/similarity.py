"""
Similarity calculation module for vector embeddings.
Provides multiple similarity metrics and utility functions for document retrieval.
"""

# Third-party imports
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional
import logging
from config.settings import Config

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """
    Handles similarity calculations between embeddings using multiple metrics.
    Provides utilities for ranking, filtering, and combining similarity scores.
    """

    def __init__(self):
        pass

    def cosine_similarity(
        self, query_embedding: np.ndarray, document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query and document embeddings.
        Optimized version using numpy operations for better performance.
        """
        # Normalize vectors for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        # Compute dot product (cosine similarity)
        return np.dot(doc_norms, query_norm.T).flatten()

    def euclidean_distance(
        self, query_embedding: np.ndarray, document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Euclidean distance between query and document embeddings.
        Converts distance to similarity score for consistent comparison.
        """
        # Calculate Euclidean distances between vectors
        distances = np.linalg.norm(document_embeddings - query_embedding, axis=1)
        # Convert to similarity (1 / (1 + distance)) - closer = more similar
        similarities = 1 / (1 + distances)
        return similarities

    def dot_product_similarity(
        self, query_embedding: np.ndarray, document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate dot product similarity between query and document embeddings.
        Alternative similarity metric for vector comparison.
        """
        return np.dot(document_embeddings, query_embedding.T).flatten()

    def get_top_k_indices(self, similarities: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Get indices of top k most similar documents.
        Returns sorted indices in descending order of similarity.
        """
        return similarities.argsort()[-k:][::-1]

    def get_top_k_similarities(self, similarities: np.ndarray, k: int = 3) -> tuple:
        """
        Get top k similarities and their corresponding indices.
        Returns both scores and indices for comprehensive ranking.
        """
        top_k_indices = self.get_top_k_indices(similarities, k)
        top_k_scores = similarities[top_k_indices]
        return top_k_indices, top_k_scores

    def filter_by_threshold(
        self, similarities: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Filter similarities by minimum threshold to ensure quality.
        Uses configuration threshold if none provided.
        """
        if threshold is None:
            threshold = Config.RAG.SIMILARITY_THRESHOLD()
        return np.where(similarities >= threshold)[0]

    def normalize_similarities(self, similarities: np.ndarray) -> np.ndarray:
        """
        Normalize similarities to 0-1 range for consistent comparison.
        Handles edge cases where all similarities are equal.
        """
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)

        if max_sim == min_sim:
            return np.ones_like(similarities)

        return (similarities - min_sim) / (max_sim - min_sim)

    def weighted_similarity(
        self, similarities: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Apply weights to similarity scores for custom ranking.
        Useful for combining multiple similarity metrics.
        """
        return similarities * weights

    def combine_similarities(self, *similarity_arrays, weights=None) -> np.ndarray:
        """
        Combine multiple similarity arrays with optional weights.
        Enables hybrid search combining different similarity metrics.
        """
        if weights is None:
            weights = np.ones(len(similarity_arrays)) / len(similarity_arrays)

        combined = np.zeros_like(similarity_arrays[0])
        for i, similarities in enumerate(similarity_arrays):
            combined += similarities * weights[i]

        return combined
