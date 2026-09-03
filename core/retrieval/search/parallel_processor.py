"""
Parallel processing utilities for RAG pipeline optimization.
Enables concurrent execution of independent operations to improve performance.
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Callable, Tuple
import numpy as np
import logging
from functools import partial

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """
    Handles parallel processing of independent RAG operations.
    Optimizes performance by running compatible operations concurrently.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    def __del__(self):
        """Clean up executor on destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    
    def parallel_embedding_and_bm25(
        self, 
        query: str, 
        documents: List[str],
        embedding_service,
        tokenize_func: Callable,
        bm25_instance
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run embedding generation and BM25 scoring in parallel.
        
        Args:
            query: Search query
            documents: List of documents
            embedding_service: Embedding service instance
            tokenize_func: Tokenization function
            bm25_instance: BM25 instance
            
        Returns:
            Tuple of (embeddings, bm25_scores)
        """
        def get_embeddings():
            return embedding_service.encode([query], convert_to_numpy=True)
        
        def get_bm25_scores():
            if bm25_instance:
                return bm25_instance.get_scores(tokenize_func(query))
            return np.zeros(len(documents))
        
        # Submit both tasks concurrently
        embedding_future = self.executor.submit(get_embeddings)
        bm25_future = self.executor.submit(get_bm25_scores)
        
        # Wait for both to complete
        embeddings = embedding_future.result()
        bm25_scores = bm25_future.result()
        
        return embeddings, bm25_scores
    
    def parallel_similarity_calculation(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        similarity_calculator,
        use_faiss: bool = False,
        faiss_index = None,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate similarity scores with optional FAISS optimization.
        
        Args:
            query_embedding: Query vector
            document_embeddings: Document vectors
            similarity_calculator: Similarity calculator instance
            use_faiss: Whether to use FAISS for faster search
            faiss_index: FAISS index if available
            k: Number of top results to return
            
        Returns:
            Tuple of (similarities, top_indices)
        """
        if use_faiss and faiss_index is not None:
            # Use FAISS for faster similarity search
            def faiss_search():
                return faiss_index.search(query_embedding.reshape(1, -1).astype('float32'), k)
            
            def cosine_similarity():
                return similarity_calculator.cosine_similarity(query_embedding, document_embeddings)
            
            # Run FAISS search and fallback cosine similarity in parallel
            faiss_future = self.executor.submit(faiss_search)
            cosine_future = self.executor.submit(cosine_similarity)
            
            try:
                # Try FAISS first
                faiss_scores, faiss_indices = faiss_future.result(timeout=1.0)
                return faiss_scores[0], faiss_indices[0]
            except Exception:
                # Fallback to cosine similarity
                cosine_scores = cosine_future.result()
                top_indices = cosine_scores.argsort()[-k:][::-1]
                return cosine_scores[top_indices], top_indices
        else:
            # Use regular cosine similarity
            similarities = similarity_calculator.cosine_similarity(query_embedding, document_embeddings)
            top_indices = similarities.argsort()[-k:][::-1]
            return similarities[top_indices], top_indices
    
    def parallel_context_processing(
        self,
        results: List[Dict[str, Any]],
        context_expansion: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process context expansion and metadata in parallel.
        
        Args:
            results: Search results to process
            context_expansion: Whether to expand context
            
        Returns:
            Processed results with expanded context
        """
        def process_single_result(result):
            # Add any additional processing here
            if context_expansion:
                # Simulate context expansion processing
                result['processed'] = True
            return result
        
        # Process results in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(results))) as executor:
            processed_results = list(executor.map(process_single_result, results))
        
        return processed_results
    
    def shutdown(self):
        """Shutdown the executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Global instance for easy access
_parallel_processor = None

def get_parallel_processor(max_workers: int = 4) -> ParallelProcessor:
    """Get or create the global parallel processor instance."""
    global _parallel_processor
    if _parallel_processor is None:
        _parallel_processor = ParallelProcessor(max_workers=max_workers)
    return _parallel_processor

def shutdown_parallel_processor():
    """Shutdown the global parallel processor."""
    global _parallel_processor
    if _parallel_processor is not None:
        _parallel_processor.shutdown()
        _parallel_processor = None
