"""
RAG Context Retriever - Hybrid search combining semantic and keyword-based approaches.
Handles document retrieval and result aggregation for enhanced search accuracy.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
import logging
from rank_bm25 import BM25Okapi
from ...ai_services.embeddings.embeddings import EmbeddingService, get_embedding_service
from ...storage.vector_stores.vector_store_optimized import OptimizedVectorStore
from ..similarity.similarity import SimilarityCalculator
from .reranker import Reranker
from config.settings import Config
import re
from utils.text_processing import TextUtils

logger = logging.getLogger(__name__)


class ContextRetriever:
    """
    Enhanced context retriever for RAG systems with hybrid search capabilities.
    Combines semantic embeddings and keyword search (BM25) for optimal retrieval.
    """

    def __init__(self, embedding_service=None, similarity_calculator=None, reranker=None,
                 vector_store_provider=None):
        # Initialize core components for search and similarity calculation
        if embedding_service is None:
            self.embedding_service = get_embedding_service()
        else:
            self.embedding_service = embedding_service

        self.vector_store = OptimizedVectorStore()  # Use optimized vector store with FAISS/HDF5
        # Callable returning (shared store, version); None keeps the private copy above
        self._store_provider = vector_store_provider
        self._store_version = None
        if similarity_calculator is None:
            self.similarity_calculator = SimilarityCalculator()
        else:
            self.similarity_calculator = similarity_calculator
        # Only initialize reranker if enabled; reuse the caller's to avoid a second model load
        if not Config.RAG.RERANKING_ENABLED():
            self.reranker = None
        elif reranker is not None:
            self.reranker = reranker
        else:
            self.reranker = Reranker()

        # Best-effort: load query adapter once at initialization
        try:
            self.embedding_service.load_query_adapter(Config.RAG.QUERY_ADAPTER_PATH())
        except Exception:
            pass

        # BM25 cache per loaded document corpus
        self._bm25_cache_key = None
        self._bm25_instance = None
        self._tokenized_docs = None
        self._query_cache = {}  # Cache for BM25 query results
        self._query_preprocessing_cache = {}  # Cache for query preprocessing

    def clear_cache(self):
        """Clear BM25 cache to ensure consistent search results."""
        self._bm25_cache_key = None
        self._bm25_instance = None
        self._tokenized_docs = None
        self._query_cache = {}

    def _ensure_numpy(self, embedding) -> np.ndarray:
        """Convert various embedding formats to numpy arrays for consistent processing."""
        if isinstance(embedding, list):
            return np.array(embedding)
        if hasattr(embedding, "numpy"):  # Handle PyTorch tensors
            return embedding.numpy()
        return np.array(embedding)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenization for BM25: lowercase, strip punctuation, remove diacritics with caching."""
        if not text:
            return []
        
        # Check cache first
        if text in self._query_preprocessing_cache:
            return self._query_preprocessing_cache[text]

        # Lowercase, remove Vietnamese accents, strip punctuation/symbols
        text = TextUtils.strip_vietnamese_accents(text.lower())
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = text.split()
        
        # Cache the result
        self._query_preprocessing_cache[text] = tokens
        
        # Limit cache size
        if len(self._query_preprocessing_cache) > 1000:
            # Remove oldest entries
            oldest_key = next(iter(self._query_preprocessing_cache))
            del self._query_preprocessing_cache[oldest_key]
        
        return tokens

    def _expand_context_with_adjacent_chunks(
        self, top_indices: List[int], total_documents: int, expansion_radius: int = 1
    ) -> List[int]:
        """
        Expand context by including adjacent chunks around the top results.
        Prioritize expanding the highest-scoring chunk first to ensure its
        neighbors are included within the maximum context cap, then expand
        remaining seeds in descending score order.
        Only include neighbors from the same source document.
        """
        if not Config.RAG.CONTEXT_EXPANSION_ENABLED():
            return top_indices

        # Get document metadata to check if chunks are from the same source
        # Ensure vector store is loaded to get metadata
        if not hasattr(self.vector_store, 'document_metadata') or self.vector_store.document_metadata is None:
            self.vector_store.load_vector_store()
        document_metadata = self.vector_store.get_metadata()
        if not document_metadata or len(document_metadata) != total_documents:
            # Fallback: expand purely by index adjacency if metadata is missing
            logger.warning(" Document metadata not available, using index-adjacent expansion")
            # Never below the number of ranked results
            max_chunks = max(Config.RAG.MAX_CONTEXT_CHUNKS(), len(top_indices))
            if max_chunks <= 0:
                return top_indices

            ordered: List[int] = []
            seen = set()

            def add_idx(i: int):
                if 0 <= i < total_documents and i not in seen and len(ordered) < max_chunks:
                    ordered.append(i)
                    seen.add(i)

            for seed in top_indices:
                add_idx(seed)

            for seed in top_indices:
                if len(ordered) >= max_chunks:
                    break
                # Add window [seed - r, ..., seed + r] in ascending order
                start = max(0, seed - expansion_radius)
                end = min(total_documents - 1, seed + expansion_radius)
                for i in range(start, end + 1):
                    add_idx(i)
                    if len(ordered) >= max_chunks:
                        break

            logger.info(
                f"Context expansion (adjacency fallback): {len(top_indices)} -> {len(ordered)} chunks"
            )
            return ordered

        # Never below the number of ranked results
        max_chunks = max(Config.RAG.MAX_CONTEXT_CHUNKS(), len(top_indices))
        if max_chunks <= 0:
            return top_indices

        # Maintain insertion order with a list, but guard uniqueness with a set
        ordered: List[int] = []
        seen = set()

        def add_idx(i: int):
            if 0 <= i < total_documents and i not in seen and len(ordered) < max_chunks:
                ordered.append(i)
                seen.add(i)

        # Group chunks by source to find proper adjacency within each source
        source_chunks = {}
        for i, metadata in enumerate(document_metadata):
            source_id = metadata.get("source_id", "unknown")
            if source_id not in source_chunks:
                source_chunks[source_id] = []
            source_chunks[source_id].append(i)

        # Every seed first: neighbours carry no score and must never displace a ranked hit
        for seed in top_indices:
            add_idx(seed)

        # Then spend whatever room is left on neighbours, within the same source only
        for seed in top_indices:
            if len(ordered) >= max_chunks:
                break

            seed_source = document_metadata[seed].get("source_id", "unknown")
            if seed_source not in source_chunks:
                continue

            source_chunk_indices = source_chunks[seed_source]
            try:
                seed_position_in_source = source_chunk_indices.index(seed)
            except ValueError:
                continue

            start_in_source = max(0, seed_position_in_source - expansion_radius)
            end_in_source = min(len(source_chunk_indices) - 1, seed_position_in_source + expansion_radius)

            for pos in range(start_in_source, end_in_source + 1):
                add_idx(source_chunk_indices[pos])
                if len(ordered) >= max_chunks:
                    break

        logger.info(
            f"Context expansion: {len(top_indices)} -> {len(ordered)} chunks"
        )
        return ordered

    def _ensure_vector_store_loaded(self) -> None:
        """Follow the shared store when its version changes, else load the private copy once."""
        if self._store_provider is not None:
            try:
                store, version = self._store_provider()
            except Exception as e:
                logger.warning(f" Shared vector store unavailable, keeping current copy: {e}")
                store, version = None, self._store_version
            if store is not None and version != self._store_version:
                self.vector_store = store
                self._store_version = version
                logger.info(f" Retriever now uses shared vector store version {version}")
        if getattr(self.vector_store, "embeddings", None) is not None:
            return
        try:
            self.vector_store.load_vector_store()
        except Exception as e:
            logger.warning(f" Could not load vector store for search: {e}")

    def _nothing_relevant(self, reranked: List[Dict[str, Any]]) -> bool:
        """True when the best cross-encoder score is below RERANK_MIN_SCORE."""
        floor = Config.RAG.RERANK_MIN_SCORE()
        if floor is None or not reranked or not self.reranker.available():
            return False
        # Heuristic fallback scores are on a different scale
        if not all(r.get("rerank_method") == "cross-encoder" for r in reranked):
            return False
        best = max(r.get("rerank_score", float("-inf")) for r in reranked)
        if best >= floor:
            return False
        logger.info(f" Best reranker score {best:.2f} < floor {floor:.2f}: no relevant chunk, empty context")
        return True

    @staticmethod
    def _merge_reranked(
        prelim: List[Dict[str, Any]],
        reranked: List[Dict[str, Any]],
        k: int,
        guaranteed: int = 2,
    ) -> List[Dict[str, Any]]:
        """Reorder by the reranker while pinning the top retrieval hits, which it may drop."""
        merged = list(prelim[:guaranteed])
        seen = {r["index"] for r in merged}

        for source in (reranked, prelim):
            for result in source:
                if len(merged) >= k:
                    return merged[:k]
                if result["index"] not in seen:
                    merged.append(result)
                    seen.add(result["index"])

        return merged[:k]

    @staticmethod
    def _reciprocal_rank_fusion(
        semantic_scores: np.ndarray,
        keyword_scores: np.ndarray,
        semantic_weight: float,
        k_constant: int = 60,
    ) -> np.ndarray:
        """Fuse the two rankings by rank rather than by score, scaled to [0, 1]."""
        semantic_scores = np.asarray(semantic_scores, dtype=float)
        keyword_scores = np.asarray(keyword_scores, dtype=float)
        total = len(semantic_scores)

        semantic_rank = np.empty(total, dtype=float)
        semantic_rank[np.argsort(semantic_scores)[::-1]] = np.arange(total)
        keyword_rank = np.empty(total, dtype=float)
        keyword_rank[np.argsort(keyword_scores)[::-1]] = np.arange(total)

        # A chunk neither retriever scored must not gain rank credit
        semantic_rank[semantic_scores <= 0] = total
        keyword_rank[keyword_scores <= 0] = total

        fused = (
            semantic_weight / (k_constant + semantic_rank + 1)
            + (1 - semantic_weight) / (k_constant + keyword_rank + 1)
        )
        return fused / (1.0 / (k_constant + 1))

    def _faiss_semantic_scores(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        keyword_scores: np.ndarray,
        total_documents: int,
        k: int,
    ) -> np.ndarray:
        """Semantic scores from FAISS, plus exact cosine for the chunks BM25 ranks highest."""
        scores = np.zeros(total_documents, dtype=float)

        top_similarities, top_indices = self.vector_store.fast_similarity_search(
            query_embedding, min(max(k, 1), total_documents)
        )
        for sim, idx in zip(top_similarities, top_indices):
            if 0 <= int(idx) < total_documents:
                scores[int(idx)] = float(sim)

        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] != total_documents:
            return scores

        # Chunks BM25 ranks highest but FAISS did not return
        keyword_scores = np.asarray(keyword_scores, dtype=float)
        pool = min(max(k * 10, 50), total_documents)
        candidates = [
            int(i)
            for i in np.argsort(keyword_scores)[::-1][:pool]
            if keyword_scores[i] > 0 and scores[i] == 0.0
        ]
        if candidates:
            scores[candidates] = self.similarity_calculator.cosine_similarity(
                query_embedding, embeddings[candidates]
            )

        return scores

    def hybrid_search(
        self,
        query: str,
        embeddings: np.ndarray,
        documents: List[str],
        k: int = 5,
        semantic_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Perform enhanced hybrid search combining semantic and keyword-based approaches.
        Only returns results above the similarity threshold, even if fewer than k.

        Args:
            query: Search query to process
            embeddings: Document embeddings for semantic similarity
            documents: List of documents to search through
            k: Number of top results to return
            semantic_weight: Weight for semantic search vs keyword search (0-1)

        Returns:
            List of dictionaries containing documents, scores, and metadata
        """
        logger.info(f"Starting hybrid search for query: '{query[:50]}...'")
        logger.info(f" Search parameters: k={k}, semantic_weight={semantic_weight}")
        logger.info(f" Available documents: {len(documents)}")

        if len(documents) == 0:
            logger.warning(" No documents available for search")
            return []

        try:
            # Load before scoring: the store used to load only during context expansion,
            # so the first query of a process ranked differently from every later one
            self._ensure_vector_store_loaded()

            # No source filtering — search across all documents
            # Use ORIGINAL query (with diacritics) for semantic embedding
            query_embedding = self._ensure_numpy(
                self.embedding_service.encode([query], convert_to_numpy=True)
            )
            try:
                query_embedding = self.embedding_service.apply_query_adapter(query_embedding)
            except Exception:
                pass

            # Keyword search using BM25 with enhanced caching
            try:
                cache_key = (len(documents), hash(documents[0]) if documents else 0)
            except Exception:
                cache_key = len(documents)

            # Check query cache first
            query_cache_key = f"{query}_{cache_key}"
            if query_cache_key in self._query_cache:
                keyword_scores = self._query_cache[query_cache_key]
            else:
                # Initialize BM25 if needed
                if self._bm25_cache_key != cache_key:
                    self._tokenized_docs = [self._tokenize(doc) for doc in documents]
                    self._bm25_instance = BM25Okapi(self._tokenized_docs)
                    self._bm25_cache_key = cache_key

                # Compute BM25 scores
                keyword_scores = self._bm25_instance.get_scores(self._tokenize(query)) if self._bm25_instance else np.zeros(len(documents))

                # Cache the result
                self._query_cache[query_cache_key] = keyword_scores

                # Limit query cache size
                if len(self._query_cache) > 500:
                    # Remove oldest entries
                    oldest_key = next(iter(self._query_cache))
                    del self._query_cache[oldest_key]

            if Config.RAG.USE_FAISS_INDEX() and getattr(self.vector_store, "faiss_index", None) is not None:
                logger.info("Using FAISS for semantic search")
                semantic_scores = self._faiss_semantic_scores(
                    query_embedding, embeddings, keyword_scores, len(documents), k
                )
            else:
                semantic_scores = self.similarity_calculator.cosine_similarity(
                    query_embedding, embeddings
                )

            # Normalize scores to [0, 1] for the relevance gate
            norm_semantic_scores = self.similarity_calculator.normalize_similarities(
                semantic_scores
            )
            norm_keyword_scores = self.similarity_calculator.normalize_similarities(
                keyword_scores
            )
            combined_scores = self._reciprocal_rank_fusion(
                semantic_scores, keyword_scores, semantic_weight
            )

            # Gate on the stronger of the two signals, so the blend weight cannot filter
            # out a chunk that one retriever matched perfectly
            best_signal = np.maximum(norm_semantic_scores, norm_keyword_scores)
            threshold = Config.RAG.SIMILARITY_THRESHOLD()
            filtered_indices = [
                i for i in range(len(documents)) if best_signal[i] >= threshold
            ]

            # Progressive threshold lowering if no results found
            if not filtered_indices:
                logger.info(f" No results above threshold {threshold}, trying progressive lowering...")
                thresholds_to_try = [0.7, 0.25, 0.15, 0.1, 0.05]

                for lower_threshold in thresholds_to_try:
                    if lower_threshold >= threshold:  # Skip if same or higher than original
                        continue

                    filtered_indices = [
                        i for i in range(len(documents)) if best_signal[i] >= lower_threshold
                    ]

                    if filtered_indices:
                        logger.info(f" Found {len(filtered_indices)} results with threshold {lower_threshold}")
                        break
                    else:
                        logger.debug(f" No results with threshold {lower_threshold}, trying lower...")

            # Create results with metadata (store both raw and normalized scores)
            results = []
            for i in filtered_indices:
                results.append(
                    {
                        "document": documents[i],
                        "semantic_score": float(semantic_scores[i]),
                        "keyword_score": float(keyword_scores[i]),
                        "norm_semantic_score": float(norm_semantic_scores[i]),
                        "norm_keyword_score": float(norm_keyword_scores[i]),
                        "combined_score": float(combined_scores[i]),
                        "index": i,
                    }
                )

            # Sort by combined score for initial ranking
            results.sort(key=lambda x: x["combined_score"], reverse=True)
            pool = k
            if Config.RAG.RERANKING_ENABLED() and self.reranker is not None:
                pool = max(getattr(self.reranker, "candidates", k * 4), k)
            prelim_top_k = results[:pool]

            top_k_results = prelim_top_k[:k]
            if Config.RAG.RERANKING_ENABLED() and self.reranker is not None:
                try:
                    reranked = self.reranker.rerank(query, prelim_top_k, k)
                    if self._nothing_relevant(reranked):
                        return []
                    top_k_results = self._merge_reranked(prelim_top_k, reranked, k)
                except Exception as e:
                    logger.warning(f" Reranking failed, using combined scores: {e}")

            # Apply context expansion to include adjacent chunks
            top_indices = [result["index"] for result in top_k_results]
            # Get document metadata for proper file boundary handling
            document_metadata = self.vector_store.get_metadata()
            expanded_indices = self._expand_context_with_adjacent_chunks(
                top_indices, len(documents), Config.RAG.CONTEXT_EXPANSION_RADIUS()
            )

            # Rebuild final results directly from expanded indices order, preserving original scores when available
            expanded_results = []
            index_to_result = {r["index"]: r for r in top_k_results}
            
            # Pre-allocate result structure for better performance
            default_result = {
                "semantic_score": 0.0,
                "keyword_score": 0.0,
                "combined_score": 0.0,
            }
            
            for idx in expanded_indices:
                base = index_to_result.get(idx)
                if base:
                    expanded_results.append(base)
                else:
                    # Create result with minimal string operations
                    result = default_result.copy()
                    result["document"] = documents[idx]
                    result["index"] = idx
                    expanded_results.append(result)

            logger.info(
                f" Hybrid search completed: {len(expanded_results)} results (expanded from {len(top_k_results)})"
            )
            return expanded_results

        except Exception as e:
            logger.warning(f" Hybrid search failed: {e}")
            return []

    def capture_user_feedback(
        self, query: str, relevant_docs: List[str], feedback: bool
    ):
        """Capture user feedback for future improvements."""
        if feedback:
            pass
        else:
            pass

    def debug_retrieval(
        self, query: str, embeddings: np.ndarray, documents: List[str], k: int = 5
    ) -> dict:
        """Debug retrieval process. If used for API, consider using a Pydantic model."""
        # TODO: Refactor to use a Pydantic model if this is exposed as an API response.
        logger.info(f"Debugging retrieval for query: '{query[:50]}...'")

        if len(documents) == 0:
            return {"error": "No documents available"}

        try:
            # Get semantic scores
            query_embedding = self._ensure_numpy(
                self.embedding_service.encode([query], convert_to_numpy=True)
            )
            semantic_scores = self.similarity_calculator.cosine_similarity(
                query_embedding, embeddings
            )

            # Get keyword scores
            tokenized_docs = [self._tokenize(doc) for doc in documents]
            bm25 = BM25Okapi(tokenized_docs)
            keyword_scores = bm25.get_scores(self._tokenize(query))

            # Normalize scores to [0, 1] before combining
            norm_semantic_scores = self.similarity_calculator.normalize_similarities(
                semantic_scores
            )
            norm_keyword_scores = self.similarity_calculator.normalize_similarities(
                keyword_scores
            )
            # Combine normalized scores
            combined_scores = (
                Config.RAG.SEMANTIC_WEIGHT() * norm_semantic_scores
                + (1 - Config.RAG.SEMANTIC_WEIGHT()) * norm_keyword_scores
            )

            # Filter by similarity threshold before sorting and selecting top K (use combined score)
            threshold = Config.RAG.SIMILARITY_THRESHOLD()
            filtered_indices = [
                i for i in range(len(documents)) if combined_scores[i] >= threshold
            ]

            # Create detailed results (store both raw and normalized scores)
            all_results = []
            for i in filtered_indices:
                all_results.append(
                    {
                        "index": i,
                        "document_preview": documents[i][:100].replace("\n", " "),
                        "semantic_score": float(semantic_scores[i]),
                        "keyword_score": float(keyword_scores[i]),
                        "norm_semantic_score": float(norm_semantic_scores[i]),
                        "norm_keyword_score": float(norm_keyword_scores[i]),
                        "combined_score": float(combined_scores[i]),
                        "above_threshold": float(combined_scores[i])
                        >= Config.RAG.SIMILARITY_THRESHOLD(),
                    }
                )

            # Sort by combined score for ranking
            all_results.sort(key=lambda x: x["combined_score"], reverse=True)
            # Get top k results (or fewer if not enough above threshold)
            top_k_results = all_results[:k]

            # Apply context expansion
            top_indices = [result["index"] for result in top_k_results]
            if isinstance(top_indices, np.ndarray):
                top_indices = top_indices.tolist()

            expanded_indices = self._expand_context_with_adjacent_chunks(
                top_indices, len(documents), Config.RAG.CONTEXT_EXPANSION_RADIUS()
            )

            # Get final results with expansion
            final_results = []
            for idx in expanded_indices:
                result = next((r for r in all_results if r["index"] == idx), None)
                if result:
                    final_results.append(result)

            return {
                "query": query,
                "total_documents": len(documents),
                "top_k_original": k,
                "final_chunks_retrieved": len(final_results),
                "threshold": Config.RAG.SIMILARITY_THRESHOLD(),
                "context_expansion_enabled": Config.RAG.CONTEXT_EXPANSION_ENABLED(),
                "expansion_radius": Config.RAG.CONTEXT_EXPANSION_RADIUS(),
                "all_scores": all_results,
                "top_k_results": top_k_results,
                "final_results": final_results,
                "expanded_indices": expanded_indices,
            }

        except Exception as e:
            logger.error(f" Debug retrieval failed: {e}")
            return {"error": str(e)}
