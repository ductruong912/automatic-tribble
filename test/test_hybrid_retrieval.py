"""
Unit tests for hybrid retrieval, reciprocal rank fusion, and BM25 tokenization.
"""

import unittest
import numpy as np
from core.retrieval.search.retriever import ContextRetriever


class TestHybridRetrieval(unittest.TestCase):
    """Test suite for hybrid retrieval scoring and ranking logic."""

    def test_reciprocal_rank_fusion_weighting(self):
        # 3 documents
        # Doc 0: high semantic, low keyword
        # Doc 1: low semantic, high keyword
        # Doc 2: zero both
        semantic = np.array([0.9, 0.1, 0.0])
        keyword = np.array([0.1, 0.9, 0.0])

        # Balanced weights (0.5 semantic, 0.5 keyword)
        fused_balanced = ContextRetriever._reciprocal_rank_fusion(
            semantic_scores=semantic,
            keyword_scores=keyword,
            semantic_weight=0.5,
            k_constant=60,
        )
        self.assertEqual(len(fused_balanced), 3)
        # Doc 0 and Doc 1 should tie under symmetric weights
        self.assertAlmostEqual(fused_balanced[0], fused_balanced[1], places=4)
        # Doc 2 should have strictly lowest score
        self.assertLess(fused_balanced[2], fused_balanced[0])

        # Semantic heavy (0.9 semantic, 0.1 keyword)
        fused_semantic = ContextRetriever._reciprocal_rank_fusion(
            semantic_scores=semantic,
            keyword_scores=keyword,
            semantic_weight=0.9,
            k_constant=60,
        )
        self.assertGreater(fused_semantic[0], fused_semantic[1])

        # Keyword heavy (0.1 semantic, 0.9 keyword)
        fused_keyword = ContextRetriever._reciprocal_rank_fusion(
            semantic_scores=semantic,
            keyword_scores=keyword,
            semantic_weight=0.1,
            k_constant=60,
        )
        self.assertGreater(fused_keyword[1], fused_keyword[0])

    def test_merge_reranked_preserves_guaranteed_hits(self):
        prelim = [
            {"index": 10, "score": 0.95},
            {"index": 20, "score": 0.90},
            {"index": 30, "score": 0.85},
        ]
        reranked = [
            {"index": 30, "rerank_score": 5.0},
            {"index": 40, "rerank_score": 4.0},
            {"index": 10, "rerank_score": 1.0},
        ]

        merged = ContextRetriever._merge_reranked(prelim, reranked, k=3, guaranteed=2)
        # guaranteed=2 pins first two prelim items (index 10 and 20)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["index"], 10)
        self.assertEqual(merged[1]["index"], 20)
        # third item should come from reranked without duplicate
        self.assertEqual(merged[2]["index"], 30)

    def test_retriever_cache_clearing(self):
        retriever = ContextRetriever.__new__(ContextRetriever)
        retriever._bm25_cache_key = "key1"
        retriever._bm25_instance = object()
        retriever._tokenized_docs = [["token1"]]
        retriever._query_cache = {"q": np.array([1.0])}
        retriever._query_preprocessing_cache = {"raw": ["tok"]}

        retriever.clear_cache()
        self.assertIsNone(retriever._bm25_cache_key)
        self.assertIsNone(retriever._bm25_instance)
        self.assertIsNone(retriever._tokenized_docs)
        self.assertEqual(retriever._query_cache, {})
        self.assertEqual(retriever._query_preprocessing_cache, {})


if __name__ == "__main__":
    unittest.main()
