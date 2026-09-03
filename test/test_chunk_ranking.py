#!/usr/bin/env python3
"""
Chunk Ranking Tester

A focused script to test and visualize chunk ranking and scoring in the RAG system.
Shows detailed scoring breakdown for semantic and keyword search components.

Usage:
    python -m test.test_chunk_ranking "your_query_here"
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

# Import project modules
from core.retrieval.search.retriever import ContextRetriever
from core.storage.vector_stores.vector_store_optimized import OptimizedVectorStore
from config.settings import Config


class ChunkRankingTester:
    """Test and visualize chunk ranking with detailed scoring."""
    
    def __init__(self):
        """Initialize the chunk ranking tester."""
        self.retriever = ContextRetriever()
        self.vector_store = OptimizedVectorStore()
        
        print(" Initializing Chunk Ranking Tester...")
        print(f" Configuration:")
        print(f"   - Semantic Weight: {Config.RAG.SEMANTIC_WEIGHT()}")
        print(f"   - Similarity Threshold: {Config.RAG.SIMILARITY_THRESHOLD()}")
        print(f"   - Retrieval Top K: {Config.RAG.RETRIEVAL_TOP_K()}")
        print()
    
    def load_data(self):
        """Load vector store data."""
        print(" Loading vector store...")
        try:
            faiss_index, embeddings, documents = self.vector_store.load_vector_store()
            
            if documents is None or len(documents) == 0:
                print("  No documents found in vector store!")
                return None, None
            
            print(f" Loaded {len(documents)} documents")
            return embeddings, documents
            
        except Exception as e:
            print(f" Error loading vector store: {e}")
            return None, None
    
    def test_ranking(self, query: str, embeddings, documents):
        """Test chunk ranking for a given query."""
        print(f" Testing ranking for: '{query}'")
        print("=" * 80)
        
        # Get search results
        search_results = self.retriever.hybrid_search(
            query=query,
            embeddings=embeddings,
            documents=documents,
            k=Config.RAG.RETRIEVAL_TOP_K(),
            semantic_weight=Config.RAG.SEMANTIC_WEIGHT()
        )
        
        if not search_results:
            print("  No chunks found above similarity threshold")
            print(f"    Current threshold: {Config.RAG.SIMILARITY_THRESHOLD()}")
            return
        
        print(f" Found {len(search_results)} relevant chunks:")
        print()
        
        # Display detailed ranking
        for i, result in enumerate(search_results, 1):
            print(f"🏆 Rank #{i}")
            print(f"   📄 Chunk Index: {result['index']}")
            print(f"    Combined Score: {result['combined_score']:.4f}")
            print(f"    Semantic Score: {result['semantic_score']:.4f}")
            print(f"   🔤 Keyword Score: {result['keyword_score']:.4f}")
            print(f"   📏 Length: {len(result['document'])} chars")
            print(f"   📝 Content Preview:")
            print(f"      {result['document'][:200]}...")
            print()
        
        # Show score distribution
        self._show_score_distribution(search_results)
        
        # Show top keywords
        self._show_keyword_analysis(query, search_results)
    
    def _show_score_distribution(self, results: List[Dict[str, Any]]):
        """Show score distribution analysis."""
        print("📈 Score Distribution Analysis:")
        print("-" * 40)
        
        combined_scores = [r['combined_score'] for r in results]
        semantic_scores = [r['semantic_score'] for r in results]
        keyword_scores = [r['keyword_score'] for r in results]
        
        print(f" Combined Scores:")
        print(f"   Min: {min(combined_scores):.4f}")
        print(f"   Max: {max(combined_scores):.4f}")
        print(f"   Avg: {sum(combined_scores)/len(combined_scores):.4f}")
        print()
        
        print(f" Semantic Scores:")
        print(f"   Min: {min(semantic_scores):.4f}")
        print(f"   Max: {max(semantic_scores):.4f}")
        print(f"   Avg: {sum(semantic_scores)/len(semantic_scores):.4f}")
        print()
        
        print(f"🔤 Keyword Scores:")
        print(f"   Min: {min(keyword_scores):.4f}")
        print(f"   Max: {max(keyword_scores):.4f}")
        print(f"   Avg: {sum(keyword_scores)/len(keyword_scores):.4f}")
        print()
    
    def _show_keyword_analysis(self, query: str, results: List[Dict[str, Any]]):
        """Show keyword analysis for the query."""
        print(" Keyword Analysis:")
        print("-" * 40)
        
        # Tokenize query
        query_tokens = self.retriever._tokenize(query)
        print(f"🔤 Query tokens: {query_tokens}")
        print()
        
        # Show which chunks contain which query terms
        for i, result in enumerate(results[:3], 1):  # Top 3 only
            chunk_tokens = self.retriever._tokenize(result['document'])
            matching_tokens = [token for token in query_tokens if token in chunk_tokens]
            
            print(f"📄 Chunk #{i} (Index: {result['index']}):")
            print(f"    Matching tokens: {matching_tokens}")
            print(f"    Match ratio: {len(matching_tokens)}/{len(query_tokens)} = {len(matching_tokens)/len(query_tokens):.2%}")
            print()


def main():
    """Main function."""
    print(" Chunk Ranking Tester")
    print("=" * 80)
    print()
    
    # Initialize tester
    tester = ChunkRankingTester()
    
    # Load data
    embeddings, documents = tester.load_data()
    if embeddings is None or documents is None:
        return
    
    # Get query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("💬 Enter your query:")
        query = input("> ").strip()
    
    if not query:
        print(" No query provided")
        return
    
    print()
    
    # Test ranking
    tester.test_ranking(query, embeddings, documents)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
