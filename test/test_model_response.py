#!/usr/bin/env python3
"""
Simple Model Response Tester

A focused script to test the complete RAG pipeline and show the model's response.
Shows the query, retrieved context, and the final model response.

Usage:
    python -m test.test_model_response "your_query_here"
"""

import os
import sys
import time
import logging
import numpy as np
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
from core.ai_services.llm.chatbot import ChatbotService


class ModelResponseTester:
    """Simple tester for model responses with RAG."""
    
    def __init__(self):
        """Initialize the model response tester."""
        self.retriever = ContextRetriever()
        # Preload retriever's vector store to ensure metadata is available
        try:
            self.retriever.vector_store.load_vector_store()
        except Exception:
            pass
        self.vector_store = OptimizedVectorStore()
        self.chatbot_service = ChatbotService(context_retriever=self.retriever)
        
        print(" Initializing Model Response Tester...")
        print()
    
    def load_data(self):
        """Load vector store data."""
        print(" Loading vector store...")
        try:
            faiss_index, embeddings, documents = self.vector_store.load_vector_store()
            
            if documents is None or len(documents) == 0:
                print("  No documents found in vector store!")
                print("💡 This is normal if no files have been uploaded yet.")
                print("   The system will still work but responses may be less accurate.")
                # Return empty arrays instead of None to allow the test to continue
                return np.array([]), []
            
            print(f" Loaded {len(documents)} documents")
            return embeddings, documents
            
        except Exception as e:
            print(f" Error loading vector store: {e}")
            return None, None
    
    def test_response(self, query: str, embeddings, documents):
        """Test model response for a given query."""
        print(f" Query: '{query}'")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Get response from chatbot service
            response = self.chatbot_service.get_response(
                query=query,
                embeddings=embeddings,
                documents=documents
            )
            
            processing_time = time.time() - start_time
            
            print(f"⏱️  Processing time: {processing_time:.3f}s")
            print()
            
            # Show the response
            print(" MODEL RESPONSE:")
            print("-" * 80)
            print(response.response if hasattr(response, 'response') else 'No response generated')
            print("-" * 80)
            print()
            
            # Show confidence if available
            if hasattr(response, 'confidence') and response.confidence:
                confidence = response.confidence
                print(f" Confidence: {confidence.get('score', 'N/A')} ({confidence.get('level', 'N/A')})")
                print()
            
            # Show search metadata if available
            if hasattr(response, 'search_metadata') and response.search_metadata:
                search_meta = response.search_metadata
                print(f" Search Results: {search_meta.get('results_count', 'N/A')} chunks found")
                if 'top_scores' in search_meta:
                    scores = search_meta['top_scores']
                    print(f" Top Scores: {[f'{s:.3f}' for s in scores[:3]]}")
                print()
            
            return response
            
        except Exception as e:
            print(f" Error generating response: {e}")
            return {"error": str(e)}


def main():
    """Main function."""
    print(" Model Response Tester")
    print("=" * 80)
    print()
    
    # Initialize tester
    tester = ModelResponseTester()
    
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
    
    # Test response
    response = tester.test_response(query, embeddings, documents)
    
    # Interactive mode
    if len(sys.argv) == 1:  # Only if not called with command line args
        while True:
            print("💬 Enter another query (or 'quit' to exit):")
            query = input("> ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print()
            tester.test_response(query, embeddings, documents)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
