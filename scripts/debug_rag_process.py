#!/usr/bin/env python3
"""
RAG Process Debugging Script

This script demonstrates and inspects the complete RAG process:
1. Query processing and embedding generation
2. Hybrid search (semantic + keyword) with chunk ranking
3. Context construction from retrieved chunks
4. Complete prompt assembly for the LLM
5. Detailed scoring and ranking information

Usage:
    python scripts/debug_rag_process.py "your_query_here"
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
from core.retrieval.search.retriever import ContextRetriever
from core.ai_services.embeddings.embeddings import get_embedding_service
from core.storage.vector_stores.vector_store_optimized import OptimizedVectorStore
from core.ai_services.llm.prompts import PromptManager
from core.ai_services.llm.chatbot import ChatbotService
from config.settings import Config


class RAGProcessTester:
    """Comprehensive tester for the RAG pipeline."""
    
    def __init__(self):
        """Initialize the RAG process tester."""
        self.retriever = ContextRetriever()
        self.vector_store = OptimizedVectorStore()
        self.prompt_manager = PromptManager()
        self.embedding_service = get_embedding_service()
        self.chatbot_service = ChatbotService(context_retriever=self.retriever)
        
        print(" Initializing RAG Process Tester...")
        print(f" Configuration:")
        print(f"   - Embedding Model: {Config.RAG.EMBEDDING_MODEL()}")
        print(f"   - Retrieval Top K: {Config.RAG.RETRIEVAL_TOP_K()}")
        print(f"   - Semantic Weight: {Config.RAG.SEMANTIC_WEIGHT()}")
        print(f"   - Similarity Threshold: {Config.RAG.SIMILARITY_THRESHOLD()}")
        print(f"   - Max Context Chunks: {Config.RAG.MAX_CONTEXT_CHUNKS()}")
        print()
    
    def load_vector_store(self) -> tuple:
        """Load the vector store and return embeddings and documents."""
        print(" Loading vector store...")
        try:
            faiss_index, embeddings, documents = self.vector_store.load_vector_store()
            
            if documents is None or len(documents) == 0:
                print("  No documents found in vector store!")
                print("   Please upload some documents first using the /files/upload endpoint")
                return None, None
            
            print(f" Loaded {len(documents)} documents from vector store")
            print(f"   - Embeddings shape: {embeddings.shape if embeddings is not None else 'None'}")
            print(f"   - FAISS index: {'Available' if faiss_index is not None else 'Not available'}")
            print()
            
            return embeddings, documents
            
        except Exception as e:
            print(f" Error loading vector store: {e}")
            return None, None
    
    def process_query(self, query: str, embeddings, documents) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline."""
        print(f" Processing query: '{query}'")
        print("=" * 80)
        
        start_time = time.time()
        
        # Step 1: Generate query embedding
        print("1️⃣ Generating query embedding...")
        query_start = time.time()
        query_embedding = self.embedding_service.encode([query], convert_to_numpy=True)
        query_time = time.time() - query_start
        print(f"    Query embedding generated in {query_time:.3f}s")
        print(f"   📐 Embedding dimensions: {query_embedding.shape}")
        print()
        
        # Step 2: Perform hybrid search
        print("2️⃣ Performing hybrid search...")
        search_start = time.time()
        
        try:
            search_results = self.retriever.hybrid_search(
                query=query,
                embeddings=embeddings,
                documents=documents,
                k=Config.RAG.RETRIEVAL_TOP_K(),
                semantic_weight=Config.RAG.SEMANTIC_WEIGHT()
            )
            search_time = time.time() - search_start
            
            print(f"    Hybrid search completed in {search_time:.3f}s")
            print(f"    Found {len(search_results)} relevant chunks")
            print()
            
        except Exception as e:
            print(f"    Search failed: {e}")
            return {"error": str(e)}
        
        # Step 3: Display detailed ranking information
        print("3️⃣ Chunk Ranking Details:")
        print("-" * 80)
        
        if not search_results:
            print("     No chunks found above similarity threshold")
            print(f"    Current threshold: {Config.RAG.SIMILARITY_THRESHOLD()}")
            print()
        else:
            # Get document metadata for source information
            document_metadata = self.vector_store.get_metadata()
            
            for i, result in enumerate(search_results, 1):
                chunk_idx = result['index']
                source = document_metadata[chunk_idx].get("source_id", "unknown") if document_metadata else "unknown"
                
                print(f"   📄 Chunk #{i} (Index: {result['index']})")
                print(f"       Document Source: {source}")
                print(f"       Combined Score: {result['combined_score']:.4f}")
                print(f"       Semantic Score: {result['semantic_score']:.4f}")
                print(f"      🔤 Keyword Score: {result['keyword_score']:.4f}")
                print(f"      📏 Document Length: {len(result['document'])} chars")
                print(f"      📝 Preview: {result['document'][:100]}...")
                print()
        
        # Step 4: Construct context from chunks
        print("4️⃣ Context Construction:")
        print("-" * 80)
        
        if search_results:
            context_chunks = [result['document'] for result in search_results[:Config.RAG.MAX_CONTEXT_CHUNKS()]]
            constructed_context = "\n\n".join(context_chunks)
            print(f"   🧩 Used {len(context_chunks)} chunks for context")
            print(f"   📏 Total context length: {len(constructed_context)} characters")
            print(f"   📝 Context preview:\n{constructed_context[:200]}...")
            print()
        else:
            constructed_context = "No relevant context found."
            print("    No context available (no matching chunks)")
            print()
        
        # Step 5: Generate prompt
        print("5️⃣ Prompt Assembly:")
        print("-" * 80)
        
        prompt_messages = self.prompt_manager.create_chat_prompt(
            query=query,
            context=constructed_context
        )
        
        print(f"   📋 Generated {len(prompt_messages)} messages for LLM")
        for i, msg in enumerate(prompt_messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            print(f"    Message {i+1} ({role}): {len(content)} chars")
            print(f"      Preview: {content[:100]}...")
        print()
        
        # Step 6: Generate LLM response (optional)
        total_time = time.time() - start_time
        print(f"⏱️ Total pipeline time: {total_time:.3f}s")
        print("=" * 80)
        
        return {
            "query": query,
            "query_embedding_shape": query_embedding.shape,
            "search_results_count": len(search_results),
            "search_results": search_results,
            "context_length": len(constructed_context),
            "prompt_messages_count": len(prompt_messages),
            "total_time": total_time
        }


def main():
    """Main function."""
    print(" RAG Process Testing Script")
    print("=" * 80)
    print()
    
    # Initialize tester
    tester = RAGProcessTester()
    
    # Load vector store
    embeddings, documents = tester.load_vector_store()
    if embeddings is None or documents is None:
        return
    
    # Get query from arguments or prompt
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("💬 Enter your test query:")
        query = input("> ").strip()
    
    if not query:
        print(" No query provided")
        return
    
    print()
    
    # Process query
    results = tester.process_query(query, embeddings, documents)
    
    if "error" not in results:
        print(" RAG process completed successfully!")
    else:
        print(f"❌ RAG process failed: {results['error']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Process interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
