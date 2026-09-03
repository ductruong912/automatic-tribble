#!/usr/bin/env python3
"""
RAG Process Testing Script

This script demonstrates the complete RAG (Retrieval-Augmented Generation) process:
1. Query processing and embedding generation
2. Hybrid search (semantic + keyword) with chunk ranking
3. Context construction from retrieved chunks
4. Complete prompt assembly for the LLM
5. Detailed scoring and ranking information

Usage:
    python -m test.test_rag_process "your_query_here"
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
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
    """Test and visualize the complete RAG process with detailed output."""
    
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
        
        # Step 4: Build context
        print("4️⃣ Building context from retrieved chunks...")
        context_start = time.time()
        
        if search_results:
            context_chunks = []
            for result in search_results:
                context_chunks.append(f"[Chunk {result['index']}]\n{result['document']}\n")
            context = "\n".join(context_chunks)
            
            # Truncate context if too long
            max_context_length = Config.LLM.MAX_CONTEXT_LENGTH()
            if len(context) > max_context_length:
                context = context[:max_context_length]
                print(f"     Context truncated to {max_context_length} characters")
        else:
            context = ""
            print("     No context available (no chunks retrieved)")
        
        context_time = time.time() - context_start
        print(f"    Context built in {context_time:.3f}s")
        print(f"   📏 Context length: {len(context)} characters")
        print()
        
        # Step 5: Generate system prompt
        print("5️⃣ Generating system prompt...")
        prompt_start = time.time()
        
        system_prompt = self.prompt_manager.get_system_prompt(context=context)
        prompt_time = time.time() - prompt_start
        
        print(f"    System prompt generated in {prompt_time:.3f}s")
        print(f"   📏 System prompt length: {len(system_prompt)} characters")
        print()
        
        # Step 6: Display complete prompt
        print("6️⃣ Complete Prompt Assembly:")
        print("=" * 80)
        print(" SYSTEM PROMPT:")
        print("-" * 40)
        print(system_prompt)
        print("-" * 40)
        print()
        print("👤 USER MESSAGE:")
        print("-" * 40)
        print(query)
        print("-" * 40)
        print()
        
        # Step 7: Generate model response
        print("7️⃣ Generating Model Response:")
        print("-" * 80)
        response_start = time.time()
        
        try:
            # Use the same search results from step 2 for consistency
            # Build context from the same search results used in step 4
            context_chunks = []
            for result in search_results:
                context_chunks.append(f"[Chunk {result['index']}]\n{result['document']}\n")
            context = "\n".join(context_chunks)
            
            # Truncate context if too long (same as step 4)
            max_context_length = Config.LLM.MAX_CONTEXT_LENGTH()
            if len(context) > max_context_length:
                context = context[:max_context_length]
                print(f"     Context truncated to {max_context_length} characters")
            
            # Generate system prompt using the same context
            system_prompt = self.prompt_manager.get_system_prompt(context=context)
            
            # Generate response using the chatbot service with the same context
            response = self.chatbot_service.get_response_with_context(
                query=query,
                context=context,
                search_results=search_results
            )
            response_time = time.time() - response_start
            
            print(f"    Model response generated in {response_time:.3f}s")
            response_text = response.response if hasattr(response, 'response') else ''
            print(f"   📏 Response length: {len(response_text)} characters")
            print()
            
            print(" MODEL RESPONSE:")
            print("-" * 40)
            print(response_text)
            print("-" * 40)
            print()
            
            # Show response metadata if available
            if hasattr(response, 'confidence') and response.confidence:
                print(" Response Confidence:")
                confidence = response.confidence
                print(f"    Overall Score: {confidence.get('score', 'N/A')}")
                print(f"   📈 Level: {confidence.get('level', 'N/A')}")
                if 'details' in confidence:
                    details = confidence['details']
                    print(f"   📋 Details:")
                    for key, value in details.items():
                        if key != 'reasoning':
                            print(f"      - {key}: {value}")
                print()
            
            if hasattr(response, 'search_metadata') and response.search_metadata:
                print(" Search Metadata:")
                search_meta = response.search_metadata
                print(f"    Results Count: {search_meta.get('results_count', 'N/A')}")
                print(f"    Top Scores: {search_meta.get('top_scores', 'N/A')}")
                print(f"   💾 Cached: {search_meta.get('cached_response', 'N/A')}")
                print()
                
        except Exception as e:
            print(f"    Error generating response: {e}")
            response_time = time.time() - response_start
            response = {"error": str(e)}
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Return comprehensive results
        return {
            "query": query,
            "processing_time": total_time,
            "query_embedding_time": query_time,
            "search_time": search_time,
            "context_build_time": context_time,
            "prompt_generation_time": prompt_time,
            "response_generation_time": response_time if 'response_time' in locals() else 0,
            "search_results": search_results,
            "context": context,
            "system_prompt": system_prompt,
            "model_response": response.response if 'response' in locals() and hasattr(response, 'response') else '',
            "response_confidence": response.confidence if 'response' in locals() and hasattr(response, 'confidence') else {},
            "search_metadata": response.search_metadata if 'response' in locals() and hasattr(response, 'search_metadata') else {},
            "context_length": len(context),
            "prompt_length": len(system_prompt),
            "response_length": len(response.response) if 'response' in locals() and hasattr(response, 'response') else 0,
            "chunks_retrieved": len(search_results),
            "total_documents": len(documents)
        }
    
    def display_summary(self, results: Dict[str, Any]):
        """Display a summary of the RAG process results."""
        if "error" in results:
            print(f" Process failed: {results['error']}")
            return
        
        print(" RAG Process Summary:")
        print("=" * 80)
        print(f" Query: {results['query']}")
        print(f"⏱️  Total Processing Time: {results['processing_time']:.3f}s")
        print(f"   - Query Embedding: {results['query_embedding_time']:.3f}s")
        print(f"   - Hybrid Search: {results['search_time']:.3f}s")
        print(f"   - Context Building: {results['context_build_time']:.3f}s")
        print(f"   - Prompt Generation: {results['prompt_generation_time']:.3f}s")
        print(f"   - Response Generation: {results['response_generation_time']:.3f}s")
        print()
        print(f" Documents: {results['total_documents']} total, {results['chunks_retrieved']} retrieved")
        print(f"📏 Context: {results['context_length']} characters")
        print(f"📝 System Prompt: {results['prompt_length']} characters")
        print(f" Model Response: {results['response_length']} characters")
        print()
        
        if results['search_results']:
            print("🏆 Top Chunk Scores:")
            for i, result in enumerate(results['search_results'][:3], 1):
                print(f"   {i}. Chunk {result['index']}: {result['combined_score']:.4f} "
                      f"(Semantic: {result['semantic_score']:.4f}, "
                      f"Keyword: {result['keyword_score']:.4f})")
        else:
            print("  No chunks retrieved above similarity threshold")
        print()
        
        # Show response quality metrics if available
        if results['response_confidence']:
            print(" Response Quality Metrics:")
            confidence = results['response_confidence']
            print(f"    Overall Confidence: {confidence.get('score', 'N/A')}")
            print(f"   📈 Quality Level: {confidence.get('level', 'N/A')}")
            if 'details' in confidence:
                details = confidence['details']
                print(f"   📋 Quality Breakdown:")
                for key, value in details.items():
                    if key != 'reasoning':
                        print(f"      - {key.replace('_', ' ').title()}: {value}")
            print()
        
        if results['search_metadata']:
            print(" Search Performance:")
            search_meta = results['search_metadata']
            print(f"    Chunks Found: {search_meta.get('results_count', 'N/A')}")
            print(f"    Top Scores: {search_meta.get('top_scores', 'N/A')}")
            print(f"   💾 Cached Response: {search_meta.get('cached_response', 'N/A')}")
            print()


def main():
    """Main function to run the RAG process tester."""
    print(" RAG Process Tester")
    print("=" * 80)
    print()
    
    # Initialize tester
    tester = RAGProcessTester()
    
    # Load vector store
    embeddings, documents = tester.load_vector_store()
    if embeddings is None or documents is None:
        print(" Cannot proceed without vector store data")
        return
    
    # Get query from command line or interactive input
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("💬 Enter your query (or 'quit' to exit):")
        query = input("> ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            return
    
    if not query:
        print(" No query provided")
        return
    
    print()
    
    # Process the query
    results = tester.process_query(query, embeddings, documents)
    
    # Display summary
    tester.display_summary(results)
    
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
            results = tester.process_query(query, embeddings, documents)
            tester.display_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        logger.exception("Unexpected error in RAG process tester")
