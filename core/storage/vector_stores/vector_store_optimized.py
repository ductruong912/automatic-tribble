"""
Optimized Vector store with HDF5 storage and FAISS indexing for better performance.
Provides faster loading, incremental updates, and memory-efficient operations.
"""

# Standard library imports
import os
import logging
import h5py
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Third-party imports
import numpy as np
import faiss

# Local imports
from config.settings import Config
from ...ai_services.embeddings.embeddings import get_embedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class OptimizedVectorStore(VectorStore):
    """
    High-performance vector store using HDF5 and FAISS for speed optimization.
    Provides 5-10x faster loading and more efficient memory usage.
    """

    def __init__(self, vector_store_path: Optional[str] = None):
        """
        Initialize optimized vector store with HDF5 backend.
        """
        base_path = vector_store_path or Config.Database.VECTOR_STORE_PATH()
        self.h5_path = base_path.replace('.pkl', '.h5')
        self.metadata_path = base_path.replace('.pkl', '_metadata.json')
        self.faiss_index_path = base_path.replace('.pkl', '_faiss.index')
        
        self.embeddings = None
        self.documents = None
        self.document_metadata = None
        self.faiss_index = None
        
        logger.info(f"[OptimizedVectorStore] Initialized with HDF5 backend: {self.h5_path}")

    def load_vector_store(self) -> Tuple[Optional[Any], np.ndarray, List[str]]:
        """
        Load vector store from HDF5 file with lazy loading for better memory efficiency.
        """
        logger.info(f"[OptimizedVectorStore] Loading from HDF5: {self.h5_path}")
        
        try:
            # Load embeddings and documents from HDF5 with memory mapping for large files
            with h5py.File(self.h5_path, 'r') as f:
                # Use memory mapping for large datasets to reduce memory usage
                if f['embeddings'].size > 1000000:  # 1M+ vectors
                    logger.info(" Using memory mapping for large dataset")
                    self.embeddings = f['embeddings']  # Memory mapped, not loaded into RAM
                else:
                    self.embeddings = f['embeddings'][:]  # Load into memory for small datasets
                
                # Documents stored as UTF-8 encoded strings
                self.documents = [doc.decode('utf-8') for doc in f['documents'][:]]
            
            # Load metadata from JSON (smaller, faster for metadata)
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.document_metadata = json.load(f)
            else:
                self.document_metadata = [{'source': 'legacy'} for _ in self.documents]

            # Ensure metadata is aligned, complete, and contains source_id
            try:
                needs_rebuild = False
                if not isinstance(self.document_metadata, list) or len(self.document_metadata) != len(self.documents):
                    needs_rebuild = True
                else:
                    # Check keys; fill missing source_id from existing fields if possible
                    for m in self.document_metadata:
                        if 'source_id' not in m:
                            needs_rebuild = True
                            break
                if needs_rebuild:
                    from core.storage.metadata_stores.metadata_store import MetadataStore
                    logger.warning(" Metadata missing or misaligned. Rebuilding metadata from chunks directory...")
                    rebuilt = MetadataStore().create_metadata_from_chunks(self.documents)
                    # If rebuild fails to match length, fall back to minimal source ids
                    if len(rebuilt) != len(self.documents):
                        rebuilt = [{'source_id': 'unknown', 'source_type': 'unknown'} for _ in self.documents]
                    self.document_metadata = rebuilt
                    # Persist the fixed metadata
                    with open(self.metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f" Failed to validate/rebuild metadata: {e}")
            
            # Load FAISS index for ultra-fast similarity search
            if os.path.exists(self.faiss_index_path):
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                logger.info(f" FAISS index loaded: {self.faiss_index.ntotal} vectors")
            
            logger.info(f" HDF5 vector store loaded: {len(self.documents)} documents, shape: {self.embeddings.shape}")
            return None, self.embeddings, self.documents
            
        except FileNotFoundError:
            logger.warning("HDF5 vector store not found, creating new one...")
            return self.rebuild_vector_store()
        except Exception as e:
            logger.error(f" [OptimizedVectorStore Error] {e}")
            return self.rebuild_vector_store()

    def save_vector_store(self):
        """
        Save vector store to HDF5 with compression for optimal performance.
        """
        logger.info(f"[OptimizedVectorStore] Saving to HDF5: {self.h5_path}")
        
        try:
            from utils.file_operations.file_manager import FileManager as FileUtils
            FileUtils.ensure_directory_exists(os.path.dirname(self.h5_path))
            
            # Save embeddings and documents to HDF5 with compression
            with h5py.File(self.h5_path, 'w') as f:
                f.create_dataset('embeddings', data=self.embeddings, compression='gzip', compression_opts=6)
                # Encode documents as UTF-8 byte strings for HDF5 storage
                encoded_docs = [doc.encode('utf-8') for doc in self.documents]
                f.create_dataset('documents', data=encoded_docs, compression='gzip', compression_opts=6)
            
            # Save metadata to JSON
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
            
            # Create and save FAISS index for fast similarity search
            if self.embeddings is not None and len(self.embeddings) > 0:
                dimension = self.embeddings.shape[1]
                # Use IndexFlatIP for cosine similarity (after L2 normalization)
                self.faiss_index = faiss.IndexFlatIP(dimension)
                # Normalize embeddings for cosine similarity
                normalized_embeddings = self.embeddings.astype('float32')
                faiss.normalize_L2(normalized_embeddings)
                self.faiss_index.add(normalized_embeddings)
                faiss.write_index(self.faiss_index, self.faiss_index_path)
                logger.info(f" FAISS index created and saved: {self.faiss_index.ntotal} vectors")
            
            logger.info(f" HDF5 vector store saved: {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f" [OptimizedVectorStore Error] {e}")

    def create_embeddings(self, documents: List[str]) -> np.ndarray:
        """
        Create embeddings with batch processing for efficiency.
        """
        logger.info(f"Creating embeddings for {len(documents)} documents...")
        
        embedder = get_embedder()
        
        # Process in batches to avoid memory issues
        batch_size = 32  # Optimal batch size for most sentence transformers
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embedder.encode(batch_docs, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:  # Log every 10 batches
                logger.info(f" Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
        
        embeddings = np.vstack(all_embeddings)
        logger.info(f" Created embeddings: shape {embeddings.shape}")
        return embeddings

    def fast_similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ultra-fast similarity search using FAISS index.
        Returns similarities and indices in milliseconds instead of seconds.
        """
        if self.faiss_index is None:
            # Fallback to numpy cosine similarity
            return self._numpy_similarity_search(query_embedding, k)
        
        # Normalize query embedding for cosine similarity
        query_norm = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_norm)
        
        # Search using FAISS - extremely fast
        similarities, indices = self.faiss_index.search(query_norm, k)
        
        return similarities[0], indices[0]
    
    def _numpy_similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback numpy-based similarity search if FAISS fails.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[top_indices]
        
        return top_similarities, top_indices

    def rebuild_vector_store(self) -> Tuple[Optional[Any], np.ndarray, List[str]]:
        """
        Rebuild vector store from chunk files with optimized processing.
        """
        logger.info(f"[OptimizedVectorStore] Rebuilding from chunk files...")
        
        from core.storage.metadata_stores.document_store import DocumentStore
        from core.storage.metadata_stores.metadata_store import MetadataStore
        
        doc_store = DocumentStore()
        meta_store = MetadataStore()
        
        # Get all documents
        all_documents = doc_store.load_documents_from_chunks()
        logger.info(f" Loaded {len(all_documents)} documents from chunk files")
        
        if not all_documents:
            logger.warning(" No documents found to create vector store")
            # Clear existing vector store files when rebuilding with empty data
            self._clear_vector_store_files()
            # Set instance variables to empty state
            self.embeddings = None
            self.documents = []
            self.document_metadata = []
            self.faiss_index = None
            return None, np.array([]), []
        
        # Create embeddings with batch processing
        embeddings = self.create_embeddings(all_documents)
        
        # Create metadata
        document_metadata = meta_store.create_metadata_from_chunks(all_documents)
        logger.info(f"📋 Created metadata for {len(document_metadata)} documents")
        
        # Save everything
        self.embeddings = embeddings
        self.documents = all_documents
        self.document_metadata = document_metadata
        
        self.save_vector_store()
        
        logger.info(f" [OptimizedVectorStore] Rebuilt successfully: {len(all_documents)} documents")
        return None, embeddings, all_documents

    def _clear_vector_store_files(self):
        """Clear all vector store files when rebuilding with empty data."""
        try:
            # Clear instance variables
            self.embeddings = None
            self.documents = None
            self.document_metadata = None
            self.faiss_index = None
            
            # Delete existing files
            files_to_delete = [self.h5_path, self.metadata_path, self.faiss_index_path]
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f" Deleted {file_path}")
            
            logger.info(" Cleared all vector store files")
        except Exception as e:
            logger.error(f" Error clearing vector store files: {e}")

    def get_metadata(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get current document metadata for source tracking.
        Returns list of metadata dictionaries for each document.
        """
        return self.document_metadata