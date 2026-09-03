"""
Embedding service for text vectorization using sentence transformers.
Provides GPU/CPU fallback and multiple model support for robust embedding generation.
"""

# Third-party imports
try:
    import torch
except Exception:
    torch = None
from sentence_transformers import SentenceTransformer
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Lazy loading to avoid network issues during import
embedder = None
_embedding_service_instance = None


class EmbeddingService:
    """
    Optimized service for handling text embeddings generation with GPU/CPU fallback.
    Supports multiple embedding models with automatic device selection, caching, and batch processing.
    """

    def __init__(self):
        # Lazy initialization to avoid loading models until needed
        self.embedder = None
        self._embedding_cache = {}  # Simple in-memory cache
        self._cache_enabled = True
        self._cache_hits = 0
        self._cache_misses = 0

    def get_embedder(self):
        """
        Get the embedder instance with automatic GPU/CPU fallback.
        Loads the best available model with device optimization.
        """
        if self.embedder is None:
            # Check GPU availability for performance optimization
            gpu_available = bool(torch) and torch.cuda.is_available()
            print(f"GPU availability for embeddings: {gpu_available}")

            # Get model from config, with fallback to multilingual support
            try:
                from config.settings import Config
                primary_model = Config.LLM.EMBEDDING_MODEL()
            except Exception:
                primary_model = "paraphrase-multilingual-MiniLM-L12-v2"
            
            # List of models to try in order of preference (respect env config first)
            models_to_try = [
                primary_model,  # Use configured model first
                "paraphrase-multilingual-MiniLM-L12-v2",  # Best multilingual support
                "all-MiniLM-L6-v2",  # Fast and efficient
            ]

            for model_name in models_to_try:
                try:
                    print(f" Attempting to load {model_name}...")

                    # Try GPU first for better performance if available
                    if gpu_available:
                        try:
                            self.embedder = SentenceTransformer(
                                model_name, device="cuda"
                            )
                            print(f" {model_name} loaded successfully on GPU")
                            break
                        except Exception as gpu_error:
                            print(f" GPU loading failed for {model_name}: {gpu_error}")
                            print("🔄 Falling back to CPU...")

                    # Fallback to CPU if GPU fails or unavailable
                    self.embedder = SentenceTransformer(model_name, device="cpu")
                    print(f" {model_name} loaded successfully on CPU")
                    break

                except Exception as e:
                    print(f" Failed to load {model_name}: {e}")
                    continue

            if self.embedder is None:
                raise RuntimeError("Failed to load any embedding model")

        return self.embedder

    def encode(self, texts, convert_to_numpy=True):
        """
        Encode texts into vector embeddings for similarity search with caching.

        Args:
            texts: List of text strings to encode
            convert_to_numpy: Whether to return numpy arrays (default: True)

        Returns:
            Vector embeddings ready for similarity calculations
        """
        # Check cache first for single text queries
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
            
        if self._cache_enabled and len(texts) == 1:
            text = texts[0]
            cache_key = f"{text}_{convert_to_numpy}"
            if cache_key in self._embedding_cache:
                self._cache_hits += 1
                return self._embedding_cache[cache_key]
            self._cache_misses += 1
        
        embedder = self.get_embedder()
        embeddings = embedder.encode(texts, convert_to_numpy=convert_to_numpy)
        
        # Cache single text results
        if self._cache_enabled and len(texts) == 1:
            self._embedding_cache[cache_key] = embeddings
            # Limit cache size to prevent memory issues
            if len(self._embedding_cache) > 1000:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
        
        return embeddings

    # Query adapter (closed-form) support
    _query_adapter_matrix = None
    _adapter_loaded_path = None
    _adapter_loaded_mtime = None

    def load_query_adapter(self, path: str) -> bool:
        """
        Load a query adapter matrix from disk (NumPy .npy), returning True if loaded.
        """
        try:
            import numpy as np  # local import to avoid import-time overhead
            from pathlib import Path
            import os
            p = Path(path)
            if not p.exists():
                return False
            # Skip reload if unchanged
            mtime = os.path.getmtime(str(p))
            if (
                EmbeddingService._adapter_loaded_path == str(p)
                and EmbeddingService._adapter_loaded_mtime == mtime
                and EmbeddingService._query_adapter_matrix is not None
            ):
                return True
            EmbeddingService._query_adapter_matrix = np.load(str(p))
            EmbeddingService._adapter_loaded_path = str(p)
            EmbeddingService._adapter_loaded_mtime = mtime
            print(f" Loaded query adapter from {path}")
            return True
        except Exception as e:
            print(f" Failed to load query adapter: {e}")
            return False

    def apply_query_adapter(self, query_embedding):
        """
        Apply the query adapter matrix if available. Returns transformed embedding.
        Accepts list/np.ndarray; returns np.ndarray.
        """
        try:
            import numpy as np
            if EmbeddingService._query_adapter_matrix is None:
                return query_embedding if isinstance(query_embedding, np.ndarray) else np.array(query_embedding)
            emb = query_embedding if isinstance(query_embedding, np.ndarray) else np.array(query_embedding)
            # Handle 1D vs 2D shapes; we expect shape (1, d)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            adapter = EmbeddingService._query_adapter_matrix
            if adapter.ndim == 1:
                # Interpret as diagonal scaling vector
                adapter = np.diag(adapter)
            transformed = emb @ adapter
            return transformed
        except Exception:
            # Fail open: return original embedding
            return query_embedding

    async def async_encode(self, texts, convert_to_numpy=True):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(
                pool, self.encode, texts, convert_to_numpy
            )

    def get_device_info(self):
        """
        Get information about the embedding model device and configuration.
        Useful for debugging and performance monitoring.
        """
        if self.embedder is None:
            return {"status": "Not loaded", "device": "Unknown", "gpu_available": bool(torch) and torch.cuda.is_available()}

        try:
            device = str(self.embedder.device)
            model_name = (
                self.embedder._modules["0"].__class__.__name__
                if hasattr(self.embedder, "_modules")
                else "Unknown"
            )
            return {
                "status": "Loaded",
                "device": device,
                "model": model_name,
                "gpu_available": bool(torch) and torch.cuda.is_available(),
            }
        except:
            return {"status": "Loaded", "device": "Unknown", "model": "Unknown", "gpu_available": bool(torch) and torch.cuda.is_available()}

    def reset_embedder(self):
        """
        Reset the embedder to force reinitialization.
        Useful when GPU becomes available after initial CPU fallback.
        """
        if self.embedder is not None:
            # Clear the embedder to force reinitialization
            self.embedder = None
            print("🔄 Embedder reset - will reinitialize on next use")


# Global functions for backward compatibility and convenience
def get_embedding_service():
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance

def reset_embedding_service():
    """Reset the global embedding service to force reinitialization."""
    global _embedding_service_instance
    if _embedding_service_instance is not None:
        _embedding_service_instance.reset_embedder()
    else:
        _embedding_service_instance = None


def get_embedder():
    """Get the global embedder instance for legacy code compatibility."""
    service = get_embedding_service()
    return service.get_embedder()


def get_device_info():
    """Get information about the embedding model device for monitoring."""
    service = get_embedding_service()
    return service.get_device_info()
