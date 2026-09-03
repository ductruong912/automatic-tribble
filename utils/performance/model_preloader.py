"""
Model preloader service for loading all ML models at startup.
Eliminates cold start delays and improves response times.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class ModelPreloader:
    """
    Preloads all ML models at startup to eliminate cold start delays.
    Handles embedding models, reranker, and other ML components.
    """
    
    def __init__(self):
        """Initialize the model preloader."""
        self.models_loaded = False
        self.loading_start_time = None
        self.loading_duration = 0
        self.loaded_models = {}
        self.loading_errors = {}
        
    async def preload_all_models(self):
        """Preload all models concurrently for maximum speed."""
        if self.models_loaded:
            logger.info("Models already loaded, skipping preload")
            return
            
        self.loading_start_time = time.time()
        logger.info("Starting model preloading...")
        
        try:
            # Load models concurrently for maximum speed
            await asyncio.gather(
                self._preload_embedding_model(),
                self._preload_reranker_model(),
                self._preload_vector_store(),
                return_exceptions=True
            )
            
            self.loading_duration = time.time() - self.loading_start_time
            self.models_loaded = True
            
            logger.info(f"All models preloaded successfully in {self.loading_duration:.2f}s")
            self._log_model_status()
            
        except Exception as e:
            logger.error(f"Error during model preloading: {e}")
            self.loading_errors["general"] = str(e)
    
    async def _preload_embedding_model(self):
        """Preload embedding model."""
        try:
            logger.info("🔄 Preloading embedding model...")
            start_time = time.time()
            
            from core.ai_services.embeddings.embeddings import get_embedding_service
            embedding_service = get_embedding_service()
            embedder = embedding_service.get_embedder()
            
            # Warm up with a test encode
            test_texts = ["hello", "xin chào", "test"]
            _ = embedder.encode(test_texts, convert_to_numpy=True, show_progress_bar=False)
            
            load_time = time.time() - start_time
            self.loaded_models["embedding"] = {
                "model": embedder,
                "load_time": load_time,
                "status": "success"
            }
            logger.info(f"Embedding model loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to preload embedding model: {e}")
            self.loading_errors["embedding"] = str(e)
            self.loaded_models["embedding"] = {"status": "failed", "error": str(e)}
    
    async def _preload_reranker_model(self):
        """Preload reranker model."""
        try:
            logger.info("🔄 Preloading reranker model...")
            start_time = time.time()
            
            from core.retrieval.search.reranker import Reranker
            reranker = Reranker()
            
            # Only warm up if reranker is available
            if reranker.available():
                # Warm up with a test rerank
                test_query = "test query"
                test_results = [
                    {"document": "test document 1", "score": 0.8},
                    {"document": "test document 2", "score": 0.6}
                ]
                _ = reranker.rerank(test_query, test_results, top_k=2)
            
            load_time = time.time() - start_time
            self.loaded_models["reranker"] = {
                "model": reranker,
                "load_time": load_time,
                "status": "success"
            }
            logger.info(f"Reranker model loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to preload reranker model: {e}")
            self.loading_errors["reranker"] = str(e)
            self.loaded_models["reranker"] = {"status": "failed", "error": str(e)}
    
    async def _preload_vector_store(self):
        """Preload vector store."""
        try:
            logger.info("🔄 Preloading vector store...")
            start_time = time.time()
            
            from core.storage.vector_stores.vector_store_optimized import OptimizedVectorStore
            vs = OptimizedVectorStore()
            vector_store_data = vs.load_vector_store()
            
            load_time = time.time() - start_time
            self.loaded_models["vector_store"] = {
                "model": vs,
                "data": vector_store_data,
                "load_time": load_time,
                "status": "success"
            }
            logger.info(f"Vector store loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to preload vector store: {e}")
            self.loading_errors["vector_store"] = str(e)
            self.loaded_models["vector_store"] = {"status": "failed", "error": str(e)}
    
    def get_model(self, model_name: str):
        """Get a preloaded model by name."""
        if model_name in self.loaded_models:
            model_data = self.loaded_models[model_name]
            if model_data["status"] == "success":
                return model_data["model"]
        return None
    
    def get_vector_store_data(self):
        """Get preloaded vector store data."""
        if "vector_store" in self.loaded_models:
            model_data = self.loaded_models["vector_store"]
            if model_data["status"] == "success":
                return model_data["data"]
        return None
    
    def _log_model_status(self):
        """Log the status of all loaded models."""
        logger.info("Model Preload Status:")
        for model_name, model_data in self.loaded_models.items():
            if model_data["status"] == "success":
                load_time = model_data.get("load_time", 0)
                logger.info(f"  {model_name}: {load_time:.2f}s")
            else:
                error = model_data.get("error", "Unknown error")
                logger.warning(f"  {model_name}: {error}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive preload status."""
        return {
            "models_loaded": self.models_loaded,
            "loading_duration": self.loading_duration,
            "loaded_models": {name: data["status"] for name, data in self.loaded_models.items()},
            "loading_errors": self.loading_errors,
            "total_models": len(self.loaded_models)
        }


# Global model preloader instance
_model_preloader: Optional[ModelPreloader] = None


def get_model_preloader() -> ModelPreloader:
    """Get the global model preloader instance."""
    global _model_preloader
    if _model_preloader is None:
        _model_preloader = ModelPreloader()
    return _model_preloader


async def preload_all_models():
    """Preload all models using the global preloader."""
    preloader = get_model_preloader()
    await preloader.preload_all_models()
