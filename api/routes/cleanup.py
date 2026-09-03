"""
Maintenance API endpoints for vector store rebuild and query adapter updates.
"""

# Standard library imports
import logging
from typing import Dict, Any

# Third-party imports
from fastapi import APIRouter, HTTPException, Depends, Body

# Local imports
from config.settings import Config
from api.dependencies import get_vector_store

from core.storage.vector_stores import VectorStore
# File watching functionality removed
from models.responses import VectorRebuildResponse, StatusEnum
# RAGConfig is now available through Config.RAG

router = APIRouter()
logger = logging.getLogger(__name__)


# Auto-reload endpoint removed


@router.post("/vectors/rebuild", response_model=VectorRebuildResponse)
async def rebuild_vectors(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Force rebuild the vector store from all current chunk files.
    
    This endpoint will:
    1. Load all documents from chunk files
    2. Create new embeddings for all documents
    3. Rebuild the vector store with updated vectors
    4. Save the updated vector store to disk
    """
    try:
        import time
        logger.info("🔄 Starting vector store rebuild via cleanup API...")
        start_time = time.time()
        
        # Rebuild the vector store from chunk files
        faiss_index, embeddings, documents = vector_store.rebuild_vector_store()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        processing_time_str = f"{processing_time:.2f}s"
        
        # Prepare response details
        details = {
            "chunk_files_loaded": len(documents) if documents else 0,
            "embedding_model": Config.RAG.EMBEDDING_MODEL(),
            "vector_dimensions": embeddings.shape[1] if embeddings is not None and len(embeddings) > 0 else 0,
            "vector_store_type": "FAISS" if faiss_index is not None else "NumPy",
            "chunks_directory": Config.File.CHUNKS_DIR(),
            "vectors_directory": Config.File.VECTORS_DIR()
        }
        
        logger.info(f" Vector store rebuild completed via cleanup API: {len(documents)} documents processed in {processing_time_str}")
        
        return VectorRebuildResponse(
            status=StatusEnum.SUCCESS,
            message="Vector store rebuilt successfully",
            documents_processed=len(documents) if documents else 0,
            vectors_created=len(embeddings) if embeddings is not None else 0,
            processing_time=processing_time_str,
            vector_store_path=Config.File.VECTOR_STORE_PATH(),
            details=details
        )
        
    except Exception as e:
        logger.error(f" Error during vector store rebuild: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild vector store: {str(e)}"
        )


@router.post("/query-adapter/update")
async def update_query_adapter(payload: Dict[str, Any] = Body(default={})):  # simple admin endpoint
    """
    Compute a query adapter from provided eval pairs and save to disk.
    Request body:
    {
      "queries": ["..."],
      "positives": ["..."],
      "lambda": 0.001
    }
    """
    try:
        from core.retrieval.query_expansion.query_adapter import build_from_evals, save_query_adapter
        from core.ai_services.embeddings.embeddings import get_embedding_service
        # RAGConfig is now available through Config.RAG

        queries = payload.get("queries", [])
        positives = payload.get("positives", [])
        lambda_reg = float(payload.get("lambda", 1e-3))

        if not queries or not positives or len(queries) != len(positives):
            raise HTTPException(status_code=400, detail="queries and positives must be non-empty and same length")

        embedder = get_embedding_service().get_embedder()
        adapter = build_from_evals(queries, positives, embedder, lambda_reg)
        path = Config.RAG.QUERY_ADAPTER_PATH()
        save_query_adapter(adapter, path)
        return {"success": True, "path": path, "dim": int(adapter.shape[0])}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" Error updating query adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))
