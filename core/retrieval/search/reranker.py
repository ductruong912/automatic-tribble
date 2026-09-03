"""
Pluggable reranker for hybrid retrieval results.
Uses a cross-encoder if available, otherwise falls back to a lightweight heuristic reranker.
"""

from typing import List, Dict, Any
import logging
import os

# Set trust_remote_code environment variable before importing sentence_transformers
os.environ["HF_TRUST_REMOTE_CODE"] = "True"

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception:
    CrossEncoder = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False


class Reranker:
    """
    Rerank retrieval results given the user query and candidate chunks.
    If a CrossEncoder model is available, use it; otherwise, rely on combined scores provided upstream.
    """

    def __init__(self, model_name: str = None):
        self._model = None
        if model_name is None:
            try:
                from config.settings import Config
                model_name = Config.RAG.RERANKER_MODEL()
            except Exception:
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model_name = model_name
        if CrossEncoder is not None or TRANSFORMERS_AVAILABLE:
            # Try configured model first, then known fallback models
            for candidate in [model_name, "cross-encoder/ms-marco-MiniLM-L-6-v2"]:
                try:
                    # Special handling for jinaai models that require trust_remote_code
                    if candidate.startswith("jinaai/") and TRANSFORMERS_AVAILABLE:
                        # Use transformers directly for jinaai models
                        tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
                        model = AutoModelForSequenceClassification.from_pretrained(candidate, trust_remote_code=True)
                        # Create a CrossEncoder-like wrapper
                        self._model = self._create_cross_encoder_wrapper(model, tokenizer)
                        logger.info(f" Reranker loaded (transformers): {candidate}")
                    else:
                        # Use sentence_transformers for other models
                        self._model = CrossEncoder(candidate)
                        logger.info(f" Reranker loaded (sentence_transformers): {candidate}")
                    
                    self._model_name = candidate
                    break
                except Exception as e:
                    logger.warning(f" Failed to load reranker '{candidate}': {e}")

    def _create_cross_encoder_wrapper(self, model, tokenizer):
        """Create a CrossEncoder-like wrapper for transformers models."""
        class CrossEncoderWrapper:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
                self.model.eval()
            
            def predict(self, pairs):
                """Predict scores for query-document pairs."""
                import torch
                import numpy as np
                
                scores = []
                for query, document in pairs:
                    # Tokenize the pair
                    inputs = self.tokenizer(
                        query, 
                        document, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512,
                        padding=True
                    )
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Get the score (logits) for the positive class
                        score = torch.sigmoid(outputs.logits).item()
                        scores.append(score)
                
                return np.array(scores)
        
        return CrossEncoderWrapper(model, tokenizer)

    def available(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not results:
            return []

        # Cross-encoder based reranking
        if self._model is not None:
            try:
                pairs = [(query, r.get("document", "")) for r in results]
                scores = self._model.predict(pairs)
                for i, s in enumerate(scores):
                    results[i]["rerank_score"] = float(s)
                results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                return results[:top_k]
            except Exception as e:
                logger.warning(f" Cross-encoder reranking failed, falling back: {e}")

        # Heuristic fallback: prefer combined score, then semantic, then keyword
        for r in results:
            r["rerank_score"] = (
                1.0 * float(r.get("combined_score", 0.0))
                + 0.25 * float(r.get("norm_semantic_score", r.get("semantic_score", 0.0)))
                + 0.1 * float(r.get("norm_keyword_score", r.get("keyword_score", 0.0)))
            )
        results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return results[:top_k]


