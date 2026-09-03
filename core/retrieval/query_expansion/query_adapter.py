"""
Compute a closed-form query adapter from eval data and save to disk as .npy.
This mirrors the idea from RAGLite: fit a linear adapter mapping query embeddings
to better align with relevant chunk embeddings.
"""

from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_query_adapter(
    query_embeddings: np.ndarray,
    positive_embeddings: np.ndarray,
    regularization_lambda: float = 1e-3,
) -> np.ndarray:
    """
    Closed-form ridge-regression adapter A that minimizes ||Q A - P||^2 + λ||A||^2
    where Q are query embeddings and P are target (positive) embeddings.
    Returns matrix A of shape (d, d).
    """
    assert query_embeddings.ndim == 2 and positive_embeddings.ndim == 2
    assert query_embeddings.shape == positive_embeddings.shape

    n, d = query_embeddings.shape
    q = query_embeddings
    p = positive_embeddings

    # Solve (Q^T Q + λI) A = Q^T P  => A = (Q^T Q + λI)^{-1} Q^T P
    qtq = q.T @ q
    reg = regularization_lambda * np.eye(d, dtype=qtq.dtype)
    lhs = qtq + reg
    rhs = q.T @ p
    A = np.linalg.solve(lhs, rhs)
    return A


def save_query_adapter(adapter: np.ndarray, path: str) -> None:
    import os
    from pathlib import Path
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    np.save(path, adapter)
    logger.info(f" Saved query adapter to {path}")


def build_from_evals(
    queries: List[str],
    positives: List[str],
    embedder,
    lambda_reg: float = 1e-3,
) -> np.ndarray:
    """
    Encode queries and their matched positive chunks and compute the adapter.
    """
    q_emb = embedder.encode(queries, convert_to_numpy=True)
    p_emb = embedder.encode(positives, convert_to_numpy=True)
    return compute_query_adapter(q_emb, p_emb, regularization_lambda=lambda_reg)


