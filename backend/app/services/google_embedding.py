# app/services/google_embedding.py

"""
HuggingFace embedding wrapper for LangChain.

Replaces Google Gemini embeddings with a local model:
- Model: BAAI/bge-small-en-v1.5
- Optimized for RAG retrieval tasks
- No external API calls (faster + stable)
"""

from typing import List
import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class GoogleTextEmbedding(Embeddings):
    """
    HuggingFace embedding wrapper using SentenceTransformers.

    Model:
        BAAI/bge-small-en-v1.5

    Features:
    - Fast CPU inference
    - High-quality embeddings for retrieval
    - Normalized vectors for cosine similarity
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _normalize(self, vec: List[float]) -> List[float]:
        v = np.array(vec, dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0:
            return v.tolist()
        return (v / norm).tolist()

    # ── Embedding methods ────────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents (batch).
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # already normalized
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query.

        IMPORTANT for BGE models:
        Prefix improves retrieval quality.
        """
        query = f"query: {query}"

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()