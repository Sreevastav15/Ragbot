# app/services/google_embedding.py

"""
HuggingFace embedding wrapper for LangChain.

Replaces Google Gemini embeddings with a local model:
- Model: BAAI/bge-small-en-v1.5  (384-dim, matches existing ChromaDB collections)
- Optimized for RAG retrieval tasks
- No external API calls (faster + stable)

BGE Model Prefix Notes:
- Documents: NO prefix (embed raw text)
- Queries:   Instruction prefix "Represent this sentence for searching relevant passages: "
  This asymmetric prefixing is REQUIRED for BGE models to work correctly.
  Using the wrong prefix (e.g. "query: " which is for E5 models) causes
  query/document embedding space mismatch → poor retrieval quality.

DIMENSION NOTE:
  bge-small-en-v1.5 → 384 dims  ← what existing ChromaDB collections expect
  bge-base-en-v1.5  → 768 dims  ← DO NOT switch without wiping chroma_stores/
  If you want to upgrade to bge-base, delete static/chroma_stores/ entirely
  and re-upload all documents so collections are rebuilt at the new dimension.
"""

from typing import List
import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# BGE instruction prefix for queries — DO NOT use "query: " (that is for E5 models)
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class GoogleTextEmbedding(Embeddings):
    """
    HuggingFace embedding wrapper using SentenceTransformers.

    Model:
        BAAI/bge-small-en-v1.5 (384-dim)

    Features:
    - Fast CPU inference
    - High-quality embeddings for retrieval
    - Normalized vectors for cosine similarity
    - Correct asymmetric BGE prefixing (query instruction vs bare document text)
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

        BGE documents are embedded WITHOUT any prefix — the model was trained
        to receive plain passage text on the document side.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query.

        IMPORTANT for BGE models:
        The query-side instruction prefix significantly improves retrieval
        quality. BGE is trained with asymmetric prompting:
          - documents → no prefix
          - queries   → _BGE_QUERY_INSTRUCTION prefix

        Do NOT use "query: " here — that prefix belongs to E5-family models
        (e.g. intfloat/e5-base-v2). Using the wrong prefix causes the query
        and document embeddings to live in misaligned sub-spaces, degrading
        retrieval relevance.
        """
        prefixed_query = _BGE_QUERY_INSTRUCTION + query

        embedding = self.model.encode(
            prefixed_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()