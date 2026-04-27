# app/services/embedding_service.py

"""
Vector store creation using ChromaDB + HuggingFace BGE embeddings.

Key properties:
- Uses clean chunk text (no prefixes)
- Preserves metadata (page_number, etc.)
- Uses batch embeddings via GoogleTextEmbedding (bge-small-en-v1.5, 384-dim)
- Persists per-document vectorstore

FIX — dimension mismatch on re-upload:
  ChromaDB stores the embedding dimension when a collection is first created.
  If the persist directory already exists from a previous upload (or a previous
  model with a different dimension), Chroma raises:
      InvalidArgumentError: Collection expecting embedding with dimension of
      384, got 768   (or vice-versa)
  The fix is to delete the old persist directory before recreating the
  collection, so every upload always starts from a clean slate.
"""

from __future__ import annotations
import os
import shutil
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LCDoc

from app.services.google_embedding import GoogleTextEmbedding


def create_vectorstore(chunks: List[LCDoc], doc_id: str) -> str:
    """
    Embed and persist chunks to a per-document ChromaDB collection.

    Parameters
    ----------
    chunks : List[Document]
        LangChain Document objects with:
        - page_content (text)
        - metadata (must include page_number ideally)

    doc_id : str
        Unique identifier for the document (used as directory name).

    Returns
    -------
    str
        Path to persisted ChromaDB directory.
    """

    persist_dir = f"static/chroma_stores/{doc_id}"

    # ── Wipe any stale collection ─────────────────────────────────────────────
    # If persist_dir already exists it may have been built with a different
    # embedding model / dimension.  Deleting it guarantees we always create a
    # fresh collection that matches the current model's output dimension.
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    os.makedirs(persist_dir, exist_ok=True)

    # Initialize embedding model
    embeddings = GoogleTextEmbedding()

    # Sanity check: ensure chunks are clean
    cleaned_chunks = [
        chunk for chunk in chunks if chunk.page_content.strip()
    ]

    if not cleaned_chunks:
        raise ValueError("No valid chunks to embed.")

    # Create vectorstore using batch embeddings
    vectorstore = Chroma.from_documents(
        documents=cleaned_chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    # Persist to disk
    vectorstore.persist()

    return persist_dir