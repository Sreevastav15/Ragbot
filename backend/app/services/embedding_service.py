# app/services/embedding_service.py

"""
Vector store creation using ChromaDB + Google Text Embeddings.

Key properties:
- Uses clean chunk text (no prefixes)
- Preserves metadata (page_number, etc.)
- Uses batch embeddings via GoogleTextEmbedding
- Persists per-document vectorstore
"""

from __future__ import annotations
import os
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
        Unique identifier for the document

    Returns
    -------
    str
        Path to persisted ChromaDB directory
    """

    persist_dir = f"static/chroma_stores/{doc_id}"
    os.makedirs(persist_dir, exist_ok=True)

    # Initialize embedding model
    embeddings = GoogleTextEmbedding()

    # Sanity check: ensure chunks are clean
    cleaned_chunks = []
    for chunk in chunks:
        text = chunk.page_content.strip()

        # Skip empty chunks
        if not text:
            continue

        cleaned_chunks.append(chunk)

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