# app/services/hybrid_search.py
"""
Hybrid retrieval: BM25 (keyword) + vector (semantic) with score fusion.
Falls back gracefully to vector-only if BM25 index is unavailable.

FIX: BM25 documents now emit "source_filename" in their metadata (not "source"),
matching the key that _build_context() in qa_service.py expects.
The qa_service already overwrites this after retrieval, but using the correct key
from the start prevents silent fallback to "Unknown" if the overwrite were skipped.
"""

from __future__ import annotations
from typing import List
from langchain_core.documents import Document as LCDoc
from rank_bm25 import BM25Okapi
import re


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def bm25_retrieve(query: str, db_chunks, top_k: int = 15) -> List[LCDoc]:
    """Retrieve top_k chunks from DB-stored BM25 corpus."""
    if not db_chunks:
        return []

    corpus = [_tokenize(c.content) for c in db_chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        if score > 0:
            chunk = db_chunks[idx]
            results.append(
                LCDoc(
                    page_content=chunk.content,
                    metadata={
                        "page_number": chunk.page_number,
                        "bm25_score": float(score),
                        "document_id": chunk.document_id,
                        # Use "source_filename" to match what _build_context() reads.
                        # _source_filename is set on the ORM chunk object by qa_service
                        # before this function is called.
                        "source_filename": getattr(chunk, "_source_filename", ""),
                    },
                )
            )
    return results


def reciprocal_rank_fusion(
    vector_docs: List[LCDoc],
    bm25_docs: List[LCDoc],
    k: int = 60,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[LCDoc]:
    """
    Merge two ranked lists via Reciprocal Rank Fusion (RRF).
    Returns a unified, deduplicated list sorted by fused score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, LCDoc] = {}

    def key(doc: LCDoc) -> str:
        return doc.page_content[:120]  # content fingerprint

    for rank, doc in enumerate(vector_docs):
        dk = key(doc)
        scores[dk] = scores.get(dk, 0) + vector_weight * (1 / (k + rank + 1))
        doc_map[dk] = doc

    for rank, doc in enumerate(bm25_docs):
        dk = key(doc)
        scores[dk] = scores.get(dk, 0) + bm25_weight * (1 / (k + rank + 1))
        if dk not in doc_map:
            doc_map[dk] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]