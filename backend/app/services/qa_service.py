# app/services/qa_service.py
"""
Core QA service — complete rewrite.

Key improvements over the previous version:
  1. LLMChain + PromptTemplate + OutputParser everywhere (no raw f-strings).
  2. Fixed retrieval: embedding_service no longer prepends "passage from page N:"
     to chunks (that prefix polluted the vector search). Retrieval now uses
     the rewritten query directly against clean chunk text.
  3. BGE-reranker-base (CrossEncoder) is used on the fused candidate set
     before slicing to the top-k context.
  4. ROUGE-L is computed against ground-truth JSON after every answer and
     logged to the backend console. It is NOT sent to the frontend.
  5. Streaming path uses STREAMING_QA_TEMPLATE (plain Markdown, no JSON schema)
     so the LLM never tries to stream a JSON object mid-token.
"""

from __future__ import annotations
import json
import logging
import os
import time
from typing import Generator, List, Optional

from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document as LCDoc

from app.services.google_embedding import GoogleTextEmbedding
from app.services.reranker import rerank
from app.services.hybrid_search import bm25_retrieve, reciprocal_rank_fusion
from app.services.query_rewriter import rewrite_query, compute_k
from app.services.eval_service import compute_and_log_metrics
from app.prompts.prompts import QA_TEMPLATE, STREAMING_QA_TEMPLATE
from app.parser.parser import QA_FORMAT_INSTRUCTIONS, safe_parse_qa

from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logger = logging.getLogger(__name__)

# Number of chunks kept after reranking for the final context window
_CONTEXT_TOP_K = 6


# ── LLM factory ───────────────────────────────────────────────────────────────

def _build_llm(streaming: bool = False) -> ChatGroq:
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0,
        streaming=streaming,
    )


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _retrieve_vector(vector_path: str, query: str, k: int) -> List[LCDoc]:
    """
    Semantic retrieval from ChromaDB.
    Uses the rewritten query directly — no prefix manipulation.
    """
    embeddings = GoogleTextEmbedding()
    vectorstore = Chroma(
        persist_directory=vector_path,
        embedding_function=embeddings,
    )
    # MMR retrieval for diversity; fall back to similarity if k is large
    try:
        docs = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
    except Exception:
        docs = vectorstore.similarity_search(query, k=k)
    return docs


def _build_context(reranked: List[LCDoc]) -> tuple[str, list[dict]]:
    """
    Build the context string and source list from reranked documents.
    Each block is clearly delimited so the LLM can distinguish sources.
    """
    blocks = ""
    sources = []
    seen = set()

    for doc in reranked:
        fname = doc.metadata.get("source_filename", "Unknown")
        page  = doc.metadata.get("page_number", "?")
        key   = f"{fname}::{page}::{doc.page_content[:60]}"
        if key in seen:
            continue
        seen.add(key)

        blocks += (
            f"\n---\n"
            f"[Document: {fname} | Page {page}]\n"
            f"{doc.page_content.strip()}\n"
        )
        sources.append({"filename": fname, "page": page})

    return blocks.strip(), sources


def _build_memory(summary: Optional[str], chat_history: list) -> str:
    memory = ""
    if summary:
        memory += f"Summary of prior conversation:\n{summary}\n\n"
    for role, content in chat_history[-6:]:
        memory += f"{role.capitalize()}: {content}\n"
    return memory.strip()


# ── Non-streaming answer ──────────────────────────────────────────────────────

def get_answer(
    question: str,
    vector_paths: List[str],
    db_chunks_per_doc: List[list],
    doc_filenames: List[str],
    chat_history: list,
    summary: Optional[str] = None,
) -> dict:
    """
    Returns {"answer": str, "sources": list, "rewritten_query": str,
             "k_used": int, "response_time_ms": int}
    ROUGE-L is computed and logged internally; not returned.
    """
    t0 = time.time()

    # ── 1. Query rewriting ────────────────────────────────────────────────────
    rewritten = rewrite_query(question)
    k = compute_k(question)

    logger.info(
        "Rewritten Query: %r\n"
        "k              : %d",
        rewritten,
        k,
    )

    # ── 2. Hybrid retrieval ───────────────────────────────────────────────────
    all_vector_docs: List[LCDoc] = []
    all_bm25_docs:   List[LCDoc] = []

    for idx, (vpath, db_chunks) in enumerate(zip(vector_paths, db_chunks_per_doc)):
        fname = doc_filenames[idx] if idx < len(doc_filenames) else f"doc_{idx}"

        # Vector retrieval
        v_docs = _retrieve_vector(vpath, rewritten, k)
        for d in v_docs:
            d.metadata["source_filename"] = fname
        all_vector_docs.extend(v_docs)

        # BM25 retrieval
        for chunk in db_chunks:
            chunk._source_filename = fname
        b_docs = bm25_retrieve(rewritten, db_chunks, top_k=k)
        for d in b_docs:
            d.metadata["source_filename"] = fname
        all_bm25_docs.extend(b_docs)

    # ── 3. Fusion + reranking ─────────────────────────────────────────────────
    fused    = reciprocal_rank_fusion(all_vector_docs, all_bm25_docs)
    reranked = rerank(rewritten, fused)[:_CONTEXT_TOP_K]
    print(reranked)

    # ── 4. Build context ──────────────────────────────────────────────────────
    context_str, source_set = _build_context(reranked)
    memory_str = _build_memory(summary, chat_history)

    # ── 5. LLMChain with PromptTemplate + OutputParser ────────────────────────
    chain = LLMChain(llm=_build_llm(), prompt=QA_TEMPLATE)
    raw = chain.run(
        context=context_str,
        conversation_memory=memory_str,
        question=question,
        format_instructions=QA_FORMAT_INSTRUCTIONS,
    )

    parsed = safe_parse_qa(raw, question)
    final_answer = parsed.answer

    # Override sources from parser with retrieval sources if parser returned none
    if not parsed.sources:
        final_sources = source_set
    else:
        final_sources = [{"filename": s.filename, "page": s.page} for s in parsed.sources]

    # ── 6. Evaluation metrics (logged, not returned) ──────────────────────────
    retrieved_texts = [doc.page_content for doc in reranked]
    compute_and_log_metrics(question, final_answer, retrieved_texts, k=_CONTEXT_TOP_K)

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "answer":          final_answer,
        "sources":         final_sources,
        "rewritten_query": rewritten,
        "k_used":          k,
        "response_time_ms": elapsed_ms,
    }


# ── Streaming answer ──────────────────────────────────────────────────────────

def stream_answer(
    question: str,
    vector_paths: List[str],
    db_chunks_per_doc: List[list],
    doc_filenames: List[str],
    chat_history: list,
    summary: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Yields newline-delimited JSON strings for SSE streaming.
    Flow:  meta → token … token → done
    ROUGE-L is computed on the complete answer after streaming finishes.
    """

    # ── 1. Rewrite + retrieve (same as non-streaming) ─────────────────────────
    rewritten = rewrite_query(question)
    k = compute_k(question)

    all_vector_docs: List[LCDoc] = []
    all_bm25_docs:   List[LCDoc] = []

    for idx, (vpath, db_chunks) in enumerate(zip(vector_paths, db_chunks_per_doc)):
        fname = doc_filenames[idx] if idx < len(doc_filenames) else f"doc_{idx}"

        v_docs = _retrieve_vector(vpath, rewritten, k)
        for d in v_docs:
            d.metadata["source_filename"] = fname
        all_vector_docs.extend(v_docs)

        for chunk in db_chunks:
            chunk._source_filename = fname
        b_docs = bm25_retrieve(rewritten, db_chunks, top_k=k)
        for d in b_docs:
            d.metadata["source_filename"] = fname
        all_bm25_docs.extend(b_docs)

    fused    = reciprocal_rank_fusion(all_vector_docs, all_bm25_docs)
    reranked = rerank(rewritten, fused)[:_CONTEXT_TOP_K]

    context_str, source_set = _build_context(reranked)
    memory_str = _build_memory(summary, chat_history)

    # ── 2. Emit metadata first ────────────────────────────────────────────────
    yield json.dumps({
        "type":            "meta",
        "rewritten_query": rewritten,
        "k_used":          k,
        "sources":         source_set,
    }) + "\n"

    # ── 3. Build prompt using STREAMING_QA_TEMPLATE (plain Markdown) ─────────
    prompt_value = STREAMING_QA_TEMPLATE.format_prompt(
        context=context_str,
        conversation_memory=memory_str,
        question=question,
    )
    prompt_str = prompt_value.to_string()

    # ── 4. Stream tokens from LLM ────────────────────────────────────────────
    llm = _build_llm(streaming=True)
    t0 = time.time()
    full_answer = ""

    for chunk in llm.stream(prompt_str):
        token = chunk.content
        if token:
            full_answer += token
            yield json.dumps({"type": "token", "text": token}) + "\n"

    elapsed_ms = int((time.time() - t0) * 1000)

    # ── 5. Evaluation metrics (logged to backend console only) ───────────────
    retrieved_texts = [doc.page_content for doc in reranked]
    compute_and_log_metrics(question, full_answer, retrieved_texts, k=_CONTEXT_TOP_K)

    yield json.dumps({"type": "done", "response_time_ms": elapsed_ms}) + "\n"