# app/services/qa_service.py
"""
Core QA service.
Features:
  - Query rewriting
  - Dynamic k (based on query complexity)
  - Hybrid BM25 + vector retrieval
  - CrossEncoder reranking
  - Multi-document reasoning with source citation
  - Streaming SSE support
"""

from __future__ import annotations
import os
import time
from typing import Generator, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document as LCDoc

from app.services.google_embedding import GoogleTextEmbedding
from app.services.reranker import rerank
from app.services.hybrid_search import bm25_retrieve, reciprocal_rank_fusion
from app.services.query_rewriter import rewrite_query, compute_k
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def _build_llm() -> ChatGroq:
    return ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")


def _retrieve_from_vectorstore(vector_path: str, query: str, k: int) -> List[LCDoc]:
    embeddings = GoogleTextEmbedding()
    vectorstore = Chroma(persist_directory=vector_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def get_answer(
    question: str,
    vector_paths: List[str],          # list of vector store dirs (multi-doc)
    db_chunks_per_doc: List[list],     # list of DB DocumentChunk lists per doc
    doc_filenames: List[str],          # filename per doc (for citation)
    chat_history: list,
    summary: Optional[str] = None,
) -> dict:
    """
    Returns {"answer": str, "sources": list, "rewritten_query": str,
             "k_used": int, "response_time_ms": int}
    """
    t0 = time.time()

    # 1. Query rewriting
    rewritten = rewrite_query(question)
    k = compute_k(question)

    # 2. Retrieve from ALL documents
    all_vector_docs: List[LCDoc] = []
    all_bm25_docs: List[LCDoc] = []

    for idx, (vpath, db_chunks) in enumerate(zip(vector_paths, db_chunks_per_doc)):
        fname = doc_filenames[idx] if idx < len(doc_filenames) else f"doc_{idx}"

        # Vector retrieval
        v_docs = _retrieve_from_vectorstore(vpath, rewritten, k)
        for d in v_docs:
            d.metadata["source_filename"] = fname
        all_vector_docs.extend(v_docs)

        # BM25 retrieval — tag source filename on DB chunks
        for chunk in db_chunks:
            chunk._source_filename = fname
        b_docs = bm25_retrieve(rewritten, db_chunks, top_k=k)
        for d in b_docs:
            d.metadata["source_filename"] = fname
        all_bm25_docs.extend(b_docs)

    # 3. Hybrid fusion
    fused = reciprocal_rank_fusion(all_vector_docs, all_bm25_docs)

    # 4. CrossEncoder reranking – keep top 6
    reranked = rerank(rewritten, fused)[:6]

    # 5. Build context with source tags
    context_blocks = ""
    source_set = []
    for doc in reranked:
        fname = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page_number", "?")
        context_blocks += f"\n[Source: {fname}, Page {page}]\n{doc.page_content}\n"
        source_set.append({"filename": fname, "page": page})

    # 6. Build conversation memory
    conversation_memory = ""
    if summary:
        conversation_memory += f"\nSummary of previous conversation:\n{summary}\n"
    for role, content in chat_history[-6:]:
        conversation_memory += f"{role.upper()}: {content}\n"

    # 7. Final prompt
    llm = _build_llm()
    prompt = f"""You are an intelligent assistant that answers questions using:
1. Previous conversation memory
2. Retrieved document chunks (may come from multiple documents)

Conversation Memory:
{conversation_memory}

==========
Document Chunks:
{context_blocks}
==========

Rules:
1. Use BOTH document excerpts and conversation memory.
2. If the question refers to previous conversation, use memory.
3. Only say "I am not sure" if BOTH memory and documents lack content.
4. Do NOT use HTML.
5. At the bottom of your answer, list each source used as:
   **Sources:** [filename, page X], [filename, page Y]
6. Do NOT use LaTeX or mathematical equation formatting.
   Write equations in plain text only.
7. If multiple documents are referenced, specify which document each fact comes from.

Question: {question}

Answer in Markdown:"""

    result = llm.invoke(prompt)
    final_answer = result.content.strip()

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "answer": final_answer,
        "sources": source_set,
        "rewritten_query": rewritten,
        "k_used": k,
        "response_time_ms": elapsed_ms,
    }


def stream_answer(
    question: str,
    vector_paths: List[str],
    db_chunks_per_doc: List[list],
    doc_filenames: List[str],
    chat_history: list,
    summary: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Yields token chunks for SSE streaming.
    First yields metadata as a special JSON line, then streams tokens.
    """
    import json

    rewritten = rewrite_query(question)
    k = compute_k(question)

    all_vector_docs: List[LCDoc] = []
    all_bm25_docs: List[LCDoc] = []

    for idx, (vpath, db_chunks) in enumerate(zip(vector_paths, db_chunks_per_doc)):
        fname = doc_filenames[idx] if idx < len(doc_filenames) else f"doc_{idx}"
        v_docs = _retrieve_from_vectorstore(vpath, rewritten, k)
        for d in v_docs:
            d.metadata["source_filename"] = fname
        all_vector_docs.extend(v_docs)
        for chunk in db_chunks:
            chunk._source_filename = fname
        b_docs = bm25_retrieve(rewritten, db_chunks, top_k=k)
        for d in b_docs:
            d.metadata["source_filename"] = fname
        all_bm25_docs.extend(b_docs)

    fused = reciprocal_rank_fusion(all_vector_docs, all_bm25_docs)
    reranked = rerank(rewritten, fused)[:6]

    context_blocks = ""
    source_set = []
    for doc in reranked:
        fname = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page_number", "?")
        context_blocks += f"\n[Source: {fname}, Page {page}]\n{doc.page_content}\n"
        source_set.append({"filename": fname, "page": page})

    conversation_memory = ""
    if summary:
        conversation_memory += f"\nSummary of previous conversation:\n{summary}\n"
    for role, content in chat_history[-6:]:
        conversation_memory += f"{role.upper()}: {content}\n"

    prompt = f"""You are an intelligent assistant that answers questions using:
1. Previous conversation memory
2. Retrieved document chunks (may come from multiple documents)

Conversation Memory:
{conversation_memory}

==========
Document Chunks:
{context_blocks}
==========

Rules:
1. Use BOTH document excerpts and conversation memory.
2. If the question refers to previous conversation, use memory.
3. Only say "I am not sure" if BOTH memory and documents lack content.
4. Do NOT use HTML.
5. At the bottom, list each source as: **Sources:** [filename, page X]
6. Do NOT use LaTeX.
7. Specify which document each fact comes from if multiple documents used.

Question: {question}

Answer in Markdown:"""

    llm = _build_llm()

    # Yield metadata header first
    yield json.dumps({
        "type": "meta",
        "rewritten_query": rewritten,
        "k_used": k,
        "sources": source_set,
    }) + "\n"

    # Stream tokens
    t0 = time.time()
    for chunk in llm.stream(prompt):
        token = chunk.content
        if token:
            yield json.dumps({"type": "token", "text": token}) + "\n"

    elapsed_ms = int((time.time() - t0) * 1000)
    yield json.dumps({"type": "done", "response_time_ms": elapsed_ms}) + "\n"