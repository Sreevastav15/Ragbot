# app/services/query_rewriter.py
"""
Query rewriting: converts vague or ambiguous questions into
retrieval-optimised queries using a lightweight LLM call.
Also computes query complexity to dynamically set k.
"""

from __future__ import annotations
import re
import os
from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def _llm() -> ChatGroq:
    return ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")


def rewrite_query(question: str) -> str:
    """
    Rewrites a vague user question into a more specific retrieval query.
    Returns the original question if rewriting fails or adds no value.
    """
    prompt = f"""You are a search query optimizer for a RAG system.
Rewrite the following user question into a precise, keyword-rich retrieval query
that will help find the most relevant document chunks.
Output ONLY the rewritten query, nothing else.

User question: {question}
Rewritten query:"""

    try:
        result = _llm().invoke(prompt)
        rewritten = result.content.strip().strip('"').strip("'")
        # Sanity check: must be non-empty and not too long
        if 5 <= len(rewritten) <= 300:
            return rewritten
    except Exception:
        pass
    return question  # fallback to original


def compute_k(question: str) -> int:
    """
    Dynamically choose how many chunks to retrieve based on query complexity.
    Simple heuristic + keyword analysis.
    """
    q = question.lower()
    word_count = len(q.split())

    # Multi-part or comparison questions → need more context
    is_complex = any(kw in q for kw in [
        "compare", "difference", "versus", "vs", "all", "list",
        "summarize", "explain", "describe", "overview", "both",
        "how many", "why", "what are all",
    ])

    if is_complex or word_count > 20:
        return 20   # wide net for complex queries
    elif word_count > 10:
        return 12
    else:
        return 8    # tight retrieval for simple factual questions