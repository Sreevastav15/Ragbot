# app/services/query_rewriter.py

"""
Query rewriting using LangChain 0.2+ Runnable pipeline.

Replaces deprecated LLMChain with:
    PromptTemplate | LLM | OutputParser

Uses:
- QUERY_REWRITER_TEMPLATE (PromptTemplate)
- QUERY_REWRITER_FORMAT_INSTRUCTIONS
- safe_parse_rewritten_query (custom parser)

Ensures:
- structured JSON output from LLM
- safe fallback to original query
"""

from __future__ import annotations
import os
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from app.prompts.prompts import QUERY_REWRITER_TEMPLATE
from app.parser.parser import (
    QUERY_REWRITER_FORMAT_INSTRUCTIONS,
    safe_parse_rewritten_query,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ── LLM Factory ──────────────────────────────────────────────────────────────

def _llm() -> ChatGroq:
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0,
    )


# ── Query Rewriting ──────────────────────────────────────────────────────────

def rewrite_query(question: str) -> str:
    """
    Rewrites a vague user question into a precise, keyword-rich retrieval
    query using LangChain 0.2 Runnable pipeline.

    Flow:
        PromptTemplate → LLM → StrOutputParser → safe_parse_rewritten_query

    Falls back to original question if parsing fails or output is invalid.
    """

    try:
        # Build runnable chain
        chain = (
            QUERY_REWRITER_TEMPLATE
            | _llm()
            | StrOutputParser()
        )

        # Invoke chain
        raw: str = chain.invoke({
            "question": question,
            "format_instructions": QUERY_REWRITER_FORMAT_INSTRUCTIONS,
        })

        # Parse + validate output
        return safe_parse_rewritten_query(raw, question)

    except Exception:
        return question  # safe fallback


# ── Dynamic Retrieval Size ───────────────────────────────────────────────────

def compute_k(question: str) -> int:
    """
    Dynamically choose how many chunks to retrieve based on query complexity.
    """

    q = question.lower()
    word_count = len(q.split())

    is_complex = any(kw in q for kw in [
        "compare", "difference", "versus", "vs", "all", "list",
        "summarize", "explain", "describe", "overview", "both",
        "how many", "why", "what are all",
    ])

    if is_complex or word_count > 20:
        return 20
    elif word_count > 10:
        return 12
    else:
        return 8