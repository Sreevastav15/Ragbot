# app/parser/parser.py
"""
LangChain output parsers for all LLM chains in the RAGBot pipeline.

Each parser defines a strict schema for what the LLM must return, and
exposes `format_instructions` that are injected into every PromptTemplate
so the model always knows the exact format expected.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser


# ── 1. QA Answer parser ───────────────────────────────────────────────────────

class Source(BaseModel):
    filename: str = Field(description="Name of the source document")
    page: int = Field(description="Page number of the source")


class QAAnswer(BaseModel):
    """Structured output for the main question-answering chain."""
    answer: str = Field(
        description=(
            "A clear, concise answer to the user's question based strictly on "
            "the provided document context. Written in plain Markdown. "
            "Do NOT include HTML or LaTeX."
        )
    )
    sources: List[Source] = Field(
        description="List of sources (filename + page) that support the answer.",
        default_factory=list,
    )
    confidence: str = Field(
        description=(
            "One of: 'high' (answer clearly supported by context), "
            "'medium' (partial support), 'low' (inferred or weak support)."
        ),
        default="medium",
    )


qa_parser = PydanticOutputParser(pydantic_object=QAAnswer)
QA_FORMAT_INSTRUCTIONS = qa_parser.get_format_instructions()


# ── 2. Query Rewriter parser ──────────────────────────────────────────────────

class RewrittenQuery(BaseModel):
    """Output for the query-rewriting chain."""
    rewritten_query: str = Field(
        description=(
            "A precise, keyword-rich retrieval query derived from the user's "
            "original question. Must be a single sentence, ≤ 30 words, with "
            "no filler phrases like 'Find documents about'."
        )
    )


query_rewriter_parser = PydanticOutputParser(pydantic_object=RewrittenQuery)
QUERY_REWRITER_FORMAT_INSTRUCTIONS = query_rewriter_parser.get_format_instructions()


# ── 3. Streaming / free-text answer parser ────────────────────────────────────
# Used when streaming tokens — no structured output needed, just clean text.

str_parser = StrOutputParser()


# ── Helper: safe parse with fallback ─────────────────────────────────────────

def safe_parse_qa(raw: str, question: str = "") -> QAAnswer:
    """
    Attempt structured parse; fall back to a plain-text QAAnswer on failure.
    """
    try:
        return qa_parser.parse(raw)
    except Exception:
        # Graceful degradation: wrap raw string as answer
        return QAAnswer(answer=raw.strip(), sources=[], confidence="low")


def safe_parse_rewritten_query(raw: str, original: str) -> str:
    """
    Attempt to parse a RewrittenQuery; fall back to the original question.
    """
    try:
        parsed = query_rewriter_parser.parse(raw)
        q = parsed.rewritten_query.strip().strip('"').strip("'")
        if 5 <= len(q) <= 300:
            return q
    except Exception:
        pass
    return original