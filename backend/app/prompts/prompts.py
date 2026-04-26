# app/prompts/prompts.py
"""
All LangChain PromptTemplates used across the RAGBot pipeline.

Every template that calls the LLM receives {format_instructions} so the
model always knows what shape to return.  Import the ready-made chains
from here — never build raw f-string prompts in service files.
"""

from __future__ import annotations
from langchain_core.prompts import PromptTemplate


# ── 1. QA / Answer generation ─────────────────────────────────────────────────

QA_TEMPLATE = PromptTemplate(
    input_variables=[
        "context",
        "conversation_memory",
        "question",
        "format_instructions",
    ],
    template="""You are an expert document analyst. Your job is to answer the user's
question using ONLY the information found in the document context below.

### Conversation History
{conversation_memory}

### Document Context
{context}

### Instructions
- Base your answer solely on the Document Context above.
- Do NOT reference "conversation memory" or describe how you found the answer.
- If the context does not contain enough information, say:
  "The document does not contain enough information to answer this question."
- Write in clear, concise Markdown. No HTML. No LaTeX.
- Do not repeat the question back to the user.
- If multiple documents are referenced, state which document each fact comes from.
- Keep the answer focused and to the point.

### Output Format
{format_instructions}

### Question
{question}

Answer:""",
)


# ── 2. Streaming QA (plain-text, no structured output) ───────────────────────

STREAMING_QA_TEMPLATE = PromptTemplate(
    input_variables=[
        "context",
        "conversation_memory",
        "question",
    ],
    template="""You are an expert document analyst. Answer the user's question using
ONLY the information found in the document context below.

### Conversation History
{conversation_memory}

### Document Context
{context}

### Rules
1. Answer strictly from the Document Context. Do not invent information.
2. Do NOT say "based on conversation memory" or describe your reasoning process.
3. If the context lacks the answer, say: "The document does not contain enough
   information to answer this question."
4. Write in plain Markdown. No HTML. No LaTeX.
5. If you use information from multiple documents, cite which document each
   fact comes from inline, e.g., *(Source: filename.pdf, page 3)*.
6. At the very end, list all sources used:
   **Sources:** filename.pdf (page 2), filename2.pdf (page 5)

### Question
{question}

Answer:""",
)


# ── 3. Query Rewriter ─────────────────────────────────────────────────────────

QUERY_REWRITER_TEMPLATE = PromptTemplate(
    input_variables=["question", "format_instructions"],
    template="""You are a search query optimizer for a Retrieval-Augmented Generation (RAG) system.

Your task: Convert the user's question into a precise, keyword-rich retrieval query
that will surface the most relevant document passages from a vector database.

### Rules
- Output a SINGLE query sentence (≤ 30 words).
- Preserve all domain-specific terms exactly.
- Remove conversational filler ("tell me about", "can you explain", etc.).
- Do NOT add "Find documents about" or similar prefixes.
- Do NOT answer the question — only rewrite it.

### Output Format
{format_instructions}

### User Question
{question}

Rewritten Query:""",
)


# ── 4. Conversation Summariser ────────────────────────────────────────────────

SUMMARISER_TEMPLATE = PromptTemplate(
    input_variables=["messages", "existing_summary"],
    template="""Summarise the following conversation for a RAG chatbot's memory buffer.

### Existing Summary (if any)
{existing_summary}

### New Messages
{messages}

### Instructions
- Produce a concise paragraph (≤ 120 words) capturing the key topics discussed.
- Highlight any unanswered questions or important facts that must be remembered.
- Do NOT include source citations in the summary.
- Write in third-person present tense.

Summary:""",
)


# ── 5. Fallback / cannot-answer ───────────────────────────────────────────────

FALLBACK_TEMPLATE = PromptTemplate(
    input_variables=["question"],
    template="""The retrieved document context does not contain information relevant
to the following question:

"{question}"

Politely inform the user that the uploaded documents do not cover this topic
and suggest they upload a more relevant document or rephrase their question.

Response (Markdown, 2-3 sentences):""",
)