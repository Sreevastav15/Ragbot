# app/services/memory_service.py
"""
Token-aware memory management.
Instead of counting raw messages, we track estimated token usage
and summarise/prune when the window exceeds the budget.
"""

from __future__ import annotations
from sqlalchemy.orm import Session
from app.models import ChatSession, ChatMessage
from datetime import datetime, timezone
from langchain_groq import ChatGroq
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Tunables ──────────────────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS = 3000      # prune when window exceeds this many tokens
KEEP_RECENT_TOKENS = 1000      # always preserve at least this many recent tokens
MAX_SUMMARY_CHARS  = 3000      # cap on stored summary text


# ── Token estimator ───────────────────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    """~4 chars per token (GPT-style rough estimate)."""
    return max(1, len(text) // 4)


# ── Session management ────────────────────────────────────────────────────────
def get_or_create_session(db: Session, document_id: int = None, title: str = "") -> ChatSession:
    if document_id:
        session = db.query(ChatSession).filter(ChatSession.document_id == document_id).first()
        if session:
            return session

    session = ChatSession(document_id=document_id, title=title or "")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def append_message(db: Session, session_id: int, role: str, content: str) -> ChatMessage:
    token_est = estimate_tokens(content)
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        token_estimate=token_est,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)

    # Update session cumulative token counter
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.total_tokens_used = (session.total_tokens_used or 0) + token_est
        db.commit()

    _maybe_summarize_token_aware(db, session_id)
    return msg


def get_recent_messages(db: Session, session_id: int) -> list[dict]:
    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in msgs]


# ── Token-aware summarise & prune ─────────────────────────────────────────────
def _maybe_summarize_token_aware(db: Session, session_id: int):
    """
    Summarise older messages when the total token window exceeds MAX_CONTEXT_TOKENS.
    Always keeps the most recent KEEP_RECENT_TOKENS worth of messages.
    """
    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    total_tokens = sum(m.token_estimate for m in msgs)
    if total_tokens <= MAX_CONTEXT_TOKENS:
        return  # within budget, nothing to do

    # Walk backwards collecting recent messages until we hit the budget
    recent_msgs = []
    recent_token_count = 0
    for m in reversed(msgs):
        if recent_token_count + m.token_estimate > KEEP_RECENT_TOKENS:
            break
        recent_msgs.insert(0, m)
        recent_token_count += m.token_estimate

    old_msgs = [m for m in msgs if m not in recent_msgs]
    if not old_msgs:
        return

    text_to_summarize = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in old_msgs
    )

    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
    prompt_msgs = [
        {"role": "system", "content": "Summarize the following conversation concisely. Preserve key facts and decisions."},
        {"role": "user", "content": text_to_summarize},
    ]

    try:
        summary_resp = llm.invoke(prompt_msgs)
        summary = summary_resp.content[:MAX_SUMMARY_CHARS]
    except Exception:
        summary = text_to_summarize[:MAX_SUMMARY_CHARS]

    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        if session.last_summary:
            session.last_summary = (session.last_summary + "\n" + summary).strip()[:MAX_SUMMARY_CHARS]
        else:
            session.last_summary = summary
        session.last_summary_at = datetime.now(timezone.utc)
        db.add(session)

    for m in old_msgs:
        db.delete(m)

    db.commit()