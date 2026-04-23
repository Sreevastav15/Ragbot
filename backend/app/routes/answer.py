# backend/routes/answer.py
from __future__ import annotations

import json
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from cachetools import TTLCache
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.database import SessionLocal
from app.models import Document, Answer, DocumentChunk
from app.services.qa_service import get_answer, stream_answer
from app.services.memory_service import (
    get_or_create_session,
    append_message,
    get_recent_messages,
)
from app.services.eval_service import compute_rouge, compute_retrieval_recall, compute_mrr

router = APIRouter(prefix="/answer", tags=["Answer"])
limiter = Limiter(key_func=get_remote_address)

# ── Simple in-memory cache (TTL = 5 min) ─────────────────────────────────────
_answer_cache: TTLCache = TTLCache(maxsize=256, ttl=300)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AnswerRequest(BaseModel):
    question: str
    document_ids: List[int]           # supports multiple docs
    stream: bool = False              # if True → SSE stream


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_key(question: str, doc_ids: List[int]) -> str:
    return f"{question.strip().lower()}|{sorted(doc_ids)}"


def _load_docs(db: Session, doc_ids: List[int]):
    docs = []
    for did in doc_ids:
        doc = db.query(Document).filter(Document.id == did).first()
        if not doc:
            raise HTTPException(404, f"Document {did} not found")
        docs.append(doc)
    return docs


def _load_chunks(db: Session, doc_id: int):
    return (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == doc_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )


# ── Standard (non-streaming) endpoint ────────────────────────────────────────

@router.post("/")
@limiter.limit("30/minute")
async def answer_question(request: Request, payload: AnswerRequest, db: Session = Depends(get_db)):
    if not payload.document_ids:
        raise HTTPException(400, "At least one document_id required")

    # Cache lookup
    ck = _cache_key(payload.question, payload.document_ids)
    if ck in _answer_cache:
        cached = _answer_cache[ck]
        cached["cached"] = True
        return cached

    docs = _load_docs(db, payload.document_ids)

    # Use first doc's session (primary document)
    primary_doc = docs[0]
    session = get_or_create_session(db, document_id=primary_doc.id, title=primary_doc.filename)
    append_message(db, session.id, "user", payload.question)
    recent_messages = get_recent_messages(db, session.id)

    chat_history = [
        ("human" if m["role"] == "user" else "ai", m["content"])
        for m in recent_messages
    ]
    full_history = []
    if session.last_summary:
        full_history.append(("system", f"Conversation summary: {session.last_summary}"))
    full_history.extend(chat_history)

    vector_paths = [d.vector_path for d in docs]
    db_chunks_per_doc = [_load_chunks(db, d.id) for d in docs]
    doc_filenames = [d.filename for d in docs]

    result = get_answer(
        question=payload.question,
        vector_paths=vector_paths,
        db_chunks_per_doc=db_chunks_per_doc,
        doc_filenames=doc_filenames,
        chat_history=full_history,
        summary=session.last_summary,
    )

    append_message(db, session.id, "assistant", result["answer"])

    # Compute RAG eval metrics
    retrieved_texts = [s.get("filename", "") for s in result.get("sources", [])]
    rouge = compute_rouge(payload.question, result["answer"])
    recall = compute_retrieval_recall(payload.question, retrieved_texts)
    mrr = compute_mrr(payload.question, retrieved_texts)

    # Save answer with metrics
    answer_entry = Answer(
        document_id=primary_doc.id,
        question_text=payload.question,
        answer_text=result["answer"],
        response_time_ms=result["response_time_ms"],
        rouge_score=rouge,
        retrieval_recall=recall,
        mrr_score=mrr,
        sources=result.get("sources", []),
    )
    db.add(answer_entry)
    db.commit()

    response = {
        "question": payload.question,
        "rewritten_query": result["rewritten_query"],
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "k_used": result["k_used"],
        "response_time_ms": result["response_time_ms"],
        "metrics": {"rouge": rouge, "recall_at_k": recall, "mrr": mrr},
        "session_id": session.id,
        "cached": False,
    }

    _answer_cache[ck] = response
    return response


# ── Streaming endpoint ────────────────────────────────────────────────────────

@router.post("/stream")
@limiter.limit("20/minute")
async def answer_stream(request: Request, payload: AnswerRequest, db: Session = Depends(get_db)):
    """Server-Sent Events streaming endpoint."""
    if not payload.document_ids:
        raise HTTPException(400, "At least one document_id required")

    docs = _load_docs(db, payload.document_ids)
    primary_doc = docs[0]
    session = get_or_create_session(db, document_id=primary_doc.id, title=primary_doc.filename)
    append_message(db, session.id, "user", payload.question)
    recent_messages = get_recent_messages(db, session.id)

    chat_history = [
        ("human" if m["role"] == "user" else "ai", m["content"])
        for m in recent_messages
    ]
    full_history = []
    if session.last_summary:
        full_history.append(("system", f"Conversation summary: {session.last_summary}"))
    full_history.extend(chat_history)

    vector_paths = [d.vector_path for d in docs]
    db_chunks_per_doc = [_load_chunks(db, d.id) for d in docs]
    doc_filenames = [d.filename for d in docs]

    # Collect full answer in background for persistence
    full_answer_parts: list[str] = []
    sources_ref: list[dict] = []
    timing_ref: list[int] = [0]

    def event_generator():
        for chunk_json in stream_answer(
            question=payload.question,
            vector_paths=vector_paths,
            db_chunks_per_doc=db_chunks_per_doc,
            doc_filenames=doc_filenames,
            chat_history=full_history,
            summary=session.last_summary,
        ):
            data = json.loads(chunk_json)
            if data["type"] == "meta":
                sources_ref.extend(data.get("sources", []))
            elif data["type"] == "token":
                full_answer_parts.append(data["text"])
            elif data["type"] == "done":
                timing_ref[0] = data.get("response_time_ms", 0)
            yield f"data: {chunk_json}\n\n"

    def wrapped_generator():
        yield from event_generator()
        # Persist after stream ends
        full_answer = "".join(full_answer_parts)
        if full_answer:
            append_message(db, session.id, "assistant", full_answer)
            rouge = compute_rouge(payload.question, full_answer)
            recall = compute_retrieval_recall(payload.question, [s.get("filename", "") for s in sources_ref])
            mrr = compute_mrr(payload.question, [s.get("filename", "") for s in sources_ref])
            answer_entry = Answer(
                document_id=primary_doc.id,
                question_text=payload.question,
                answer_text=full_answer,
                response_time_ms=timing_ref[0],
                rouge_score=rouge,
                retrieval_recall=recall,
                mrr_score=mrr,
                sources=sources_ref,
            )
            db.add(answer_entry)
            db.commit()

    return StreamingResponse(
        wrapped_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )