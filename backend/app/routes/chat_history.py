# app/routes/chat_history.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Document, ChatMessage, ChatSession, Question, Answer

router = APIRouter(prefix="/chathistory", tags=["Chathistory"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/session")
async def load_session(doc_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == doc_id).first()
    if not document:
        raise HTTPException(404, "Document not found")

    session = db.query(ChatSession).filter(ChatSession.document_id == doc_id).first()
    if not session:
        raise HTTPException(404, "Chat session not found")

    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    messages = [
        {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
        for m in msgs
    ]

    return {
        "document_id": document.id,
        "session_id": session.id,
        "filename": document.filename,
        "summary": session.last_summary or "",
        "total_tokens_used": session.total_tokens_used or 0,
        "messages": messages,
    }


@router.get("/full")
async def full_history(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    answers = (
        db.query(Answer)
        .filter(Answer.document_id == doc_id)
        .order_by(Answer.created_at.asc())
        .all()
    )

    return {
        "doc_id": doc_id,
        "filename": doc.filename,
        "conversation": [
            {
                "question": a.question_text,
                "answer": a.answer_text,
                "sources": a.sources or [],
                "response_time_ms": a.response_time_ms,
                "metrics": {
                    "rouge": a.rouge_score,
                    "recall": a.retrieval_recall,
                    "mrr": a.mrr_score,
                },
            }
            for a in answers
        ],
    }


@router.get("/all")
async def all_chats(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.upload_date.desc()).all()
    return [
        {
            "doc_id": d.id,
            "filename": d.filename,
            "chunk_count": d.chunk_count,
            "file_size_bytes": d.file_size_bytes,
            "upload_date": d.upload_date.isoformat() if d.upload_date else None,
        }
        for d in docs
    ]