# app/routes/eval_routes.py
"""Expose RAG evaluation metrics via API."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import SessionLocal
from app.models import Answer, Document

router = APIRouter(prefix="/eval", tags=["Evaluation"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/metrics/{doc_id}")
async def get_metrics(doc_id: int, db: Session = Depends(get_db)):
    """Aggregated RAG metrics for a document."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    answers = db.query(Answer).filter(Answer.document_id == doc_id).all()
    if not answers:
        return {"document_id": doc_id, "total_answers": 0, "metrics": {}}

    rouge_vals = [a.rouge_score for a in answers if a.rouge_score is not None]
    recall_vals = [a.retrieval_recall for a in answers if a.retrieval_recall is not None]
    mrr_vals = [a.mrr_score for a in answers if a.mrr_score is not None]
    rt_vals = [a.response_time_ms for a in answers if a.response_time_ms]

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "document_id": doc_id,
        "filename": doc.filename,
        "total_answers": len(answers),
        "metrics": {
            "avg_rouge_l": avg(rouge_vals),
            "avg_recall_at_k": avg(recall_vals),
            "avg_mrr": avg(mrr_vals),
            "avg_response_time_ms": avg(rt_vals),
        },
        "per_answer": [
            {
                "question": a.question_text,
                "rouge": a.rouge_score,
                "recall": a.retrieval_recall,
                "mrr": a.mrr_score,
                "response_time_ms": a.response_time_ms,
                "sources": a.sources,
            }
            for a in answers
        ],
    }


@router.get("/metrics/global")
async def global_metrics(db: Session = Depends(get_db)):
    """Global metrics across all documents."""
    answers = db.query(Answer).all()
    if not answers:
        return {"total_answers": 0}

    rouge_vals = [a.rouge_score for a in answers if a.rouge_score is not None]
    recall_vals = [a.retrieval_recall for a in answers if a.retrieval_recall is not None]
    mrr_vals = [a.mrr_score for a in answers if a.mrr_score is not None]
    rt_vals = [a.response_time_ms for a in answers if a.response_time_ms]

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "total_answers": len(answers),
        "avg_rouge_l": avg(rouge_vals),
        "avg_recall_at_k": avg(recall_vals),
        "avg_mrr": avg(mrr_vals),
        "avg_response_time_ms": avg(rt_vals),
    }