# app/routes/delete.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Document, Answer, Question, ChatSession, DocumentChunk
import os, shutil

router = APIRouter(prefix="/delete", tags=["Delete"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.delete("/")
async def delete_chat(doc_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == doc_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    db.query(Question).filter(Question.document_id == doc_id).delete()
    db.query(Answer).filter(Answer.document_id == doc_id).delete()
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()

    sessions = db.query(ChatSession).filter(ChatSession.document_id == doc_id).all()
    for s in sessions:
        db.delete(s)

    # Clean up files
    base_name = os.path.splitext(document.filename)[0]
    vector_path = f"static/chroma_stores/{base_name}"
    if os.path.exists(vector_path):
        shutil.rmtree(vector_path)

    upload_path = document.original_path
    if upload_path and os.path.exists(upload_path):
        os.remove(upload_path)

    db.delete(document)
    db.commit()

    return {"message": f"Chat for '{document.filename}' deleted successfully"}