# backend/routes/upload.py
from fastapi import APIRouter, UploadFile, Depends, File
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.services.pdf_service import extract_text
from app.services.embedding_service import create_vectorstore
from app.models import Document, DocumentChunk
import shutil, os

router = APIRouter(prefix="/upload", tags=["Upload"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("/")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = f"{upload_dir}/{file.filename}"

    # Save file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(file_path)

    # Extract chunks
    chunks = extract_text(file_path)

    # Build vector store
    base_name = os.path.splitext(file.filename)[0]
    vector_path = create_vectorstore(chunks, base_name)

    # Persist to DB
    doc = Document(
        filename=file.filename,
        original_path=file_path,
        vector_path=vector_path,
        chunk_count=len(chunks),
        file_size_bytes=file_size,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Store chunks in DB for BM25 hybrid search
    for idx, chunk in enumerate(chunks):
        db_chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=idx,
            page_number=chunk.metadata.get("page_number", 0),
            content=chunk.page_content,
        )
        db.add(db_chunk)
    db.commit()

    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "chunk_count": len(chunks),
        "file_size_bytes": file_size,
    }


@router.post("/multi")
async def upload_multiple(files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    """Upload multiple documents in one request."""
    results = []
    for file in files:
        # Reuse single-upload logic by delegating
        ext = os.path.splitext(file.filename or "")[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({"filename": file.filename, "error": f"Unsupported type '{ext}'"})
            continue

        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = f"{upload_dir}/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)
        chunks = extract_text(file_path)
        base_name = os.path.splitext(file.filename)[0]
        vector_path = create_vectorstore(chunks, base_name)

        doc = Document(
            filename=file.filename,
            original_path=file_path,
            vector_path=vector_path,
            chunk_count=len(chunks),
            file_size_bytes=file_size,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        for idx, chunk in enumerate(chunks):
            db_chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=idx,
                page_number=chunk.metadata.get("page_number", 0),
                content=chunk.page_content,
            )
            db.add(db_chunk)
        db.commit()

        results.append({
            "document_id": doc.id,
            "filename": doc.filename,
            "chunk_count": len(chunks),
            "file_size_bytes": file_size,
        })

    return {"uploaded": results}