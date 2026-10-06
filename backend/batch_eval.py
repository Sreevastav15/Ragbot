"""
batch_eval.py — Automated RAGAS evaluation over an entire eval JSON file.

Mirrors the reference example structure:
  1. Load questions + ground truths from eval/doc1.json  (or doc2.json)
  2. For each question: retrieve context + generate answer via your RAG pipeline
  3. Build a Dataset from all rows
  4. Run ragas.evaluate() once over the whole dataset (more efficient than per-question)
  5. Print per-question rows and aggregate scores

USAGE:
    # From the backend/ directory:
    python batch_eval.py --doc-id 1 --json eval/doc1.json

    # To evaluate doc2.json against document ID 2:
    python batch_eval.py --doc-id 2 --json eval/doc2.json

ARGUMENTS:
    --doc-id   INT   The database ID of the document you uploaded via the app.
                     Find it in the Postgres `documents` table or from the upload
                     API response field "document_id".
    --json     PATH  Path to the eval JSON file (default: app/eval/doc1.json).
    --k        INT   Number of chunks to retrieve per question (default: 6).

HOW TO FIND YOUR DOCUMENT ID:
    After uploading a PDF through the frontend, the backend logs:
        "document_id": 3    ← use this number as --doc-id

    Or query Postgres directly:
        SELECT id, filename FROM documents ORDER BY upload_date DESC LIMIT 5;

INSTALL (if not already done):
    pip install ragas datasets

NOTE:
    This script uses your existing Groq + BGE setup — no new API keys needed.
    It imports directly from app/ so it must be run from the backend/ directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from app.services.google_embedding import GoogleTextEmbedding
from ragas.embeddings import LangchainEmbeddingsWrapper

# ── Make app/ importable when run from backend/ ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet noisy libraries so RAGAS output is readable
for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("batch_eval")

embedding_model = LangchainEmbeddingsWrapper(
    GoogleTextEmbedding()
)
# ── Load eval JSON ─────────────────────────────────────────────────────────────

def load_eval_json(json_path: str) -> list[dict]:
    """Load questions + ground-truth answers from the eval JSON file."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Eval JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    logger.info("Loaded %d Q&A pairs from %s", len(entries), path)
    return entries  # [{"question": ..., "answer": ...}, ...]


# ── Load document chunks from DB (for BM25 hybrid search) ─────────────────────

def load_doc_from_db(doc_id: int):
    """
    Returns (document_orm, db_chunks_list) for the given doc_id.
    Raises if the document isn't found.
    """
    from app.database import SessionLocal
    from app.models import Document, DocumentChunk

    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise ValueError(
                f"Document ID {doc_id} not found in the database.\n"
                f"Upload the PDF through the app first, then re-run with its ID."
            )
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )
        logger.info(
            "Loaded document '%s' (id=%d) with %d chunks",
            doc.filename, doc.id, len(chunks),
        )
        return doc, chunks
    finally:
        db.close()


# ── RAG pipeline (retrieve + generate) ────────────────────────────────────────

def rag_for_question(
    question: str,
    vector_path: str,
    db_chunks: list,
    doc_filename: str,
    k: int,
) -> tuple[str, list[str]]:
    """
    Run the full RAG pipeline for one question.
    Returns (generated_answer, retrieved_chunk_texts).

    Reuses qa_service.get_answer() so retrieval + generation is identical
    to what the live app does — no duplicate logic.
    """
    from app.services.qa_service import get_answer

    result = get_answer(
        question=question,
        vector_paths=[vector_path],
        db_chunks_per_doc=[db_chunks],
        doc_filenames=[doc_filename],
        chat_history=[],
        summary=None,
    )
    # get_answer returns the answer string; we also need the chunk texts.
    # Re-run retrieval to get them (or patch get_answer to return them).
    # For simplicity we re-run hybrid search directly here:
    from app.services.query_rewriter import rewrite_query, compute_k
    from app.services.hybrid_search import bm25_retrieve, reciprocal_rank_fusion
    from app.services.reranker import rerank
    from langchain_community.vectorstores import Chroma

    rewritten = rewrite_query(question)
    k_dynamic = compute_k(question)

    embeddings = GoogleTextEmbedding()
    vectorstore = Chroma(persist_directory=vector_path, embedding_function=embeddings)
    try:
        v_docs = vectorstore.max_marginal_relevance_search(rewritten, k=k_dynamic, fetch_k=max(k_dynamic * 3, k_dynamic))
    except Exception:
        v_docs = vectorstore.similarity_search(rewritten, k=k_dynamic)

    for d in v_docs:
        d.metadata["source_filename"] = doc_filename

    for chunk in db_chunks:
        chunk._source_filename = doc_filename
    b_docs = bm25_retrieve(rewritten, db_chunks, top_k=k_dynamic)
    for d in b_docs:
        d.metadata["source_filename"] = doc_filename

    fused    = reciprocal_rank_fusion(v_docs, b_docs)
    reranked = rerank(rewritten, fused)[:k]

    retrieved_texts = [doc.page_content for doc in reranked]
    return result["answer"], retrieved_texts


# ── RAGAS batch evaluation ─────────────────────────────────────────────────────

def run_ragas_batch(rows: list[dict]) -> object:
    """
    Run RAGAS over all rows at once (more efficient than one-by-one).

    rows: [{"question": ..., "answer": ..., "contexts": [...], "reference": ...}]
    Returns the RAGAS EvaluationResult object.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
    )
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLMWrapper
    from app.services.google_embedding import GoogleTextEmbedding 
    from ragas.run_config import RunConfig

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set.")

    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",   # or a Flash-Lite model if you hit limits; confirm ID in AI Studio
            google_api_key=google_api_key
        )
    )

    metrics = [
        answer_correctness,
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
    ]
    for m in metrics:
        m.llm = judge_llm

    answer_relevancy.strictness = 1
    dataset = Dataset.from_list(rows)
    return evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embedding_model,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=2, timeout=180, max_retries=10, max_wait=60),
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch RAGAS evaluation for RAGBot")
    parser.add_argument(
        "--doc-id", type=int, required=True,
        help="Database ID of the uploaded document (from the documents table)."
    )
    parser.add_argument(
        "--json", type=str, default="app/eval/doc1.json",
        help="Path to the eval JSON file (default: app/eval/doc1.json)."
    )
    parser.add_argument(
        "--k", type=int, default=6,
        help="Number of chunks to keep after reranking (default: 6)."
    )
    args = parser.parse_args()

    # ── 1. Load ground truths ─────────────────────────────────────────────────
    eval_entries = load_eval_json(args.json)
    questions    = [e["question"]  for e in eval_entries]
    ground_truths = [e["answer"]   for e in eval_entries]

    # ── 2. Load document from DB ──────────────────────────────────────────────
    doc, db_chunks = load_doc_from_db(args.doc_id)

    # ── 3. Build rows (retrieve + generate for each question) ─────────────────
    logger.info("Running RAG pipeline for %d questions …", len(questions))
    rows = []
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths), 1):
        logger.info("[%d/%d] %s", i, len(questions), question)
        answer, contexts = rag_for_question(
            question=question,
            vector_path=doc.vector_path,
            db_chunks=db_chunks,
            doc_filename=doc.filename,
            k=args.k,
        )
        rows.append({
            "question":  question,
            "answer":    answer,
            "contexts":  contexts,
            "ground_truth": ground_truth,
        })

    # ── 4. Run RAGAS over all rows at once ────────────────────────────────────
    logger.info("Running RAGAS evaluation on %d rows …", len(rows))
    scores = run_ragas_batch(rows)

    # ── 5. Print results (mirroring the reference example) ───────────────────
    print("\n" + "=" * 60)
    print("PER-QUESTION ROWS")
    print("=" * 60)
    for row in rows:
        print(f"\nQ : {row['question']}")
        print(f"A : {row['answer'][:120]}{'…' if len(row['answer']) > 120 else ''}")
        print(f"Ref: {row['ground_truth']}")
        print(f"Ctx: {len(row['contexts'])} chunks retrieved")

    print("\n" + "=" * 60)
    print("AGGREGATE RAGAS SCORES")
    print("=" * 60)
    print(scores)


if __name__ == "__main__":
    main()