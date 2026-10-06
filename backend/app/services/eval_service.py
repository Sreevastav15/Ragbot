# app/services/eval_service.py
"""
RAG Evaluation Service — powered by RAGAS.

Mirrors the structure of the reference example:
  - Ground-truth references loaded from eval/doc*.json (question → reference answer)
  - For each answered question, builds a single-row Dataset with:
        question, answer, contexts (retrieved chunk texts), reference
  - Passes it to ragas.evaluate() with 5 metrics:
        answer_correctness, answer_relevancy, faithfulness,
        context_precision, context_recall
  - Scores are logged to the backend console only (not sent to the frontend).

HOW IT HOOKS INTO YOUR PIPELINE:
  qa_service.py already calls:
      compute_and_log_metrics(question, final_answer, retrieved_texts, k=_CONTEXT_TOP_K)
  where retrieved_texts is a list of reranked chunk page_content strings.
  No changes needed in qa_service.py — this file is a drop-in replacement.

RAGAS METRICS (brief):
  answer_correctness  — semantic similarity + F1 vs reference answer
  answer_relevancy    — does the answer address the question?
  faithfulness        — are all answer claims grounded in the retrieved context?
  context_precision   — are relevant chunks ranked above irrelevant ones?
  context_recall      — does the retrieved context cover the reference answer?

NOTE: RAGAS makes LLM calls for each metric. We override the default judge
      with your existing Groq LLM so no new API key is needed.

INSTALL:  pip install ragas datasets
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional
from ragas.embeddings import LangchainEmbeddingsWrapper
from app.services.google_embedding import GoogleTextEmbedding  # or your HF one
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper

logger = logging.getLogger(__name__)

# ── Ground-truth loader (same source: eval/doc*.json) ─────────────────────────

_EVAL_DIR = Path(__file__).parent.parent / "eval"
_gt_cache: Optional[dict] = None  # question (lower) → reference answer string
embedding_model = LangchainEmbeddingsWrapper(
    GoogleTextEmbedding()
)


def _load_ground_truth() -> dict:
    """
    Load all eval JSON files once and cache.
    Returns {lowercased_question: reference_answer}.
    """
    global _gt_cache
    if _gt_cache is not None:
        return _gt_cache

    _gt_cache = {}
    if not _EVAL_DIR.exists():
        logger.warning("[Eval] eval/ directory not found at %s", _EVAL_DIR)
        return _gt_cache

    for json_file in sorted(_EVAL_DIR.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                q = entry.get("question", "").strip().lower()
                a = entry.get("answer", "").strip()
                if q and a:
                    _gt_cache[q] = a
        except Exception as exc:
            logger.warning("[Eval] Failed to load %s: %s", json_file, exc)

    logger.info("[Eval] Loaded %d ground-truth Q&A pairs from eval/", len(_gt_cache))
    return _gt_cache


def _find_reference(question: str) -> Optional[str]:
    """Return the ground-truth reference answer, or None if not in eval JSONs."""
    return _load_ground_truth().get(question.strip().lower())


# ── RAGAS single-turn evaluation ───────────────────────────────────────────────

class SafeChatGroq(ChatGroq):
    def _combine_llm_outputs(self, llm_outputs):
        # Ignore token usage aggregation completely
        return {}

def _run_ragas_single(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> dict:
    """
    Evaluate one RAG turn with RAGAS — mirrors the reference example exactly.

    Builds a single-row Dataset:
        { question, answer, contexts, reference }
    then calls ragas.evaluate() with all 5 metrics.
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

    # ── Inject your existing Groq LLM as the RAGAS judge ─────────────────────
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY not set — cannot run RAGAS evaluation.")

    judge_llm = LangchainLLMWrapper(
        SafeChatGroq(
            groq_api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0,
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

    # ── Single-row Dataset (lists of length 1, matching the example) ──────────
    rows =[]
    rows.append(
        {
            "question":  question,
            "answer":    answer,
            "contexts":  contexts,   # list of retrieved chunk texts
            "ground_truth": ground_truth,  # ground-truth answer string
        }
    )
    dataset = Dataset.from_list(rows)

    result = evaluate(dataset, metrics=metrics, embeddings=embedding_model, raise_exceptions=False)
    row = result.to_pandas().iloc[0]

    def _f(key: str) -> Optional[float]:
        try:
            v = float(row[key])
            return round(v, 4) if v == v else None  # NaN → None
        except (KeyError, TypeError, ValueError):
            return None

    return {
        "answer_correctness":  _f("answer_correctness"),
        "answer_relevancy":    _f("answer_relevancy"),
        "faithfulness":        _f("faithfulness"),
        "context_precision":   _f("context_precision"),
        "context_recall":      _f("context_recall"),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_and_log_metrics(
    question: str,
    generated_answer: str,
    retrieved_chunks: List[str],
    k: int = 6,
) -> dict:
    """
    Called automatically by qa_service.py after every answered question.
    Signature is identical to the old eval_service — no changes needed elsewhere.

    Args:
        question:          Original user question (used to look up ground truth).
        generated_answer:  The LLM-generated answer string.
        retrieved_chunks:  Reranked chunk page_content strings — already passed
                           correctly by qa_service, no change needed there.
        k:                 Kept for backward compatibility; ignored by RAGAS.

    Returns:
        dict with RAGAS metric scores (all None if no ground truth found).
    """
    reference = _find_reference(question)
    print(reference)

    if reference is None:
        logger.debug(
            "[Eval] No ground-truth entry for %r — skipping RAGAS.", question[:100]
        )
        return _null_metrics()

    try:
        scores = _run_ragas_single(
            question=question,
            answer=generated_answer,
            contexts=retrieved_chunks,
            ground_truth=reference,
        )
        _log_scores(question, reference, generated_answer, scores)
        return scores

    except Exception as exc:
        logger.error("[Eval] RAGAS evaluation failed: %s", exc, exc_info=True)
        return _null_metrics()


def _null_metrics() -> dict:
    return {
        "answer_correctness": None,
        "answer_relevancy":   None,
        "faithfulness":       None,
        "context_precision":  None,
        "context_recall":     None,
    }


def _log_scores(question: str, reference: str, answer: str, scores: dict) -> None:
    logger.info(
        "[Eval-RAGAS]\n"
        "  Question           : %r\n"
        "  Reference          : %r\n"
        "  Generated          : %r\n"
        "  Answer Correctness : %s\n"
        "  Answer Relevancy   : %s\n"
        "  Faithfulness       : %s\n"
        "  Context Precision  : %s\n"
        "  Context Recall     : %s",
        question[:100],
        reference[:100],
        answer[:100],
        scores["answer_correctness"],
        scores["answer_relevancy"],
        scores["faithfulness"],
        scores["context_precision"],
        scores["context_recall"],
    )