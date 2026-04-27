# app/services/eval_service.py
"""
RAG Evaluation Service.

Metrics computed per question/answer:
  1. ROUGE-L  — measures answer quality vs ground-truth reference answer
                (loaded from eval/doc*.json files).
  2. Recall@k — measures retrieval quality: fraction of ground-truth
                reference answer tokens that appear in the retrieved chunks.
                NOTE: this is *not* trivially 1.0 — it measures semantic
                coverage of the reference answer by the retrieved context.
  3. MRR      — Mean Reciprocal Rank: rank of the first retrieved chunk
                that contains substantial content from the reference answer.

ROOT CAUSE OF PREVIOUS BUGS:
  - Recall@k was computed by checking if the *question* appeared in retrieved
    chunks, which is nearly always True (trivially 1.0).
  - MRR was similarly question-based (trivially 1.0).
  - ROUGE-L was computed but the reference was not loaded from ground-truth
    JSON; it used the question itself as the reference, giving nonsense scores.

FIXED:
  - All metrics are computed against the *reference answer* from eval JSON.
  - Recall@k checks what fraction of reference answer key terms appear in
    the union of retrieved chunk text.
  - MRR checks at which rank a chunk first covers ≥30% of reference terms.
  - ROUGE-L compares the *generated answer* against the *reference answer*.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Ground-truth loader ───────────────────────────────────────────────────────

_EVAL_DIR = Path(__file__).parent.parent / "eval"
_gt_cache: Optional[dict] = None  # question → reference_answer


def _load_ground_truth() -> dict:
    """
    Load all eval JSON files once and cache them.
    Returns a dict mapping lower-cased question → reference answer string.
    """
    global _gt_cache
    if _gt_cache is not None:
        return _gt_cache

    _gt_cache = {}
    if not _EVAL_DIR.exists():
        logger.warning("[Eval] eval/ directory not found at %s", _EVAL_DIR)
        return _gt_cache

    for json_file in _EVAL_DIR.glob("*.json"):
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

    logger.info("[Eval] Loaded %d ground-truth Q&A pairs", len(_gt_cache))
    return _gt_cache


def _find_reference(question: str) -> Optional[str]:
    """Return the ground-truth reference answer for *question*, or None."""
    gt = _load_ground_truth()
    return gt.get(question.strip().lower())


# ── Tokenisation helper ───────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lower-case word tokeniser — strips punctuation."""
    return re.findall(r"\b\w+\b", text.lower())


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Longest Common Subsequence length (iterative, O(|a|·|b|) space)."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    # Use two-row DP to save memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def compute_rouge_l(generated: str, reference: str) -> float:
    """
    ROUGE-L F1 between *generated* answer and *reference* answer.

    Previously this was computed using the question as the reference,
    which gave nonsense scores. Now it correctly compares the model's
    generated answer against the human reference answer.
    """
    if not generated or not reference:
        return 0.0

    gen_tokens = _tokenize(generated)
    ref_tokens = _tokenize(reference)

    if not gen_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(gen_tokens, ref_tokens)
    precision = lcs / len(gen_tokens) if gen_tokens else 0.0
    recall    = lcs / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


# ── Retrieval Recall@k ────────────────────────────────────────────────────────

def compute_recall_at_k(retrieved_chunks: List[str], reference: str, k: int) -> float:
    """
    Recall@k: what fraction of reference answer key-terms are covered by
    the top-k retrieved chunks (union of their text).

    FIX: Previously this checked whether the *question* appeared in retrieved
    chunks — trivially True, always giving Recall@k = 1.0.
    Now it checks coverage of the *reference answer* terms.

    Args:
        retrieved_chunks: list of chunk text strings (already top-k after reranking)
        reference:        ground-truth reference answer string
        k:                number of chunks to consider (uses min(k, len(chunks)))
    """
    if not reference or not retrieved_chunks:
        return 0.0

    # Consider only top-k chunks
    top_chunks = retrieved_chunks[:k]
    retrieved_text = " ".join(top_chunks).lower()

    ref_tokens = set(_tokenize(reference))
    if not ref_tokens:
        return 0.0

    # Filter out stopwords for a more meaningful recall signal
    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "it", "its",
        "in", "on", "at", "to", "for", "of", "and", "or", "but",
        "with", "by", "from", "that", "this", "be", "as", "not",
        "have", "has", "had", "do", "does", "did",
    }
    key_terms = ref_tokens - _STOPWORDS
    if not key_terms:
        key_terms = ref_tokens  # fall back if everything was a stopword

    covered = sum(1 for term in key_terms if term in retrieved_text)
    recall = covered / len(key_terms)
    return round(recall, 4)


# ── MRR ───────────────────────────────────────────────────────────────────────

def compute_mrr(retrieved_chunks: List[str], reference: str, threshold: float = 0.3) -> float:
    """
    Mean Reciprocal Rank: 1/rank of the first chunk that covers ≥ threshold
    fraction of reference answer key-terms.

    FIX: Previously MRR was computed against the *question* (trivially 1.0).
    Now it measures at which retrieved chunk the reference answer first
    appears meaningfully.

    Args:
        retrieved_chunks: list of chunk text strings in retrieval rank order
        reference:        ground-truth reference answer string
        threshold:        minimum fraction of reference key-terms a chunk must
                          cover to be considered a "hit" (default 0.30)
    """
    if not reference or not retrieved_chunks:
        return 0.0

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "it", "its",
        "in", "on", "at", "to", "for", "of", "and", "or", "but",
        "with", "by", "from", "that", "this", "be", "as", "not",
        "have", "has", "had", "do", "does", "did",
    }

    ref_tokens = set(_tokenize(reference))
    key_terms = ref_tokens - _STOPWORDS
    if not key_terms:
        key_terms = ref_tokens

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        chunk_lower = chunk.lower()
        covered = sum(1 for term in key_terms if term in chunk_lower)
        coverage = covered / len(key_terms)
        if coverage >= threshold:
            return round(1.0 / rank, 4)

    return 0.0  # reference answer not found in any retrieved chunk


# ── Main entry point ─────────────────────────────────────────────────────────

def compute_and_log_metrics(
    question: str,
    generated_answer: str,
    retrieved_chunks: List[str],
    k: int = 6,
) -> dict:
    """
    Compute and log all three RAG metrics.

    Returns a dict with keys: rouge_l, recall_at_k, mrr
    All metrics require a ground-truth reference answer from the eval JSON.
    If no reference is found for the question, metrics are skipped.

    Args:
        question:          the original user question
        generated_answer:  the LLM-generated answer string
        retrieved_chunks:  list of retrieved chunk text strings (in rank order)
        k:                 retrieval depth for Recall@k
    """
    reference = _find_reference(question)

    if reference is None:
        # No ground-truth available — skip metric computation silently
        logger.debug("[Eval] No ground-truth found for question: %r", question[:80])
        return {"rouge_l": None, "recall_at_k": None, "mrr": None}

    rouge  = compute_rouge_l(generated_answer, reference)
    recall = compute_recall_at_k(retrieved_chunks, reference, k=k)
    mrr    = compute_mrr(retrieved_chunks, reference)

    # Truncate strings for readable log output
    chunk_preview = retrieved_chunks[0][:80].replace("\n", " ") if retrieved_chunks else "(none)"
    ref_preview   = reference[:80]
    gen_preview   = generated_answer[:80].replace("\n", " ")

    logger.info(
        "[Eval] Q: %r\n"
        "  Chunk       : %r\n"
        "  Reference   : %r\n"
        "  Generated   : %r\n"
        "  ROUGE-L     : %.4f\n"
        "  Recall@%d    : %.4f\n"
        "  MRR         : %.4f",
        question,
        chunk_preview,
        ref_preview,
        gen_preview,
        rouge,
        k,
        recall,
        mrr,
    )

    return {"rouge_l": rouge, "recall_at_k": recall, "mrr": mrr}