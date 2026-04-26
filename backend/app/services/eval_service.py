# app/services/eval_service.py
"""
RAG Evaluation — three metrics, all ground-truth driven.

┌─────────────────┬──────────────────────────────────────────────────────────┐
│ Metric          │ What it measures                                         │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ ROUGE-L         │ Quality of the generated answer vs. the reference answer │
│ Retrieval       │ Did the retrieved chunks actually contain the reference   │
│   Recall@k      │ answer's key content?                                    │
│ MRR             │ How highly ranked was the first truly relevant chunk?     │
└─────────────────┴──────────────────────────────────────────────────────────┘

── Original bugs fixed ────────────────────────────────────────────────────────

Bug 1 · ROUGE-L scored against wrong reference
  Old: scorer.score(question, answer)   ← LCS(question tokens, answer tokens)
       A long, correct answer shares almost no tokens with a short question
       → scores cluster near 0% (observed: 4.9%).
  Fix: _rouge_l_f1(generated_answer, reference_answer)
       reference_answer comes from the ground-truth JSON files (doc1/doc2).

Bug 2 · Recall@k denominator is Precision, not Recall
  Old: hits / min(k, len(retrieved_texts))
       Dividing hits by the number of chunks retrieved gives Precision@k,
       NOT Recall@k.  Recall requires a denominator of "how many relevant
       items exist in the pool", not "how many items were returned".
  Fix: Recall@k = relevant_in_top_k / min(total_relevant_in_full_set, k)
       A chunk is "relevant" if it contains ≥ RELEVANCE_THRESHOLD of the
       reference answer's content tokens.  Using the full retrieved set as
       the pool approximates the universe of retrievable relevant documents.

Bug 3 · MRR relevance signal was question keywords, not answer content
  Old: matches first chunk containing ANY question keyword (after stripping
       a small stopword set).  Because question keywords appear in nearly
       every retrieved chunk after reranking, rank 1 always matches →
       score trivially ≈ 1.0, which is meaningless.
  Fix: A chunk is relevant only if it passes the same RELEVANCE_THRESHOLD
       test against the reference answer tokens used for Recall.  This makes
       MRR reflect whether the correct information was ranked highly.
"""

from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# A chunk is "relevant" if it covers at least this fraction of the
# reference answer's content tokens.  Tunable: lower → more lenient.
RELEVANCE_THRESHOLD = 0.25

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "it", "its", "this", "that",
    "these", "those", "and", "or", "but", "not", "so", "if", "because",
    "which", "who", "what", "how", "when", "where", "why",
}

# ── Ground-truth corpus ───────────────────────────────────────────────────────

_EVAL_DIR = Path(__file__).parent.parent / "eval"


def _load_gt(filename: str) -> dict[str, str]:
    """Return {question_lower: reference_answer} from a ground-truth JSON."""
    path = _EVAL_DIR / filename
    if not path.exists():
        logger.warning("[Eval] Ground-truth file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        item["question"].strip().lower(): item["answer"].strip()
        for item in data
    }


_GT_DOC1: dict[str, str] = _load_gt("doc1.json")
_GT_DOC2: dict[str, str] = _load_gt("doc2.json")
_GT_ALL:  dict[str, str] = {**_GT_DOC1, **_GT_DOC2}


def _find_reference_answer(question: str) -> Optional[str]:
    """
    Return the ground-truth reference answer for `question`, or None.

    Strategy:
      1. Exact match on lowercased question.
      2. Token-overlap fuzzy match: a GT question must share ≥ 60 % of its
         own content tokens with the user question (handles minor rephrasing).
    """
    q_lower = question.strip().lower()

    if q_lower in _GT_ALL:
        return _GT_ALL[q_lower]

    q_tokens = set(re.findall(r"\w+", q_lower)) - _STOPWORDS
    best_score = 0.0
    best_answer: Optional[str] = None

    for gt_q, gt_a in _GT_ALL.items():
        gt_tokens = set(re.findall(r"\w+", gt_q)) - _STOPWORDS
        if not gt_tokens:
            continue
        overlap = len(q_tokens & gt_tokens) / len(gt_tokens)
        if overlap > best_score and overlap >= 0.6:
            best_score = overlap
            best_answer = gt_a

    return best_answer


# ── Shared helpers ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _content_tokens(text: str) -> set[str]:
    """All non-stopword tokens from `text`."""
    return set(_tokenize(text)) - _STOPWORDS


def _is_relevant(chunk_text: str, ref_tokens: set[str]) -> bool:
    """
    True if `chunk_text` covers at least RELEVANCE_THRESHOLD of the
    reference answer's content tokens.

    This is the single relevance oracle used by both Recall and MRR so
    that both metrics measure the same underlying property.
    """
    if not ref_tokens:
        return False
    chunk_tokens = _content_tokens(chunk_text)
    overlap = len(chunk_tokens & ref_tokens) / len(ref_tokens)
    return overlap >= RELEVANCE_THRESHOLD


# ── 1. ROUGE-L ────────────────────────────────────────────────────────────────

def _lcs_length(a: list[str], b: list[str]) -> int:
    """Space-optimised O(m*n) LCS length."""
    m, n = len(a), len(b)
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


def _rouge_l_f1(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L F1 between `hypothesis` (generated) and `reference` (ground-truth).

    BUG FIXED: both arguments are *answers*.  The original code passed
    (question, answer) so the LCS was measured between the user's question
    and the model's reply, giving near-zero scores for any answer longer
    than the question.
    """
    hyp_tok = _tokenize(hypothesis)
    ref_tok = _tokenize(reference)

    if not hyp_tok or not ref_tok:
        return 0.0

    lcs       = _lcs_length(hyp_tok, ref_tok)
    precision = lcs / len(hyp_tok)
    recall    = lcs / len(ref_tok)

    if precision + recall == 0.0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 4)


# ── 2. Retrieval Recall@k ─────────────────────────────────────────────────────

def _compute_retrieval_recall(
    retrieved_texts: list[str],
    reference: str,
    k: int,
) -> float:
    """
    Recall@k = relevant_in_top_k / min(total_relevant_in_pool, k)

    BUG FIXED: the original divided by min(k, len(retrieved_texts)) which
    is the number of items *returned* — that formula is Precision@k, not
    Recall@k.  Recall's denominator must represent how many relevant items
    exist in the pool we are retrieving from.  We approximate the pool as
    the full retrieved list; capping at k keeps the score in [0, 1] and
    reflects that we can never do better than retrieving all k top results.
    """
    if not retrieved_texts:
        return 0.0

    ref_tokens = _content_tokens(reference)
    if not ref_tokens:
        return 0.0

    # How many of the top-k chunks are relevant?
    relevant_in_top_k = sum(
        1 for t in retrieved_texts[:k] if _is_relevant(t, ref_tokens)
    )

    # How many relevant chunks exist across the full retrieved set?
    total_relevant = sum(
        1 for t in retrieved_texts if _is_relevant(t, ref_tokens)
    )

    if total_relevant == 0:
        return 0.0

    # Cap denominator at k: perfect recall means we got all relevant docs
    # within our top-k budget.
    denominator = min(total_relevant, k)
    return round(relevant_in_top_k / denominator, 4)


# ── 3. MRR (per-query Reciprocal Rank) ───────────────────────────────────────

def _compute_mrr(retrieved_texts: list[str], reference: str) -> float:
    """
    Reciprocal Rank (RR) for a single query.

    RR = 1 / rank_of_first_relevant_chunk   (0 if none found)

    BUG FIXED (two issues):
      1. Old code matched question keywords against chunks.  Because question
         words appear in almost every passage after BM25 + reranking, rank 1
         nearly always matched → RR ≈ 1.0 regardless of actual relevance.
         Now relevance is judged against the reference answer tokens using
         the same _is_relevant() oracle as Recall, so RR reflects whether
         the *correct information* was retrieved at a high rank.

      2. Old stopword list for the question was much smaller than the one
         used for Recall, creating an inconsistency between the two metrics.
         Both now share _STOPWORDS and _is_relevant().

    Per-answer RRs can be averaged by the caller / eval_routes endpoint to
    obtain the true dataset-level MRR.
    """
    if not retrieved_texts:
        return 0.0

    ref_tokens = _content_tokens(reference)
    if not ref_tokens:
        return 0.0

    for rank, text in enumerate(retrieved_texts, start=1):
        if _is_relevant(text, ref_tokens):
            return round(1.0 / rank, 4)

    return 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def compute_and_log_metrics(
    question: str,
    generated_answer: str,
    retrieved_texts: list[str],
    k: int = 6,
) -> dict[str, Optional[float]]:
    """
    Compute ROUGE-L, Retrieval Recall@k, and MRR against the ground-truth
    reference answer for `question`.

    Returns
    -------
    {
        "rouge_l":  float | None,   # ROUGE-L F1, generated vs reference answer
        "recall_k": float | None,   # Recall@k, relevant chunks in top-k
        "mrr":      float | None,   # Reciprocal Rank of first relevant chunk
    }

    All three values are None when no ground-truth entry matches `question`.
    Results are logged to the backend console and returned for DB persistence.
    Nothing is forwarded to the frontend.

    Parameters
    ----------
    question         : original user question (used to look up ground truth)
    generated_answer : full answer text produced by the LLM
    retrieved_texts  : chunk texts in retrieval rank order (index 0 = top rank)
    k                : budget for Recall@k (should match _CONTEXT_TOP_K in qa_service)
    """
    reference = _find_reference_answer(question)

    if reference is None:
        logger.info(
            "[Eval] Q: %r — no ground-truth match, skipping metrics.",
            question[:80],
        )
        return {"rouge_l": None, "recall_k": None, "mrr": None}

    rouge  = _rouge_l_f1(generated_answer, reference)
    recall = _compute_retrieval_recall(retrieved_texts, reference, k)
    mrr    = _compute_mrr(retrieved_texts, reference)

    logger.info(
        "[Eval] Q: %r\n"
        "  Chunk       : %s\n"
        "  Reference   : %s\n"
        "  Generated   : %s\n"
        "  ROUGE-L     : %.4f\n"
        "  Recall@%-3d  : %.4f\n"
        "  MRR         : %.4f",
        question[:80],
        str(retrieved_texts)[:80],
        reference[:100],
        generated_answer[:100],
        rouge,
        k,
        recall,
        mrr,
    )

    return {"rouge_l": rouge, "recall_k": recall, "mrr": mrr}