# app/services/eval_service.py
"""
RAG Evaluation Metrics (all free, no external APIs).
  - ROUGE-L  : answer quality vs question overlap
  - Recall@k : how many relevant sources were retrieved
  - MRR      : Mean Reciprocal Rank of the first relevant result
  - Hallucination proxy: grounding score (how much of the answer is supported by retrieved context)
"""

from __future__ import annotations
import re
from typing import List


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def _lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge(question: str, answer: str) -> float:
    """
    ROUGE-L F1 between question tokens and answer tokens.
    Serves as a rough answer-relevance proxy.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(question, answer)
        return round(scores["rougeL"].fmeasure, 4)
    except Exception:
        # Fallback: manual ROUGE-L
        def tok(t: str) -> List[str]:
            return re.findall(r"\w+", t.lower())
        q_tok, a_tok = tok(question), tok(answer)
        if not q_tok or not a_tok:
            return 0.0
        lcs = _lcs_length(q_tok, a_tok)
        p = lcs / len(a_tok)
        r = lcs / len(q_tok)
        if p + r == 0:
            return 0.0
        return round(2 * p * r / (p + r), 4)


# ── Retrieval Recall@k ────────────────────────────────────────────────────────

def compute_retrieval_recall(question: str, retrieved_texts: List[str], k: int = 5) -> float:
    """
    Approximate Recall@k.
    Checks how many of the top-k retrieved sources contain at least one key
    token from the question.
    """
    if not retrieved_texts:
        return 0.0
    key_tokens = set(re.findall(r"\w+", question.lower())) - {
        "the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
        "who", "does", "do", "of", "in", "on", "at", "to", "for", "with",
    }
    if not key_tokens:
        return 0.0
    hits = 0
    for text in retrieved_texts[:k]:
        text_tokens = set(re.findall(r"\w+", text.lower()))
        if key_tokens & text_tokens:
            hits += 1
    return round(hits / min(k, len(retrieved_texts)), 4)


# ── MRR ───────────────────────────────────────────────────────────────────────

def compute_mrr(question: str, retrieved_texts: List[str]) -> float:
    """
    Mean Reciprocal Rank.
    Ranks the first retrieved source that contains any question keyword.
    """
    if not retrieved_texts:
        return 0.0
    key_tokens = set(re.findall(r"\w+", question.lower())) - {
        "the", "a", "an", "is", "are", "what", "how", "why", "who",
    }
    if not key_tokens:
        return 0.0
    for rank, text in enumerate(retrieved_texts, start=1):
        text_tokens = set(re.findall(r"\w+", text.lower()))
        if key_tokens & text_tokens:
            return round(1 / rank, 4)
    return 0.0


# ── Hallucination proxy ───────────────────────────────────────────────────────

def grounding_score(answer: str, context_chunks: List[str]) -> float:
    """
    Estimates what fraction of answer sentences are 'grounded' in the
    retrieved context (contain overlapping n-grams).
    Returns a score in [0, 1]. Higher = less likely hallucination.
    """
    if not context_chunks or not answer:
        return 0.0

    context_tokens = set(
        tok
        for chunk in context_chunks
        for tok in re.findall(r"\w+", chunk.lower())
    )

    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    grounded = 0
    for sent in sentences:
        sent_tokens = set(re.findall(r"\w+", sent.lower()))
        overlap = len(sent_tokens & context_tokens) / max(len(sent_tokens), 1)
        if overlap >= 0.4:  # at least 40 % of sentence words appear in context
            grounded += 1

    return round(grounded / len(sentences), 4)