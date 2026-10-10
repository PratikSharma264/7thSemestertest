"""Utility functions for evaluating retrieval quality in the RAG project.

This module provides a lightweight evaluation toolkit for ranking and retrieval
systems. It is intentionally dependency-free so it can be used in tests and in
small offline evaluation workflows without requiring the full MedCPT model stack.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence


def normalize_text(text: str) -> str:
    """Lowercase text and normalize whitespace and punctuation."""
    if text is None:
        return ""

    normalized = str(text).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize(text: str) -> List[str]:
    """Split a string into lowercase tokens."""
    return normalize_text(text).split()


def overlap_score(query: str, document: str) -> float:
    """Return a Jaccard-style overlap score for the query and document."""
    query_tokens = set(tokenize(query))
    document_tokens = set(tokenize(document))

    if not query_tokens and not document_tokens:
        return 0.0

    if not query_tokens or not document_tokens:
        return 0.0

    intersection = query_tokens & document_tokens
    union = query_tokens | document_tokens
    return len(intersection) / len(union) if union else 0.0


def precision_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Compute precision@k for binary relevance labels."""
    if k <= 0:
        return 0.0

    relevant_set = {normalize_text(item) for item in relevant}
    top_k = [normalize_text(item) for item in retrieved[:k]]

    seen = set()
    hits = 0
    for item in top_k:
        if item in relevant_set and item not in seen:
            hits += 1
            seen.add(item)

    return hits / k if k > 0 else 0.0


def recall_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Compute recall@k for binary relevance labels."""
    if k <= 0:
        return 0.0

    relevant_set = {normalize_text(item) for item in relevant}
    if not relevant_set:
        return 0.0

    top_k = [normalize_text(item) for item in retrieved[:k]]
    seen = set()
    hits = 0
    for item in top_k:
        if item in relevant_set and item not in seen:
            hits += 1
            seen.add(item)

    return hits / len(relevant_set)


def f1_score(precision: float, recall: float) -> float:
    """Compute the F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_reciprocal_rank(relevant: Iterable[str], retrieved: Sequence[str]) -> float:
    """Return the reciprocal rank of the first relevant hit, or 0.0 if none."""
    relevant_set = {normalize_text(item) for item in relevant}
    if not relevant_set:
        return 0.0

    for rank, item in enumerate(retrieved, start=1):
        normalized = normalize_text(item)
        if normalized in relevant_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Compute a binary-relevance NDCG@k score.

    This version uses a standard discounting scheme while treating each unique
    relevant hit as a binary gain of 1.0.
    """
    if k <= 0:
        return 0.0

    relevant_set = {normalize_text(item) for item in relevant}
    if not relevant_set:
        return 0.0

    retrieved_top = [normalize_text(item) for item in retrieved[:k]]

    seen = set()
    dcg = 0.0
    for rank, item in enumerate(retrieved_top, start=1):
        if item in relevant_set and item not in seen:
            dcg += 1.0 / math.log2(rank + 1)
            seen.add(item)

    ideal_hits = min(len(relevant_set), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def evaluate_retrieval(
    query: str,
    retrieved_docs: Sequence[str],
    relevant_docs: Sequence[str],
    top_k: int,
) -> dict:
    """Assess a retrieved result list against a list of relevant documents."""
    relevance = {normalize_text(item) for item in relevant_docs}
    retrieved_top = [normalize_text(item) for item in retrieved_docs[:top_k]]

    precision = precision_at_k(relevance, retrieved_top, top_k)
    recall = recall_at_k(relevance, retrieved_top, top_k)
    score = f1_score(precision, recall)
    rr = mean_reciprocal_rank(relevance, retrieved_top)
    ndcg = ndcg_at_k(relevance, retrieved_top, top_k)

    return {
        "query": normalize_text(query),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_score": score,
        "mrr": rr,
        "ndcg_at_k": ndcg,
        "retrieved_count": len(retrieved_top),
        "relevant_count": len(relevance),
    }
