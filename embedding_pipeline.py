"""Testable sentence-embedding ranking used by the notebook demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class Encoder(Protocol):
    def encode(self, sentences: Sequence[str], *, normalize_embeddings: bool) -> object: ...


@dataclass(frozen=True)
class RankedSentence:
    sentence: str
    score: float


def rank_sentences(
    query: str,
    candidates: Sequence[str],
    *,
    model: Encoder | None = None,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> list[RankedSentence]:
    """Rank candidates by cosine similarity using normalized embeddings."""

    if not candidates:
        return []
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

    embeddings = np.asarray(
        model.encode([query, *candidates], normalize_embeddings=True),
        dtype=float,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(candidates) + 1:
        raise ValueError("encoder returned an unexpected embedding shape")
    scores = embeddings[1:] @ embeddings[0]
    ranked = [
        RankedSentence(sentence=sentence, score=round(float(score), 4))
        for sentence, score in zip(candidates, scores, strict=True)
    ]
    return sorted(ranked, key=lambda item: item.score, reverse=True)
