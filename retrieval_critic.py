from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import argparse
from pathlib import Path


@dataclass(frozen=True)
class RetrievalCritique:
    status: str
    query: str
    top_match: str
    similarity: float
    warning: str


def _similarity(left: str, right: str) -> float:
    left_text = left.casefold()
    right_text = right.casefold()
    left_tokens = set(left_text.split())
    right_tokens = set(right_text.split())
    token_overlap = len(left_tokens & right_tokens) / max(len(left_tokens), 1)
    return max(token_overlap, SequenceMatcher(None, left_text, right_text).ratio())


def critique_retrieval(query: str, candidates: list[str], threshold: float = 0.62) -> RetrievalCritique:
    if not candidates:
        return RetrievalCritique("review", query, "", 0.0, "No retrieval candidates were supplied.")

    scored = sorted(((candidate, _similarity(query, candidate)) for candidate in candidates), key=lambda item: item[1], reverse=True)
    top_match, score = scored[0]
    warning = (
        "Similarity is only lexical here; treat this as retrieval evidence, not truth."
        if score >= threshold
        else "Top match is weak. Ask for more context or use semantic embeddings before answering."
    )
    return RetrievalCritique("pass" if score >= threshold else "review", query, top_match, round(score, 4), warning)


def load_sentences(path: Path) -> list[str]:
    text = path.read_text()
    return [item.strip() for item in text.split(".") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic retrieval critic over the demo corpus.")
    parser.add_argument("query", help="Query to compare against data.txt")
    parser.add_argument("--data", type=Path, default=Path("data.txt"))
    args = parser.parse_args()
    critique = critique_retrieval(args.query, load_sentences(args.data))
    print(f"{critique.status}: {critique.top_match} ({critique.similarity:.2f})")
    print(critique.warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
