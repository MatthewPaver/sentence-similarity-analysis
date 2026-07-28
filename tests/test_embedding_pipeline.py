import numpy as np

from embedding_pipeline import rank_sentences


class FakeEncoder:
    vectors = {
        "polar bear fur": [1.0, 0.0, 0.0],
        "Polar bear fur is transparent.": [0.98, 0.2, 0.0],
        "Polar bears live in the Arctic.": [0.7, 0.7, 0.0],
        "Mars is red.": [0.0, 0.0, 1.0],
    }

    def encode(self, sentences, *, normalize_embeddings):
        values = np.asarray([self.vectors[sentence] for sentence in sentences], dtype=float)
        if normalize_embeddings:
            values = values / np.linalg.norm(values, axis=1, keepdims=True)
        return values


def test_embedding_pipeline_ranks_the_semantic_match_first():
    ranked = rank_sentences(
        "polar bear fur",
        [
            "Mars is red.",
            "Polar bears live in the Arctic.",
            "Polar bear fur is transparent.",
        ],
        model=FakeEncoder(),
    )

    assert ranked[0].sentence == "Polar bear fur is transparent."
    assert ranked[0].score > ranked[1].score > ranked[2].score


def test_embedding_pipeline_handles_an_empty_corpus():
    assert rank_sentences("anything", [], model=FakeEncoder()) == []
