import numpy as np

from small_llm.embeddings import cos_sim, get_embedding, most_similar, similarity


def _toy():
    labels = ["king", "queen", "banana"]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return embeddings, labels


def test_cos_sim_identical_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cos_sim(v, v) == 1.0


def test_get_embedding_lookup_and_error():
    embeddings, labels = _toy()
    assert np.allclose(get_embedding("king", embeddings, labels), embeddings[0])
    try:
        get_embedding("missing", embeddings, labels)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_similarity_and_most_similar():
    embeddings, labels = _toy()
    assert similarity("king", "queen", embeddings, labels) > similarity(
        "king", "banana", embeddings, labels
    )
    neighbours = most_similar("king", embeddings, labels, top_k=1)
    assert neighbours[0][0] == "queen"
