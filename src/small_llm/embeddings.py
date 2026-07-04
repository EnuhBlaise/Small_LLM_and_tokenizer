"""Embedding utilities (Lab 2.5).

Cosine similarity, token lookup, nearest-neighbour search, and lightweight
visualization helpers. Nothing here depends on the course package: embeddings
are loaded from a standard ``.npz`` file and plots use matplotlib/scikit-learn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "cos_sim",
    "get_embedding",
    "similarity",
    "most_similar",
    "load_embeddings",
    "plot_embedding_dimensions",
    "plot_embeddings_tsne",
]


def cos_sim(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors.

    Args:
      u: A vector of shape ``(k,)``.
      v: A vector of shape ``(k,)``.

    Returns:
      The cosine similarity, in ``[-1, 1]``.
    """
    dot = np.dot(u, v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def get_embedding(
    token: str, embeddings: np.ndarray, labels: list[str]
) -> np.ndarray:
    """Return the embedding row for ``token``.

    Args:
      token: The token to look up.
      embeddings: The embedding matrix, one row per token in ``labels``.
      labels: Token order matching the rows of ``embeddings``.

    Returns:
      The embedding vector for ``token``.

    Raises:
      ValueError: If ``token`` is not in ``labels``.
    """
    if token not in labels:
        raise ValueError(f"No embedding for {token!r} exists.")
    return embeddings[labels.index(token)]


def similarity(
    token1: str,
    token2: str,
    embeddings: np.ndarray,
    labels: list[str],
) -> float:
    """Cosine similarity between the embeddings of two tokens."""
    e1 = get_embedding(token1, embeddings, labels)
    e2 = get_embedding(token2, embeddings, labels)
    return cos_sim(e1, e2)


def most_similar(
    token: str,
    embeddings: np.ndarray,
    labels: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return the ``top_k`` tokens most similar to ``token``.

    A small research convenience on top of the lab material: ranks the whole
    vocabulary by cosine similarity and returns the closest tokens (excluding
    the query itself).

    Args:
      token: The query token.
      embeddings: The embedding matrix.
      labels: Token order matching ``embeddings``.
      top_k: How many neighbours to return.

    Returns:
      A list of ``(token, similarity)`` pairs sorted from most to least similar.
    """
    query = get_embedding(token, embeddings, labels)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query)
    norms[norms == 0] = 1e-12
    scores = embeddings @ query / norms
    order = np.argsort(-scores)
    results: list[tuple[str, float]] = []
    for idx in order:
        if labels[idx] == token:
            continue
        results.append((labels[idx], float(scores[idx])))
        if len(results) >= top_k:
            break
    return results


def load_embeddings(
    path: str | Path,
    embeddings_key: str = "embeddings",
    labels_key: str = "labels",
) -> tuple[np.ndarray, list[str]]:
    """Load an embedding matrix and labels from a ``.npz`` file.

    Args:
      path: Path (or URL supported by numpy) to the ``.npz`` archive.
      embeddings_key: Array key holding the embedding matrix.
      labels_key: Array key holding the token labels.

    Returns:
      ``(embeddings, labels)`` where ``labels`` is a list of strings.
    """
    with np.load(path, allow_pickle=True) as data:
        embeddings = data[embeddings_key]
        labels = [str(x) for x in data[labels_key]]
    return embeddings, labels


def plot_embedding_dimensions(
    embeddings: np.ndarray,
    labels: list[str],
    dim_x: int = 0,
    dim_y: int = 1,
    ax=None,
):
    """Scatter-plot two chosen embedding dimensions, annotated with labels."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    xs, ys = embeddings[:, dim_x], embeddings[:, dim_y]
    ax.scatter(xs, ys)
    for label, x, y in zip(labels, xs, ys):
        ax.annotate(label, (x, y), fontsize=8)
    ax.set_xlabel(f"dimension {dim_x}")
    ax.set_ylabel(f"dimension {dim_y}")
    ax.set_title("Embedding dimensions")
    return ax


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: list[str],
    categories: list[int] | None = None,
    *,
    random_state: int = 42,
    perplexity: float | None = None,
    ax=None,
):
    """Project embeddings to 2-D with t-SNE and scatter-plot them.

    Args:
      embeddings: The embedding matrix.
      labels: Token labels, one per row.
      categories: Optional integer category per token, used for colour.
      random_state: Seed for reproducibility.
      perplexity: t-SNE perplexity; defaults to a value safe for small sets.
      ax: Optional matplotlib axis to draw on.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n = len(embeddings)
    if perplexity is None:
        perplexity = max(2.0, min(30.0, (n - 1) / 3))
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        init="pca",
    )
    projected = tsne.fit_transform(np.asarray(embeddings))

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(projected[:, 0], projected[:, 1], c=categories, cmap="tab10")
    for label, (x, y) in zip(labels, projected):
        ax.annotate(label, (x, y), fontsize=8)
    ax.set_title("t-SNE projection of embeddings")
    return ax
