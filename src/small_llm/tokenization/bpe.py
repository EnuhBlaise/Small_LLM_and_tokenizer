"""Byte Pair Encoding (BPE) tokenization.

This module contains two layers:

* **Functional helpers** (``bpe_initialize``, ``get_pair_frequencies``,
  ``merge_pair_in_word``, ``learn_bpe``) that spell out the BPE algorithm step
  by step (Lab 2.4). They are handy for teaching and for inspecting the
  intermediate state of training.

* **:class:`BPEWordTokenizer`** (Lab 2.6), a reusable tokenizer that learns
  merges from a corpus, encodes/decodes text to integer IDs, and can be saved
  to and loaded from disk so a trained tokenizer can be reused across
  experiments.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

__all__ = [
    "EOW_SYMBOL",
    "bpe_initialize",
    "get_pair_frequencies",
    "merge_pair_in_word",
    "learn_bpe",
    "BPEWordTokenizer",
]

# Marks the end of a word so the tokenizer can restore word boundaries.
EOW_SYMBOL = "</w>"

Pair = tuple[str, str]


# --------------------------------------------------------------------------- #
# Functional BPE (Lab 2.4)                                                     #
# --------------------------------------------------------------------------- #
def bpe_initialize(dataset: list[str]) -> tuple[list[list[str]], set[str]]:
    """Initialize the BPE corpus and vocabulary.

    Args:
      dataset: The corpus as a list of raw paragraphs.

    Returns:
      corpus: Every space-separated word as a list of characters plus the
        end-of-word symbol.
      vocabulary: The set of unique starting tokens (characters + ``</w>``).
    """
    corpus: list[list[str]] = []
    vocabulary: set[str] = {EOW_SYMBOL}
    for paragraph in dataset:
        for word in paragraph.split(" "):
            chars = list(word) + [EOW_SYMBOL]
            corpus.append(chars)
            vocabulary.update(chars)
    return corpus, vocabulary


def get_pair_frequencies(corpus: list[list[str]]) -> Counter[Pair]:
    """Count adjacent token pairs across the corpus.

    Args:
      corpus: A list of tokenized words (each a list of subword tokens).

    Returns:
      A Counter mapping each adjacent pair to its frequency.
    """
    pairs: Counter[Pair] = Counter()
    for word in corpus:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
    return pairs


def merge_pair_in_word(word: list[str], pair_to_merge: Pair) -> list[str]:
    """Merge every adjacent occurrence of ``pair_to_merge`` in one word.

    Args:
      word: Subword tokens for a single word.
      pair_to_merge: The pair of tokens to merge into one.

    Returns:
      A new token list with the pair merged wherever it occurred.
    """
    merged_symbol = pair_to_merge[0] + pair_to_merge[1]
    new_word: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair_to_merge:
            new_word.append(merged_symbol)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word


def learn_bpe(
    dataset: list[str], num_merges: int
) -> tuple[list[Pair], set[str], list[list[str]]]:
    """Learn BPE merge operations from a dataset.

    Iteratively merges the most frequent adjacent pair until ``num_merges`` is
    reached or no more pairs remain.

    Args:
      dataset: A list of paragraphs.
      num_merges: The number of merge operations to perform.

    Returns:
      merges: The merged pairs, in the order they were performed.
      vocabulary: The final set of subword tokens.
      corpus: The final tokenized corpus.
    """
    corpus, vocabulary = bpe_initialize(dataset)
    merges: list[Pair] = []
    for _ in range(num_merges):
        pair_freqs = get_pair_frequencies(corpus)
        if not pair_freqs:
            break
        most_freq_pair, freq = pair_freqs.most_common(1)[0]
        if freq < 1:
            break
        merges.append(most_freq_pair)
        vocabulary.add(most_freq_pair[0] + most_freq_pair[1])
        corpus = [merge_pair_in_word(word, most_freq_pair) for word in corpus]
    return merges, vocabulary, corpus


# --------------------------------------------------------------------------- #
# Reusable tokenizer (Lab 2.6)                                                 #
# --------------------------------------------------------------------------- #
class BPEWordTokenizer:
    """A Byte Pair Encoding subword tokenizer.

    Learns merge rules from a corpus (or is initialized with a pre-built
    vocabulary), then encodes/decodes text to integer token IDs. Includes
    ``<PAD>`` and ``<UNK>`` special tokens and can be persisted with
    :meth:`save` / :meth:`load`.

    Attributes:
      vocabulary: List of subword tokens including special tokens.
      vocabulary_size: Total number of tokens in the vocabulary.
      token_to_index: Mapping from token to index.
      index_to_token: Mapping from index to token.
      pad_token_id: Index of the padding token.
      unknown_token_id: Index of the unknown token.
      merges: The learned merge operations, in order.
      tokenized_corpus: Cached tokenized corpus from training (if any).
    """

    UNKNOWN_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    END_WORD = EOW_SYMBOL

    def __init__(
        self,
        texts: list[str] | str | None = None,
        vocabulary: list[str] | None = None,
        num_merges: int = 100,
        merges: list[Pair] | None = None,
        *,
        _skip_learning: bool = False,
    ):
        """Build a tokenizer by learning merges or from an existing vocabulary.

        Args:
          texts: A string or list of strings to learn BPE from. Optional when
            ``vocabulary`` and ``merges`` are supplied directly.
          vocabulary: Optional pre-built vocabulary; skips merge learning.
          num_merges: Number of merge rounds when learning from ``texts``.
          merges: Optional pre-learned merges (used with ``vocabulary``).
          _skip_learning: Internal flag used by :meth:`load`.
        """
        if isinstance(texts, str):
            texts = [texts]

        self.merges: list[Pair] = merges or []
        self.tokenized_corpus: list[list[list[str]]] = []

        if _skip_learning:
            # ``vocabulary`` and ``merges`` provided by the caller (e.g. load).
            self.vocabulary = list(vocabulary) if vocabulary else []
        elif vocabulary is not None:
            self.vocabulary = list(vocabulary)
            self.merges = merges or []
        else:
            if texts is None:
                raise ValueError("Provide `texts` or a `vocabulary`.")
            self.merges, tokenized, vocabulary_set = self._learn_bpe(
                texts, num_merges
            )
            self.tokenized_corpus = tokenized
            # Always keep basic alphanumerics so unseen letters are encodable.
            required = set(
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
            )
            vocabulary_set.update(required)
            self.vocabulary = (
                [self.PAD_TOKEN] + sorted(vocabulary_set) + [self.UNKNOWN_TOKEN]
            )

        self._build_index()

    # -- indexing ----------------------------------------------------------- #
    def _build_index(self) -> None:
        self.vocabulary_size = len(self.vocabulary)
        self.token_to_index = {t: i for i, t in enumerate(self.vocabulary)}
        self.index_to_token = {i: t for i, t in enumerate(self.vocabulary)}
        self.pad_token_id = self.token_to_index.get(self.PAD_TOKEN, 0)
        self.unknown_token_id = self.token_to_index.get(
            self.UNKNOWN_TOKEN, self.vocabulary_size - 1
        )

    # -- encoding / decoding ------------------------------------------------ #
    def _split_text(self, text: str) -> list[str]:
        """Split text into subword tokens using the learned merges."""
        tokens: list[str] = []
        for word in text.strip().split():
            chars = list(word) + [self.END_WORD]
            for pair in self.merges:
                chars = merge_pair_in_word(chars, pair)
            tokens.extend(chars)
        return tokens

    def join_text(self, tokens: list[str]) -> str:
        """Join subword tokens back into a string, restoring word boundaries."""
        words: list[str] = []
        current: list[str] = []
        for token in tokens:
            if token.endswith(self.END_WORD):
                current.append(token[: -len(self.END_WORD)])
                words.append("".join(current))
                current = []
            else:
                current.append(token)
        if current:
            words.append("".join(current))
        return " ".join(words).strip()

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs."""
        return [
            self.token_to_index.get(tok, self.unknown_token_id)
            for tok in self._split_text(text)
        ]

    def decode(self, token_ids: int | list[int]) -> str:
        """Decode a token ID or list of IDs back into text."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        tokens = [
            self.index_to_token.get(i, self.UNKNOWN_TOKEN + self.END_WORD)
            for i in token_ids
        ]
        return self.join_text(tokens)

    # -- training ----------------------------------------------------------- #
    def _learn_bpe(
        self, corpus: list[str], num_merges: int
    ) -> tuple[list[Pair], list[list[list[str]]], set[str]]:
        """Learn BPE merges from a corpus (keeps per-paragraph structure)."""
        try:
            from tqdm import tqdm  # optional progress bar
        except ImportError:  # pragma: no cover
            def tqdm(x, **_):  # type: ignore
                return x

        tokenized_corpus: list[list[list[str]]] = []
        vocabulary: set[str] = {self.END_WORD}
        for paragraph in corpus:
            sentence: list[list[str]] = []
            for word in paragraph.strip().split():
                sentence.append(list(word) + [self.END_WORD])
                vocabulary.update(list(word))
            tokenized_corpus.append(sentence)

        merges: list[Pair] = []
        pbar = tqdm(range(num_merges), unit="merges")
        for _ in pbar:
            flat = [word for para in tokenized_corpus for word in para]
            pair_freqs = get_pair_frequencies(flat)
            if not pair_freqs:
                break
            most_freq_pair, freq = pair_freqs.most_common(1)[0]
            if freq < 1:
                break
            merges.append(most_freq_pair)
            tokenized_corpus = [
                [merge_pair_in_word(word, most_freq_pair) for word in para]
                for para in tokenized_corpus
            ]
            vocabulary.add(most_freq_pair[0] + most_freq_pair[1])
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(vocabulary_size=f"{len(vocabulary):,}")
        return merges, tokenized_corpus, vocabulary

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        """Save the tokenizer (vocabulary + merges) to a JSON file."""
        path = Path(path)
        payload = {
            "vocabulary": self.vocabulary,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPEWordTokenizer":
        """Load a tokenizer previously written with :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in data.get("merges", [])]
        tok = cls(
            vocabulary=data["vocabulary"],
            merges=merges,
            _skip_learning=True,
        )
        return tok

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BPEWordTokenizer(vocabulary_size={self.vocabulary_size}, "
            f"num_merges={len(self.merges)})"
        )
