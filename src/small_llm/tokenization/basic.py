"""Simple tokenizers and token-frequency helpers (Lab 2.2).

These illustrate the trade-offs between character-, space-, and word-level
tokenization before moving on to subword (BPE) tokenization.
"""

from __future__ import annotations

import re
from collections import Counter

__all__ = [
    "character_tokenize",
    "space_tokenize",
    "word_tokenize",
    "preprocess_text",
    "get_token_counts",
]

_WORD_RE = re.compile(r"\b\w+\b")


def character_tokenize(text: str) -> list[str]:
    """Split text into individual characters."""
    return list(text)


def space_tokenize(text: str) -> list[str]:
    """Split text on single spaces."""
    return text.split(" ")


def word_tokenize(text: str) -> list[str]:
    """Split text into word tokens, dropping punctuation.

    Uses the regex ``\\b\\w+\\b`` so word boundaries are respected more
    precisely than a plain ``str.split``.
    """
    return _WORD_RE.findall(text)


def preprocess_text(paragraphs: list[str]) -> list[str]:
    """Lowercase and word-tokenize a list of paragraphs into a flat token list.

    Args:
      paragraphs: A list of paragraph strings.

    Returns:
      A flat list of lowercase word tokens across all paragraphs.
    """
    tokens: list[str] = []
    for paragraph in paragraphs:
        tokens.extend(word_tokenize(paragraph.lower()))
    return tokens


def get_token_counts(tokens: list[str]) -> Counter[str]:
    """Count the frequency of each token.

    Args:
      tokens: A list of string tokens.

    Returns:
      A :class:`collections.Counter` mapping each unique token to its count.
    """
    return Counter(tokens)
