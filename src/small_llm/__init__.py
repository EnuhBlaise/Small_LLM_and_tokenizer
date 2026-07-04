"""small_llm: a compact toolkit for text preprocessing, BPE tokenization,
embeddings, and training a tiny language model.

Grown out of the Google DeepMind "AI Foundations" Course 2 labs and refactored
into a reusable, dependency-light package for teaching and research.

Submodules
----------
- :mod:`small_llm.preprocess`     — HTML/Unicode text cleaning
- :mod:`small_llm.tokenization`   — character/word tokenizers and BPE
- :mod:`small_llm.embeddings`     — cosine similarity, nearest neighbours, plots
- :mod:`small_llm.data`           — dataset loaders
- :mod:`small_llm.model`          — a small transformer LM (needs the ``train`` extra)
"""

from small_llm import data, embeddings, preprocess, tokenization
from small_llm.preprocess import clean_html, clean_html_tags, clean_unicode
from small_llm.tokenization import BPEWordTokenizer

__version__ = "0.1.0"

__all__ = [
    "preprocess",
    "tokenization",
    "embeddings",
    "data",
    "clean_html",
    "clean_html_tags",
    "clean_unicode",
    "BPEWordTokenizer",
    "__version__",
]
