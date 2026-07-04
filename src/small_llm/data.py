"""Dataset loading helpers.

The labs use the "Africa Galore" dataset of short descriptive paragraphs,
hosted publicly by Google DeepMind. ``load_africa_galore`` fetches it; pass a
local path to use your own corpus instead.
"""

from __future__ import annotations

from pathlib import Path

AFRICA_GALORE_URL = (
    "https://storage.googleapis.com/dm-educational/assets/"
    "ai_foundations/africa_galore.json"
)

__all__ = ["AFRICA_GALORE_URL", "load_africa_galore", "load_text_lines"]


def load_africa_galore(
    source: str | Path = AFRICA_GALORE_URL, column: str = "description"
) -> list[str]:
    """Load the Africa Galore paragraphs as a list of strings.

    Args:
      source: URL or local path to the JSON file.
      column: Column holding the text paragraphs.

    Returns:
      A list of paragraph strings.
    """
    import pandas as pd

    frame = pd.read_json(source)
    return list(frame[column].values)


def load_text_lines(path: str | Path) -> list[str]:
    """Load a plain-text corpus, one paragraph per non-empty line."""
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]
