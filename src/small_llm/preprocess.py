"""Text cleaning utilities.

These are the preprocessing steps from Lab 2.1: stripping HTML markup and
filtering out non-text Unicode (emojis, symbols) while keeping letters,
numbers, and punctuation.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["clean_html_tags", "clean_html", "clean_unicode"]

_HTML_TAG_RE = re.compile(r"<.*?>")

# HTML entities handled by ``clean_html``. Extend this mapping to cover more.
_HTML_ENTITIES = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
}

# Unicode general-category prefixes to keep: L=letter, N=number, P=punctuation.
_DEFAULT_KEEP_CATEGORIES = frozenset({"L", "N", "P"})


def clean_html_tags(text: str) -> str:
    """Remove every HTML tag from a string.

    Applies the non-greedy pattern ``<.*?>`` so anything enclosed in angle
    brackets (``<h1>``, ``</p>``, ``<img .../>``) is deleted while surrounding
    text is left intact.

    Args:
      text: Raw text that may contain HTML tags.

    Returns:
      Plain text with all HTML tags removed.
    """
    return _HTML_TAG_RE.sub("", text)


def clean_html(text: str) -> str:
    """Strip HTML tags and decode a handful of common entities.

    This does not attempt full HTML parsing; for complex markup use
    :mod:`html` (``html.unescape``) or ``BeautifulSoup``.

    Args:
      text: Text that may contain HTML tags or entities.

    Returns:
      Cleaned text with tags removed and ``&nbsp; &amp; &lt; &gt;`` decoded.
    """
    text = clean_html_tags(text)
    for entity, replacement in _HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    return text


def clean_unicode(
    text: str, keep_categories: frozenset[str] = _DEFAULT_KEEP_CATEGORIES
) -> str:
    """Remove non-text Unicode characters (emojis, symbols) from a string.

    Whitespace is always preserved. Every other character is kept only if its
    Unicode general category starts with one of ``keep_categories``.

    Args:
      text: Original text which may contain special characters.
      keep_categories: Unicode category prefixes to keep. Defaults to letters,
        numbers, and punctuation (``{"L", "N", "P"}``).

    Returns:
      The input text with disallowed symbols removed.
    """
    keep = []
    for ch in text:
        if ch.isspace():
            keep.append(ch)
            continue
        category = unicodedata.category(ch)
        if any(category.startswith(prefix) for prefix in keep_categories):
            keep.append(ch)
    return "".join(keep)
