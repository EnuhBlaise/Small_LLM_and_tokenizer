"""Lab 2.2 — character/word tokenization and token frequencies.

Run: python examples/02_tokenize_basic.py
Needs the ``data`` extra (pandas) to load the sample corpus.
"""

from small_llm.data import load_africa_galore
from small_llm.tokenization import (
    character_tokenize,
    get_token_counts,
    preprocess_text,
    space_tokenize,
)

dataset = load_africa_galore()
print(f"Loaded {len(dataset)} paragraphs.\n")

sample = dataset[0]
print("char tokens :", character_tokenize(sample)[:10])
print("space tokens:", space_tokenize(sample)[:10])

counts = get_token_counts(preprocess_text(dataset))
print("\nTop 10 words:", counts.most_common(10))
