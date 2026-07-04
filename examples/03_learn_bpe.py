"""Lab 2.4 — learn BPE merges step by step on a toy corpus.

Run: python examples/03_learn_bpe.py
"""

from small_llm.tokenization import learn_bpe

sample = [
    "desert", "deserted", "deserts", "desert", "tested", "test",
    "deserted", "desert", "desertion", "desertion", "function",
]

merges, vocabulary, tokenized = learn_bpe(sample, num_merges=10)

print("Learned merges (in order):")
for i, (a, b) in enumerate(merges, start=1):
    print(f" {i:2d}. {a!r} + {b!r} -> {a + b!r}")

print("\nFinal tokenized corpus:")
for original, tokens in zip(sample, tokenized):
    print(f"  {original:10s} -> {' '.join(tokens)}")
