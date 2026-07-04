"""Lab 2.6 (part 1) — train a BPEWordTokenizer and save it for reuse.

Run: python examples/04_train_tokenizer.py
Needs the ``data`` extra (pandas). ``tqdm`` (``train`` extra) shows progress.
"""

from small_llm.data import load_africa_galore
from small_llm.tokenization import BPEWordTokenizer

dataset = load_africa_galore()
tokenizer = BPEWordTokenizer(dataset, num_merges=3000)
print(f"\nVocabulary size: {tokenizer.vocabulary_size:,}")

demo = "A Zimbabwian dish \U0001f60b."
ids = tokenizer.encode(demo)
print(f"encode({demo!r}) -> {ids}")
print(f"decode -> {tokenizer.decode(ids)!r}")

tokenizer.save("africa_galore_bpe.json")
print("\nSaved tokenizer to africa_galore_bpe.json")
print("Reload later with: BPEWordTokenizer.load('africa_galore_bpe.json')")
