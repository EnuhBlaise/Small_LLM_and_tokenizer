"""Lab 2.6 (part 2) — train the small transformer LM end to end.

Run: python examples/06_train_slm.py
Needs the ``train`` and ``data`` extras (keras, jax, tqdm, pandas).
"""

import os

# Choose a Keras backend before importing keras (jax is light on CPU).
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402

from small_llm.data import load_africa_galore  # noqa: E402
from small_llm.model import TextGenerator, create_model, prepare_sequences  # noqa: E402
from small_llm.tokenization import BPEWordTokenizer  # noqa: E402

dataset = load_africa_galore()

tokenizer = BPEWordTokenizer(dataset, num_merges=3000)
print(f"Vocabulary size: {tokenizer.vocabulary_size:,}")

encoded = [tokenizer.encode(paragraph) for paragraph in dataset]
inputs, targets, max_len = prepare_sequences(
    encoded, tokenizer.pad_token_id, max_length=300
)

keras.utils.set_random_seed(3112)
model = create_model(
    max_length=max_len,
    vocabulary_size=tokenizer.vocabulary_size,
    learning_rate=8e-5,
)

text_gen = TextGenerator(
    max_tokens=11,
    start_tokens=tokenizer.encode("Jide"),
    tokenizer=tokenizer,
    print_every=10,
)

model.fit(
    inputs,
    targets,
    epochs=100,
    batch_size=2,
    verbose=2,
    callbacks=[text_gen],
)
