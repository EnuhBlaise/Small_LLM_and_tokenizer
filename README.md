# small-llm

**A compact, self-contained toolkit for text preprocessing, Byte Pair Encoding
tokenization, embeddings, and training a tiny transformer language model.**

This project grew out of the [Google DeepMind *AI Foundations* Course 2 labs](https://github.com/google-deepmind/ai-foundations)
and refactors the notebook code into a clean, tested, reusable Python package —
so the same building blocks can go from teaching material to a starting point
for real research. The core (preprocessing + tokenization + embedding math) has
no heavy dependencies; only the language-model trainer needs Keras.

📄 **Project page:** `docs/index.html` (publishable via GitHub Pages)

---

## Why this exists

The original material is six Jupyter notebooks with inline solutions and a
dependency on a course-specific package. That's great for following along, but
awkward to build on. This repo turns it into:

- **Importable modules** with docstrings and type hints, not copy-pasted cells.
- **No course dependency** — embedding loading, plotting, and the transformer
  model are reimplemented with `numpy` / `scikit-learn` / `keras`.
- **A tested surface** you can trust (`pytest`, 13 tests) and extend.
- **Persistence** — trained tokenizers save to / load from JSON.

## Install

```bash
git clone https://github.com/EnuhBlaise/Small_LLM_and_tokenizer.git
cd Small_LLM_and_tokenizer

pip install -e .            # core: preprocessing + tokenization (numpy only)
pip install -e ".[data]"   # + pandas, to load the sample dataset
pip install -e ".[viz]"    # + matplotlib/scikit-learn, for embedding plots
pip install -e ".[train]"  # + keras/jax/tqdm, to train the language model
pip install -e ".[all,dev]"  # everything, plus pytest/ruff
```

## Package layout

```
src/small_llm/
├── preprocess.py          # clean_html, clean_html_tags, clean_unicode
├── tokenization/
│   ├── basic.py           # character/space/word tokenizers, token counts
│   └── bpe.py             # BPE (functional helpers + BPEWordTokenizer class)
├── embeddings.py          # cos_sim, get_embedding, most_similar, plots
├── data.py                # dataset loaders (Africa Galore, plain text)
└── model.py               # small transformer LM + TextGenerator callback
examples/                  # one runnable script per lab (01–06)
tests/                     # pytest suite
docs/index.html            # project landing page (GitHub Pages)
```

## Quick start

**Clean text** (Lab 2.1):

```python
from small_llm.preprocess import clean_html, clean_unicode

clean_html("<p>2018 &amp; pop &gt; 200000.</p>")   # "2018 & pop > 200000."
clean_unicode("Rice costs ₦150000. Ah! 😱")          # "Rice costs 150000. Ah! "
```

**Train and reuse a BPE tokenizer** (Labs 2.4 & 2.6):

```python
from small_llm.tokenization import BPEWordTokenizer

tok = BPEWordTokenizer(["the quick brown fox", "the lazy dog"], num_merges=50)
ids = tok.encode("the fox")     # [12, 7, 33, ...]
tok.decode(ids)                 # "the fox"

tok.save("my_bpe.json")
tok = BPEWordTokenizer.load("my_bpe.json")   # reuse across experiments
```

**Explore embeddings** (Lab 2.5):

```python
from small_llm.embeddings import load_embeddings, similarity, most_similar

emb, labels = load_embeddings("gemma_embeddings.npz")
similarity("king", "queen", emb, labels)        # 0.63
most_similar("king", emb, labels, top_k=5)      # [('queen', 0.63), ...]
```

**Train the tiny language model** (Lab 2.6):

```python
from small_llm.model import create_model, prepare_sequences, TextGenerator

encoded = [tok.encode(p) for p in paragraphs]
inputs, targets, max_len = prepare_sequences(encoded, tok.pad_token_id)
model = create_model(max_len, tok.vocabulary_size)
model.fit(inputs, targets, epochs=100, batch_size=2)
```

Each notebook maps to a runnable script in `examples/` — see
`examples/06_train_slm.py` for the full pipeline.

## Using it for research

The design keeps the pieces decoupled so you can swap parts in and out:

- **Bring your own corpus** — `data.load_text_lines("corpus.txt")` or pass any
  `list[str]` to the tokenizer.
- **Tune the model** — `create_model` exposes `embed_dim`, `num_heads`,
  `ff_dim`, `num_blocks`, `dropout`, and `learning_rate`.
- **Any Keras backend** — set `KERAS_BACKEND` to `jax`, `tensorflow`, or `torch`.
- **Inspect intermediate state** — the functional BPE helpers
  (`bpe_initialize`, `get_pair_frequencies`, `merge_pair_in_word`, `learn_bpe`)
  return the corpus and vocabulary at each step for analysis.

## Original notebooks

The six source notebooks are preserved under `Notebooks/` for reference. The
package reproduces their functionality in importable form.

## Development

```bash
pip install -e ".[all,dev]"
pytest        # run the test suite
ruff check .  # lint
```

## Provenance & license

Adapted from the Google DeepMind *AI Foundations* course (Apache-2.0). The
"Africa Galore" dataset and Gemma embeddings referenced in the examples are
hosted by Google DeepMind. This repository is released under the Apache-2.0
license.
