"""A small transformer language model (Lab 2.6), self-contained in Keras 3.

This reimplements the tiny GPT-style decoder used in the course so the package
has no dependency on the course's ``ai_foundations`` package. It is
backend-agnostic (JAX / TensorFlow / PyTorch via Keras 3); set
``KERAS_BACKEND`` before importing Keras to choose one.

Typical use::

    from small_llm.model import create_model, prepare_sequences, TextGenerator

    inputs, targets, max_len = prepare_sequences(encoded, tokenizer.pad_token_id)
    model = create_model(max_len, tokenizer.vocabulary_size)
    model.fit(inputs, targets, epochs=100, batch_size=2,
              callbacks=[TextGenerator(20, tokenizer.encode("Jide"), tokenizer)])
"""

from __future__ import annotations

import numpy as np

__all__ = ["prepare_sequences", "create_model", "TextGenerator"]


def prepare_sequences(
    encoded_sequences: list[list[int]],
    pad_token_id: int,
    max_length: int = 300,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad encoded sequences and build shifted (input, target) pairs.

    For next-token prediction the target is the input shifted left by one.

    Args:
      encoded_sequences: List of token-ID lists (one per paragraph).
      pad_token_id: ID used for padding.
      max_length: Sequences are padded/truncated to this length.

    Returns:
      ``(input_sequences, target_sequences, effective_max_length)`` where the
      effective length is ``max_length - 1`` after the shift.
    """
    import keras

    padded = keras.preprocessing.sequence.pad_sequences(
        encoded_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post",
        value=pad_token_id,
    )
    input_sequences = padded[:, :-1]
    target_sequences = padded[:, 1:]
    return input_sequences, target_sequences, input_sequences.shape[1]


def _build_layers():
    """Return Keras layer classes lazily (keras imported on demand)."""
    import keras
    from keras import layers

    class TokenAndPositionEmbedding(layers.Layer):
        """Sum of token embeddings and learned positional embeddings."""

        def __init__(self, max_length, vocabulary_size, embed_dim, **kwargs):
            super().__init__(**kwargs)
            self.token_emb = layers.Embedding(vocabulary_size, embed_dim)
            self.pos_emb = layers.Embedding(max_length, embed_dim)

        def call(self, x):
            positions = keras.ops.arange(0, keras.ops.shape(x)[-1])
            return self.token_emb(x) + self.pos_emb(positions)

    class TransformerBlock(layers.Layer):
        """Pre-norm transformer decoder block with causal self-attention."""

        def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.att = layers.MultiHeadAttention(num_heads, embed_dim)
            self.ffn = keras.Sequential(
                [
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ]
            )
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
            self.drop1 = layers.Dropout(dropout)
            self.drop2 = layers.Dropout(dropout)

        def call(self, x, training=False):
            attn = self.att(x, x, use_causal_mask=True, training=training)
            x = self.norm1(x + self.drop1(attn, training=training))
            ffn = self.ffn(x, training=training)
            return self.norm2(x + self.drop2(ffn, training=training))

    return TokenAndPositionEmbedding, TransformerBlock


def create_model(
    max_length: int,
    vocabulary_size: int,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_blocks: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 8e-5,
):
    """Build and compile the small transformer language model.

    Args:
      max_length: Length of the (already shifted) input sequences.
      vocabulary_size: Number of tokens in the tokenizer vocabulary.
      embed_dim: Embedding / model dimension.
      num_heads: Attention heads per block.
      ff_dim: Hidden size of the feed-forward sublayer.
      num_blocks: Number of transformer blocks.
      dropout: Dropout rate.
      learning_rate: Adam learning rate.

    Returns:
      A compiled ``keras.Model`` producing per-position vocabulary logits.
    """
    import keras
    from keras import layers

    TokenAndPositionEmbedding, TransformerBlock = _build_layers()

    inputs = layers.Input(shape=(max_length,), dtype="int32")
    x = TokenAndPositionEmbedding(max_length, vocabulary_size, embed_dim)(inputs)
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)(x)
    outputs = layers.Dense(vocabulary_size)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


class TextGenerator:
    """Keras callback that samples text from the model during training.

    Instantiated as a normal object but subclasses ``keras.callbacks.Callback``
    at construction time (Keras is imported lazily so importing this module is
    cheap and backend-free).
    """

    def __new__(cls, *args, **kwargs):
        import keras

        # Build a concrete subclass bound to keras.callbacks.Callback once.
        if not getattr(cls, "_bound", False):
            base = keras.callbacks.Callback

            class _TextGenerator(base):
                def __init__(
                    self,
                    max_tokens,
                    start_tokens,
                    tokenizer,
                    print_every=1,
                    temperature=1.0,
                ):
                    super().__init__()
                    self.max_tokens = max_tokens
                    self.start_tokens = list(start_tokens)
                    self.tokenizer = tokenizer
                    self.print_every = print_every
                    self.temperature = temperature

                def _sample(self, logits):
                    logits = np.asarray(logits, dtype="float64")
                    logits = logits / max(self.temperature, 1e-6)
                    logits -= logits.max()
                    probs = np.exp(logits)
                    probs /= probs.sum()
                    return int(np.random.choice(len(probs), p=probs))

                def on_epoch_end(self, epoch, logs=None):
                    if (epoch + 1) % self.print_every != 0:
                        return
                    import keras

                    tokens = list(self.start_tokens)
                    max_len = self.model.input_shape[-1]
                    for _ in range(self.max_tokens):
                        window = tokens[-max_len:]
                        padded = window + [self.tokenizer.pad_token_id] * (
                            max_len - len(window)
                        )
                        preds = self.model.predict(
                            np.array([padded]), verbose=0
                        )
                        next_id = self._sample(preds[0, len(window) - 1])
                        tokens.append(next_id)
                    text = self.tokenizer.decode(tokens)
                    print(f"\nEpoch {epoch + 1} sample: {text}")

            cls._impl = _TextGenerator
            cls._bound = True
        return cls._impl(*args, **kwargs)
