from small_llm.tokenization import (
    BPEWordTokenizer,
    character_tokenize,
    get_pair_frequencies,
    get_token_counts,
    learn_bpe,
    merge_pair_in_word,
    preprocess_text,
    space_tokenize,
    word_tokenize,
)


def test_basic_tokenizers():
    assert character_tokenize("ab") == ["a", "b"]
    assert space_tokenize("a b c") == ["a", "b", "c"]
    assert word_tokenize("Hello, world!") == ["Hello", "world"]


def test_preprocess_and_counts():
    tokens = preprocess_text(["The cat.", "A cat!"])
    assert tokens == ["the", "cat", "a", "cat"]
    counts = get_token_counts(tokens)
    assert counts["cat"] == 2


def test_functional_bpe_merges():
    corpus = [["l", "o", "w", "</w>"], ["l", "o", "w", "</w>"]]
    freqs = get_pair_frequencies(corpus)
    assert freqs[("l", "o")] == 2
    merged = merge_pair_in_word(corpus[0], ("l", "o"))
    assert merged == ["lo", "w", "</w>"]

    merges, vocab, _ = learn_bpe(["low low lower"], num_merges=3)
    assert len(merges) == 3
    assert "lo" in vocab


def test_bpe_tokenizer_roundtrip():
    tok = BPEWordTokenizer(["the quick brown fox", "the lazy dog"], num_merges=20)
    ids = tok.encode("the fox")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids)
    assert decoded == "the fox"


def test_bpe_tokenizer_unknown_char():
    tok = BPEWordTokenizer(["hello world"], num_merges=10)
    ids = tok.encode("hello")
    # Every id must be valid within the vocabulary.
    assert all(0 <= i < tok.vocabulary_size for i in ids)


def test_bpe_save_load(tmp_path):
    tok = BPEWordTokenizer(["the quick brown fox"], num_merges=15)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPEWordTokenizer.load(path)
    assert loaded.vocabulary == tok.vocabulary
    assert loaded.encode("the fox") == tok.encode("the fox")
