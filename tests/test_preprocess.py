from small_llm.preprocess import clean_html, clean_html_tags, clean_unicode


def test_clean_html_tags_removes_tags():
    text = "<h3>Let's <strong>win</strong> this.<br></h3>"
    assert clean_html_tags(text) == "Let's win this."


def test_clean_html_decodes_entities():
    text = "<p>2018 &amp; population &gt; 200000.</p>"
    assert clean_html(text) == "2018 & population > 200000."


def test_clean_unicode_keeps_letters_numbers_punctuation():
    text = "Bag of rice now cost ₦150000. Ah! \U0001f631 Edakun o"
    cleaned = clean_unicode(text)
    assert "\U0001f631" not in cleaned  # emoji removed
    assert "₦" not in cleaned  # currency symbol removed
    assert "!" in cleaned  # punctuation kept
    assert "150000" in cleaned  # numbers kept


def test_clean_unicode_preserves_whitespace():
    assert clean_unicode("a b\tc") == "a b\tc"
