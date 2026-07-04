"""Lab 2.1 — clean raw text (HTML + Unicode). Run: python examples/01_preprocess.py"""

from small_llm.preprocess import clean_html, clean_unicode

html = (
    "<p>The Krowor Municipal District was carved out in 2018 &amp; its"
    " population is &gt; 200000.</p>"
)
print("clean_html:  ", clean_html(html))

messy = "Bag of rice now cost ₦150000. Ah! \U0001f631 Edakun o"
print("clean_unicode:", clean_unicode(messy))
