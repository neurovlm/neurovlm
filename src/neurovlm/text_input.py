"""text_input.py — deprecated.

The text retrieval functions previously defined here have been consolidated
into :mod:`neurovlm.user_retrieval`.  For searching a user-provided corpus
with a text query use::

    from neurovlm.user_retrieval import search_text_corpus_given_text

For the high-level NeuroVLM built-in datasets, use :class:`neurovlm.NeuroVLM`
directly::

    from neurovlm import NeuroVLM
    nvlm = NeuroVLM()
    results = nvlm.text("working memory").to_text()
"""
