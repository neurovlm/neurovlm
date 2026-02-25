"""brain_input.py — deprecated.

The search and resampling functions previously defined here have been
consolidated into :mod:`neurovlm.user_retrieval`.  Import from there instead::

    from neurovlm.user_retrieval import (
        search_text_corpus_given_neuroimage,
        resample_nifti,
        resample_array_nifti,
        resample_networks_to_mask,
        _load_mask_bundle,
        _resample_to_mask,
    )
"""
