"""Text2Brain model code (Ngo et al., MICCAI 2021) from github.com/ngohgia/text2brain at commit 6f29389 (MIT, see LICENSE).

decoder.py is unchanged. text2brain_model.py has fix 1 for transformers 4.x (the encoder is called with return_dict=False)
and keeps the inputs on the model's device; each change is marked "NeuroVLM:". Fix 2 (drop the checkpoint key
encoder.embeddings.position_ids before the strict load) is in ../_text2brain.py.
"""
