Documentation
-------------
- setup site
- example: https://voyteklab.com/timescale-methods/

Docstrings
----------
- standard format for doc strings across package
- type hinting

API
---
- document each module / sub-module, e.g. all core .py files
- example: https://raw.githubusercontent.com/voytekresearch/timescale-methods/refs/heads/main/docs/api.rst

Readme
------
- brief introduction to package
- how to install
- minimal example
- example: https://raw.githubusercontent.com/voytekresearch/timescale-methods/refs/heads/main/README.rst

Pypi
----
- release the package so it can be pip installed
- example https://pypi.org/project/neurodsp/
- how to push to pypi from local, see here https://github.com/neurodsp-tools/neurodsp/blob/f5b7b69b187309b568c21f0a316cdb3453fb81e3/Makefile#L97

Tests
-----
- tests for each core/import function or class
- example: https://github.com/voytekresearch/timescale-methods/blob/main/timescales/tests/autoreg/test_spectral.py

Tutorials
---------
These should be high-level with minimal code, e.g. use a lot of imported function and abstract low-level detail away from the user.

- 01_introduction.py
	- How to fetch models/datasets from huggingface
	- Types of models/architecture:
		- autoencoder
		- text encoder (specter2)
		- contrastive
		- generative
	- Types of datasets
		- pubmed
		- neurovault
		- networks
		- neurowiki
	- Explain text-to-brain and brain-to-text
- 02_contrastive.py
	- Brain-to-text
		- Network Labeling
		- ICA Labeling (HCP and  UK Biobank from network dataset)
		- NeuroVault Labeling
	- Text-to-Brain
		- Given text query, return most similar brains from a dataset
- 03_generative_text-to-brain.ipynb
	- examples of: text_query -> specter2 -> proj_head -> autoencoder.decoder -> decoded_brain
- 04_generative_brain-to-text.ipynb
	- Extend CLIP like concept labelling to text generation using a small LLM, e.g. < 1B.
