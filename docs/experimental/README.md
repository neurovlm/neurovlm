# Experimental Notebooks

This directory contains active research branches, exploratory analyses,
superseded evaluation notebooks, and supporting experiments that are not
required to reproduce the figures in NeuroVLM preprint v3.

These notebooks are retained for research history and future development. They
may depend on unreleased artifacts, older APIs, external models, or experimental
configurations.

## Contents

- `evaluation/`: non-v3 qualitative and generative evaluation notebooks.
- [`cnn/`](cnn/technical_guide.md): the retained atlas-free CNN training and
  evaluation branch.
- `data_preparation/`: auxiliary corpus, term-extraction, and dataset-artifact
  preparation research plus its support scripts.

For the publication reproduction path, start with
[`../figures/README.md`](../figures/README.md).

```{toctree}
:hidden:
:maxdepth: 1

cnn/technical_guide
```
