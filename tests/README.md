# NeuroVLM tests

The suite is organized by scope and subsystem:

```text
tests/
├── conftest.py
├── unit/
│   ├── api/
│   ├── data/
│   ├── metrics/
│   ├── models/
│   ├── pipelines/
│   └── resources/
└── integration/
    ├── api/
    ├── evaluation/
    └── training/
```

Unit tests cover isolated functions, model definitions, loaders with mocked
resources, metrics, and run infrastructure. Integration tests exercise complete
inference, evaluation, and small offline training workflows.

## Install test dependencies

```bash
pip install -e ".[test]"
```

## Run tests

Run the deterministic offline suite:

```bash
pytest -m "not network and not requires_data and not requires_pretrained and not requires_specter and not slow"
```

Run only unit tests:

```bash
pytest tests/unit
```

Run only integration tests:

```bash
pytest tests/integration
```

Run every test, including tests that may download data or pretrained models:

```bash
pytest
```

Generate coverage:

```bash
pytest --cov=neurovlm --cov-report=term-missing
```

## Markers

- `unit`: isolated, offline behavior
- `integration`: multi-component behavior
- `slow`: tests unsuitable for the quick suite
- `network`: requires an external service
- `requires_data`: requires downloaded datasets
- `requires_pretrained`: requires released model weights
- `requires_specter`: requires a Hugging Face SPECTER model

New tests should be deterministic, use `tmp_path` for artifacts, mock network
access unless explicitly marked, and test behavior rather than notebook prose
or frozen experiment output.
