# Testing Guide for NeuroVLM

This document provides comprehensive information about the test suite for NeuroVLM.

## Overview

The test suite covers all major components of the NeuroVLM package:

- **Core functionality** (`test_core.py`) - Main NeuroVLM class, result containers, utility functions
- **Data loading** (`test_data.py`) - Dataset and latent loading functions
- **Models** (`test_models.py`) - Neural network architectures
- **Metrics** (`test_metrics.py`) - Performance metrics (Dice, recall, BPP, etc.)
- **Loss functions** (`test_loss.py`) - Training losses (Focal, InfoNCE, Truncated)
- **I/O** (`test_io.py`) - Model save/load functionality

## Installation

### Install testing dependencies

```bash
pip install -r tests/requirements.txt
```

Or install individually:

```bash
pip install pytest pytest-cov pytest-xdist
```

## Running Tests

### Quick Start

Run all tests:
```bash
pytest
```

Run with verbose output:
```bash
pytest -v
```

Run tests in parallel (faster):
```bash
pytest -n auto
```

### Run Specific Tests

Run a single test file:
```bash
pytest tests/test_metrics.py
```

Run a specific test class:
```bash
pytest tests/test_metrics.py::TestDice
```

Run a specific test function:
```bash
pytest tests/test_metrics.py::TestDice::test_dice_perfect_match
```

### Filtering Tests

Run only fast tests (skip slow ones):
```bash
pytest -m "not slow"
```

Skip tests requiring data downloads:
```bash
pytest -m "not requires_data"
```

Skip tests requiring pretrained models:
```bash
pytest -m "not requires_pretrained"
```

### Code Coverage

Generate coverage report:
```bash
pytest --cov=neurovlm --cov-report=html
```

View the coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

Generate terminal coverage report:
```bash
pytest --cov=neurovlm --cov-report=term-missing
```

## Test Structure

### Test Files

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and configuration
├── pytest.ini               # Pytest configuration (in root)
├── README.md                # Test documentation
├── requirements.txt         # Test dependencies
├── test_core.py            # Tests for core module (401 lines)
├── test_data.py            # Tests for data module (239 lines)
├── test_io.py              # Tests for io module (183 lines)
├── test_loss.py            # Tests for loss module (228 lines)
├── test_metrics.py         # Tests for metrics module (229 lines)
└── test_models.py          # Tests for models module (239 lines)
```

### Test Organization

Each test file follows this structure:

```python
"""Tests for <module_name> module."""

import pytest
# ... imports ...

class Test<ClassName>:
    """Tests for <ClassName> class."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for tests."""
        # ... setup code ...
        return data

    def test_basic_functionality(self):
        """Test basic functionality."""
        # ... test code ...
        assert result == expected

    def test_edge_case(self):
        """Test edge case handling."""
        # ... test code ...

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
    ])
    def test_multiple_cases(self, input, expected):
        """Test with multiple parameterized inputs."""
        assert function(input) == expected
```

## Test Coverage Summary

### test_metrics.py
- ✅ Dice score computation (perfect, no overlap, partial overlap, empty)
- ✅ Dice top-k computation
- ✅ Recall@k metrics
- ✅ Recall curves
- ✅ Bernoulli BCE
- ✅ Bits per pixel calculation
- ✅ Autoencoder performance metrics

### test_loss.py
- ✅ FocalLoss (initialization, forward pass, gamma=0 recovery)
- ✅ FocalWithLogitsLoss (with/without pos_weight)
- ✅ InfoNCELoss (symmetry, temperature effects)
- ✅ TruncatedLoss (L1/MSE variants, percentile filtering)

### test_io.py
- ✅ Model saving to safetensors
- ✅ Model loading (eval/train mode, CPU/CUDA)
- ✅ Save/load round-trip consistency
- ✅ Parameter preservation

### test_models.py
- ✅ NormalizeLayer
- ✅ NeuroAutoEncoder (encoder/decoder, latent normalization, activations)
- ✅ ProjHead (forward pass, custom dimensions)
- ✅ ConceptClf
- ✅ Model loading functions
- ✅ Integration tests

### test_data.py
- ✅ Data loading functions (load_dataset, load_latent, load_masker)
- ✅ Dataset aliases
- ✅ Gradient removal utility (_without_grad)
- ✅ Data directory management
- ✅ Dataset/latent consistency checks

### test_core.py
- ✅ L2 normalization utility
- ✅ TextSearchResult (top_k, formatting, dataset filtering)
- ✅ BrainSearchResult (retrieval and generation modes)
- ✅ BrainTopKResult (chainable results)
- ✅ Utility functions (text cleaning, ID normalization, type checking)
- ✅ Dataset canonicalization

## Writing New Tests

### Guidelines

1. **Naming Convention**
   - Test files: `test_<module_name>.py`
   - Test classes: `Test<ClassName>`
   - Test functions: `test_<what_is_being_tested>`

2. **Test Structure**
   - Group related tests in classes
   - Use descriptive docstrings
   - One assertion per test (when possible)
   - Use fixtures for setup/teardown

3. **Fixtures**
   - Define common fixtures in `conftest.py`
   - Use `@pytest.fixture` for reusable test data
   - Clean up resources in fixtures

4. **Markers**
   - `@pytest.mark.slow` - For tests taking >1 second
   - `@pytest.mark.requires_data` - Requires downloaded datasets
   - `@pytest.mark.requires_pretrained` - Requires pretrained models
   - `@pytest.mark.skipif` - Conditional skipping

5. **Parametrization**
   ```python
   @pytest.mark.parametrize("input_val,expected", [
       (1, 2),
       (2, 4),
       (3, 6),
   ])
   def test_multiple_inputs(self, input_val, expected):
       assert function(input_val) == expected
   ```

6. **Exception Testing**
   ```python
   with pytest.raises(ValueError, match="error message"):
       function_that_should_raise()
   ```

### Example: Adding a New Test

```python
# In tests/test_metrics.py

class TestNewMetric:
    """Tests for new_metric function."""

    def test_new_metric_basic(self):
        """Test new_metric with basic inputs."""
        from neurovlm.metrics import new_metric

        result = new_metric([1, 2, 3], [1, 2, 3])
        assert result == 1.0

    def test_new_metric_edge_case(self):
        """Test new_metric handles empty inputs."""
        from neurovlm.metrics import new_metric

        result = new_metric([], [])
        assert result == 0.0
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -e .
        pip install -r tests/requirements.txt

    - name: Run tests
      run: pytest --cov=neurovlm --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

**Import errors**
```bash
# Make sure neurovlm is installed
pip install -e .
```

**Missing test dependencies**
```bash
pip install -r tests/requirements.txt
```

**Tests requiring data failing**
```bash
# Skip tests requiring downloads
pytest -m "not requires_data"
```

**CUDA tests failing on CPU-only machine**
```bash
# Tests automatically skip if CUDA unavailable
# No action needed
```

## Best Practices

1. **Run tests before committing**
   ```bash
   pytest
   ```

2. **Write tests for bug fixes**
   - Add a test that reproduces the bug
   - Fix the bug
   - Verify the test passes

3. **Maintain test coverage**
   - Aim for >80% coverage
   - Test edge cases and error conditions
   - Test both success and failure paths

4. **Keep tests fast**
   - Mock external dependencies
   - Use small test data
   - Mark slow tests appropriately

5. **Make tests deterministic**
   - Set random seeds (done automatically)
   - Avoid time-dependent tests
   - Clean up side effects

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing best practices](https://docs.python-guide.org/writing/tests/)

## Contact

For questions about the test suite, please open an issue on GitHub.
