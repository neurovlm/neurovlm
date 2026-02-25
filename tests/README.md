# NeuroVLM Test Suite

This directory contains comprehensive tests for the NeuroVLM package.

## Test Organization

Tests are organized by module:

- `test_core.py` - Tests for the main NeuroVLM class and result containers
- `test_data.py` - Tests for data loading functions (load_dataset, load_latent, load_masker)
- `test_models.py` - Tests for model classes (NeuroAutoEncoder, ProjHead, Specter)
- `test_metrics.py` - Tests for metric functions (dice, recall, BPP, etc.)
- `test_loss.py` - Tests for loss functions (FocalLoss, InfoNCELoss, TruncatedLoss)
- `test_io.py` - Tests for model save/load functionality

## Running Tests

### Run all tests

```bash
pytest
```

### Run tests for a specific module

```bash
pytest tests/test_metrics.py
pytest tests/test_loss.py
```

### Run a specific test class or function

```bash
pytest tests/test_metrics.py::TestDice
pytest tests/test_loss.py::TestFocalLoss::test_focal_loss_forward
```

### Run tests with verbose output

```bash
pytest -v
```

### Run tests with coverage

```bash
pytest --cov=neurovlm --cov-report=html
```

### Skip slow tests

```bash
pytest -m "not slow"
```

### Skip tests requiring downloaded data

```bash
pytest -m "not requires_data"
```

## Test Markers

Tests are marked with custom markers to help organize and filter them:

- `@pytest.mark.slow` - Tests that take a long time to run
- `@pytest.mark.requires_data` - Tests that require downloaded datasets
- `@pytest.mark.requires_pretrained` - Tests that require pretrained models

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `set_random_seeds` - Automatically sets random seeds for reproducibility
- `device` - Returns the appropriate torch device (CPU or CUDA)

## Writing New Tests

When writing new tests:

1. Follow the naming convention: `test_<function_name>` or `Test<ClassName>`
2. Use descriptive test names that explain what is being tested
3. Group related tests in classes
4. Use fixtures to avoid code duplication
5. Mark tests appropriately (slow, requires_data, etc.)
6. Add docstrings to test classes and methods

### Example Test Structure

```python
class TestMyFunction:
    """Tests for my_function."""

    def test_basic_functionality(self):
        """Test that my_function works with basic inputs."""
        result = my_function(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test my_function handles edge cases."""
        result = my_function(edge_case_input)
        assert result is not None

    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_multiple_inputs(self, input_val, expected):
        """Test my_function with multiple inputs."""
        assert my_function(input_val) == expected
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. The test suite should:

- Complete in a reasonable time (< 5 minutes for quick tests)
- Not depend on external services unless properly mocked
- Clean up any temporary files or resources
- Be deterministic (same input = same output)

## Dependencies

Test dependencies are listed in the main `requirements.txt` or `pyproject.toml`:

- pytest
- pytest-cov (for coverage reports)
- torch
- numpy
- pandas
- scikit-learn
- scikit-image

## Notes

- Some tests may be skipped if required data or models are not available
- Tests requiring CUDA will automatically skip if CUDA is not available
- All tests use fixed random seeds for reproducibility
