# Run infrastructure

The pipeline utilities standardize run identity, lifecycle status, requested/effective configuration, provenance, atomic CSV/JSON output, metric history, best/last checkpoints, architecture compatibility, hashes, and resume. Training entry points should use `RunContext`, `MetricRecorder`, and `CheckpointManager` rather than constructing task-specific artifact layouts.
