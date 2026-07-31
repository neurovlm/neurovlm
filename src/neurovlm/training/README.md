# Training APIs

Use the typed configs and `train_*` functions exported from `neurovlm.training`. CNN autoencoder, contrastive, and text-to-brain workflows use the published atlas-free provider by default. MLP runners accept an explicit dataset provider. Every runner writes the same config, provenance, checkpoint, metric, plot, generated-map, and log directories and supports checkpoint-safe resume.

For CNN branches, `mixed_baseline` is the default; `finetuned` must be requested explicitly. Prefer released initialization, and use `from_run` only for deliberate local chaining. See `docs/05_cnn/technical_guide.md` for task examples.
