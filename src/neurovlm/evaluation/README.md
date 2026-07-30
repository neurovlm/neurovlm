# Evaluation APIs

This package contains reusable reconstruction, contrastive retrieval, text-to-brain, brain-to-text, MLP, and model-comparison evaluators. `default_comparison_matrix` selects MLP plus mixed-baseline CNN models; fine-tuned CNN rows are added only with `include_finetuned=True`. The comparison data defaults to `AtlasFreeCNNDataProvider`, which reads published Hugging Face resources and ignores legacy row paths.
