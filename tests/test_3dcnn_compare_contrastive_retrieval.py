from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
THREEDCNN = REPO_ROOT / "experiments" / "3dcnn"
if str(THREEDCNN) not in sys.path:
    sys.path.insert(0, str(THREEDCNN))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_free_cnn.evaluation import compare_contrastive_retrieval as compare


def _base() -> dict[str, str]:
    return {
        "dataset": "pubmed",
        "requested_model_id": "fake",
        "model_id": "fake",
        "model_family": "fake",
        "checkpoint_path": "",
    }


def _records(n: int) -> list[dict[str, object]]:
    return [
        {
            "sample_index": i,
            "map_id": f"map-{i}",
            "text_id": f"text-{i}",
            "source": "pubmed",
            "source_detail": "unit_test",
            "tensor_index": i,
        }
        for i in range(n)
    ]


def test_fake_perfect_diagonal_retrieval_beats_shuffled_pairs() -> None:
    text = torch.eye(5)
    perfect_brain = torch.eye(5)
    shuffled_brain = perfect_brain.roll(shifts=1, dims=0)

    perfect = compare.retrieval_summary_row(text, perfect_brain, base=_base())
    shuffled = compare.retrieval_summary_row(text, shuffled_brain, base=_base())

    assert perfect["status"] == "ok"
    assert perfect["n_pairs"] == 5
    assert perfect["t2i_mrr"] == 1.0
    assert perfect["i2t_mrr"] == 1.0
    assert perfect["t2i_normalized_k_recall_curve_auc"] > shuffled["t2i_normalized_k_recall_curve_auc"]
    assert perfect["i2t_normalized_k_recall_curve_auc"] > shuffled["i2t_normalized_k_recall_curve_auc"]
    assert perfect["normalized_k_recall_curve_auc"] > shuffled["normalized_k_recall_curve_auc"]


def test_curve_and_example_rows_include_directional_output_columns() -> None:
    text = torch.eye(4)
    brain = torch.eye(4)

    curves = compare.recall_curve_rows(text, brain, base=_base())
    examples = compare.retrieval_example_rows(text, brain, _records(4), base=_base())

    assert len(curves) == 4
    assert {"k", "normalized_k", "t2i_recall", "i2t_recall", "mean_recall", "random_recall"} <= set(curves[0])
    assert curves[0]["t2i_recall"] == 1.0
    assert curves[0]["i2t_recall"] == 1.0
    assert {"t2i_rank", "i2t_rank", "t2i_top1_map_id", "i2t_top1_text_id", "matched_similarity"} <= set(examples[0])
    assert all(row["t2i_rank"] == 1 for row in examples)
    assert all(row["i2t_rank"] == 1 for row in examples)


def test_missing_cnn_checkpoint_writes_skipped_summary_row(monkeypatch) -> None:
    def fake_resolve(model_ids):
        return {
            model_ids[0]: {
                "family": "cnn_contrastive",
                "domain": "pubmed",
                "branch": "specialized",
                "status": "missing_checkpoint",
                "checkpoint_path": None,
                "error": "checkpoint not uploaded yet",
            }
        }

    monkeypatch.setattr(compare.registry, "resolve_model_registry", fake_resolve)

    summary, curves, examples = compare.evaluate_model_dataset(
        dataset_name="pubmed",
        model_id="cnn_contrastive_pubmed",
        device="cpu",
        batch_size=2,
        limit=2,
        test_jsonl=None,
        text_cache={},
        text_cache_spec={"convention": "normalized_specter2", "local_cache_path": "/fake/cache.pt"},
    )

    assert len(summary) == 1
    assert summary[0]["status"] == "missing_checkpoint"
    assert summary[0]["n_pairs"] == 0
    assert "not uploaded yet" in summary[0]["skip_reason"]
    assert curves == []
    assert examples == []


def test_load_mlp_raw_pairs_binarizes_neurovault_images(monkeypatch) -> None:
    import neurovlm.data as neurovlm_data

    continuous_images = torch.tensor([[0.0, 2.5, -1.0], [0.3, 0.0, -0.2]])
    text_embeddings = torch.eye(2)

    def fake_load_dataset(name: str):
        assert name == "neurovault_images"
        return continuous_images

    def fake_load_latent(name: str):
        assert name == "neurovault_text"
        return text_embeddings

    monkeypatch.setattr(neurovlm_data, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(neurovlm_data, "load_latent", fake_load_latent)

    raw_text, flat, records, split_strategy = compare.load_mlp_raw_pairs("neurovault", limit=None)

    assert torch.equal(flat, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    assert set(flat.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(raw_text, text_embeddings)
    assert split_strategy == "main_neurovault_all_aligned_pairs_binarized"
    assert len(records) == 2


def test_cli_help_parser_accepts_prompt_model_alias() -> None:
    parser = compare.build_arg_parser()
    args = parser.parse_args(["--datasets", "pubmed", "--models", "cnn_contrastive_mixed", "--limit", "8"])

    assert args.datasets == ["pubmed"]
    assert args.models == ["cnn_contrastive_mixed"]
    assert args.limit == 8
    assert compare.normalize_model_id_for_dataset("cnn_contrastive_mixed", "pubmed") == (
        "cnn_contrastive_mixed_to_pubmed",
        "mixed_to_pubmed",
    )
