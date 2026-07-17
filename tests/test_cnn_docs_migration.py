import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "docs/cnn/training/autoencoder.ipynb",
    "docs/cnn/training/contrastive_pubmed.ipynb",
    "docs/cnn/training/contrastive_and_text_to_brain.ipynb",
    "docs/cnn/training/architecture_background.ipynb",
    "docs/cnn/evaluation/autoencoder_comparison.ipynb",
    "docs/cnn/evaluation/contrastive_comparison.ipynb",
    "docs/cnn/evaluation/text_to_brain_comparison.ipynb",
    "docs/tutorials/06_atlas_free_cnn.ipynb",
)
EVIDENCE = ROOT / "docs/cnn/evaluation/artifacts/contrastive_retrieval"


def test_cnn_notebooks_are_valid_package_only_workflows():
    forbidden = (
        "experiments/3dcnn",
        "atlas_free_cnn",
        "sys.path",
        "FileNotFoundError",
        "hf_atlas_free_cnn_rebuild",
        "/Users/",
        "/content/",
    )
    for relative in NOTEBOOKS:
        path = ROOT / relative
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        code = "\n".join(
            "".join(cell.get("source", ()))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        compile(code, str(path), "exec")
        assert not any(token in code for token in forbidden)


def test_preserved_contrastive_evidence_is_portable_and_unchanged():
    for path in EVIDENCE.rglob("*"):
        if path.suffix in {".csv", ".json"}:
            content = path.read_text(encoding="utf-8")
            assert "/Users/" not in content
            assert "experiments/3dcnn" not in content

    with (EVIDENCE / "contrastive_retrieval_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    values = {
        (row["dataset"], row["requested_model_id"]): float(
            row["mean_normalized_k_recall_curve_auc"]
        )
        for row in rows
    }
    expected = {
        ("pubmed", "cnn_contrastive_mixed"): 0.8076171875,
        ("pubmed", "cnn_contrastive_pubmed"): 0.8046875,
        ("nilearn", "cnn_contrastive_mixed"): 0.9072265625,
        ("nilearn", "cnn_contrastive_nilearn"): 0.9072265625,
        ("neurovault", "cnn_contrastive_mixed"): 0.8505859375,
        ("neurovault", "cnn_contrastive_neurovault"): 0.845703125,
    }
    assert {key: values[key] for key in expected} == expected


def test_historical_cnn_notebooks_were_moved_out_of_experiments():
    assert not list((ROOT / "experiments/3dcnn").glob("*.ipynb"))
    assert not list((ROOT / "experiments/3dcnn/model_comparison").glob("*.ipynb"))
