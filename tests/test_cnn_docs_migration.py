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


def test_mixed_baseline_decision_is_documented_without_external_artifacts():
    guide = (ROOT / "docs/cnn/technical_guide.md").read_text(encoding="utf-8")
    notebook_path = ROOT / "docs/cnn/evaluation/contrastive_comparison.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_markdown = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    decision_record = f"{guide}\n{notebook_markdown}"

    for value in ("0.807617", "0.804688", "0.907227", "0.850586", "0.845703"):
        assert value in decision_record
    assert "32 test examples per domain" in guide
    assert "mixed baseline as the safe default" in guide
    assert "do not establish universal superiority" in guide
    assert "artifacts/contrastive_retrieval" not in decision_record


def test_historical_cnn_notebooks_were_moved_out_of_experiments():
    assert not list((ROOT / "experiments/3dcnn").glob("*.ipynb"))
    assert not list((ROOT / "experiments/3dcnn/model_comparison").glob("*.ipynb"))
