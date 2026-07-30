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
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        assert all(
            output.get("output_type") != "error"
            for cell in code_cells
            for output in cell.get("outputs", ())
        )
        code = "\n".join(
            "".join(cell.get("source", ()))
            for cell in code_cells
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


def _notebook_code(name):
    notebook = json.loads((ROOT / "docs/cnn/evaluation" / name).read_text())
    return "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_comparison_notebooks_use_documented_full_or_runtime_conscious_splits():
    contrastive = _notebook_code("contrastive_comparison.ipynb")
    assert "LIMIT_PER_DOMAIN = None" in contrastive

    for name in ("autoencoder_comparison.ipynb", "text_to_brain_comparison.ipynb"):
        notebook = json.loads((ROOT / "docs/cnn/evaluation" / name).read_text())
        code = _notebook_code(name)
        markdown = "\n".join(
            "".join(cell.get("source", ()))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert '"pubmed": 200' in code
        assert '"nilearn": None' in code
        assert '"neurovault": None' in code
        assert "DOMAIN_LIMITS[domain]" in code
        assert "3,066 PubMed" in markdown
        assert 'DOMAIN_LIMITS["pubmed"] = None' in markdown


def test_cnn_tutorial_visualizes_reconstruction_and_top_five_image_to_text_results():
    notebook = json.loads(
        (ROOT / "docs/tutorials/06_atlas_free_cnn.ipynb").read_text()
    )
    code = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "TOP_K = 5" in code
    assert "candidate_rows = test_data.rows" in code
    assert "query_scores.topk" in code
    assert '"is_known_pair"' in code
    assert 'set_title("Original")' in code
    assert 'set_title("Reconstructed")' in code


def test_comparison_notebooks_include_requested_qualitative_examples():
    autoencoder = _notebook_code("autoencoder_comparison.ipynb")
    assert "VISUAL_EXAMPLES_PER_DOMAIN = 3" in autoencoder
    assert "_first_nonempty_examples" in autoencoder
    assert 'plot_reconstruction_domain("pubmed")' in autoencoder
    assert 'plot_reconstruction_domain("nilearn")' in autoencoder
    assert 'plot_reconstruction_domain("neurovault")' in autoencoder
    assert "fig, axes = plt.subplots(count, 3" in autoencoder

    text_to_brain = _notebook_code("text_to_brain_comparison.ipynb")
    assert "EXAMPLES_PER_DOMAIN = 3" in text_to_brain
    assert "generated_examples[domain] = generated" in text_to_brain
    assert "paired_originals[domain]" in text_to_brain
    assert "_first_nonempty_rows" in text_to_brain
    assert "paired_slice_specs[domain]" in text_to_brain
    assert "paired_text" in text_to_brain
    assert "len(DOMAINS), EXAMPLES_PER_DOMAIN" in text_to_brain


def test_historical_cnn_notebooks_were_moved_out_of_experiments():
    assert not list((ROOT / "experiments/3dcnn").glob("*.ipynb"))
    assert not list((ROOT / "experiments/3dcnn/model_comparison").glob("*.ipynb"))
