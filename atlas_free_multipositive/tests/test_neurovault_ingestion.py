from atlas_free_multipositive.data_building.ingest_neurovault import (
    NeuroVaultConfig,
    build_text_positives,
    collection_id_from_image,
    image_id_from_summary,
    is_probably_volumetric_nifti,
    _looks_binary_volume,
    quality_flags_for_metadata,
    quality_score,
    quality_tier,
    usable_download_url,
)


def test_neurovault_quality_scoring_prefers_metadata_rich_maps():
    image = {
        "name": "Working memory contrast",
        "description": "Activation map for a two-back greater than zero-back working memory contrast.",
        "task": "n-back task",
        "doi": "10.1234/example",
        "file": "https://neurovault.org/media/images/1/map.nii.gz",
    }
    collection = {
        "name": "Working memory study",
        "description": "A functional MRI study of cognitive control and working memory.",
    }

    flags = quality_flags_for_metadata(image, collection, download_url=image["file"])
    score = quality_score(flags, image, collection)

    assert flags["missing_metadata"] is False
    assert flags["no_task_label"] is False
    assert flags["no_doi_or_pmid"] is False
    assert score >= NeuroVaultConfig().strong_quality_score
    assert quality_tier(score, flags, NeuroVaultConfig()) == "strong"


def test_neurovault_quality_scoring_skips_sparse_metadata():
    image = {
        "name": "",
        "file": "https://neurovault.org/media/images/1/map.nii.gz",
    }
    collection = {}

    flags = quality_flags_for_metadata(image, collection, download_url=image["file"])
    score = quality_score(flags, image, collection)

    assert flags["missing_metadata"] is True
    assert flags["no_task_label"] is True
    assert flags["no_doi_or_pmid"] is True
    assert flags["low_quality_text"] is True
    assert quality_tier(score, flags, NeuroVaultConfig()) == "skipped"


def test_neurovault_text_positives_include_required_sources():
    image = {
        "name": "Face greater than shape",
        "description": "A statistical map for viewing faces compared with shape stimuli.",
        "cognitive_concepts": [{"name": "face perception", "definition": "Perception and interpretation of faces."}],
        "pmid": "12345",
    }
    collection = {
        "name": "Social perception fMRI",
        "description": "A collection of fMRI maps for social and visual perception tasks.",
        "paper_title": "Neural systems for face perception",
    }

    positives = build_text_positives(image, collection, map_id="neurovault_1")
    categories = {pos["category"] for pos in positives}
    texts = [pos["text"] for pos in positives]

    assert "image_description" in categories
    assert "cognitive_task_or_contrast" in categories
    assert "collection_description" in categories
    assert "paper_title" in categories
    assert any("[SEP]" in text for text in texts)


def test_neurovault_download_url_and_nifti_filter():
    image = {"file": "/media/images/10/stat_map.nii.gz"}
    url = usable_download_url(image, api_base="https://neurovault.org/api/")

    assert url == "https://neurovault.org/media/images/10/stat_map.nii.gz"
    assert is_probably_volumetric_nifti(image, url) is True
    assert is_probably_volumetric_nifti({"file": "surface.gii"}, "https://example.org/surface.gii") is False


def test_neurovault_ids_can_be_extracted_from_api_urls():
    assert image_id_from_summary({"url": "https://neurovault.org/api/images/123/"}) == "123"
    assert collection_id_from_image({"collection": "https://neurovault.org/api/collections/45/"}) == "45"


def test_binary_volume_detection_for_nearest_resampling():
    assert _looks_binary_volume([[0, 1], [1, 0]]) is True
    assert _looks_binary_volume([[0.0, 0.5], [1.0, 0.0]]) is False
