from collections import Counter

from atlas_free_multipositive.data_building.build_unified_dataset import split_rows


def test_split_rows_preserves_source_proportions():
    rows = []
    for source, n in (("pubmed", 100), ("neurovault", 20), ("nilearn:schaefer_2018", 10)):
        for i in range(n):
            rows.append({"map_id": f"{source}_{i}", "source": source, "positive_texts": [{"text": "x"}]})

    splits = split_rows(rows, seed=0, val_frac=0.1, test_frac=0.1)

    counts = {name: Counter(row["source"] for row in split) for name, split in splits.items()}
    assert counts["train"] == {"pubmed": 80, "neurovault": 16, "nilearn:schaefer_2018": 8}
    assert counts["val"] == {"pubmed": 10, "neurovault": 2, "nilearn:schaefer_2018": 1}
    assert counts["test"] == {"pubmed": 10, "neurovault": 2, "nilearn:schaefer_2018": 1}
