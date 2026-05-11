import numpy as np

from atlas_free_multipositive.data_building.ingest_nilearn_atlases import _iter_3d_components, _labels


def test_singleton_4d_label_atlas_is_split_as_labels():
    data = np.zeros((3, 3, 3, 1), dtype=np.float32)
    data[0, 0, 0, 0] = 1
    data[1, 1, 1, 0] = 2
    labels = ["Background", "Network 1", "Network 2"]

    components = list(_iter_3d_components(data, labels, "yeo_2011", "network_map"))

    assert len(components) == 2
    assert components[0][1] == "Network 1"
    assert components[0][3] is True


def test_recarray_labels_prefer_readable_name_column():
    labels = np.rec.array(
        [("Mode A", "Visual", 1.0), ("Mode B", "Default", 2.0)],
        dtype=[("difumo_names", "U16"), ("yeo_networks7", "U16"), ("x", "f4")],
    )

    assert _labels({"labels": labels}) == ["Mode A", "Mode B"]
