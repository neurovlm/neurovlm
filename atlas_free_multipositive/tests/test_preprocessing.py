import numpy as np
import nibabel as nib

from atlas_free_multipositive.data_building.preprocessing import nifti_metadata


def test_nifti_metadata_reports_shape_affine_resolution():
    img = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), np.diag([2, 2, 2, 1]))
    meta = nifti_metadata(img)

    assert meta["shape"] == [2, 3, 4]
    assert meta["resolution"] == [2.0, 2.0, 2.0]
    assert len(meta["affine"]) == 4

