# Atlas-Free CNN Preprocessing Audit

## PubMed ALE cache

The old good PubMed-only 3D CNN cache is:

`experiments/3dcnn/atlas_free_cnn/data/ale_caches/atlas_free_4mm_fwhm9_crop_float16.pt`

Its saved preprocessing config is:

- source: PubMed MNI coordinates
- kernel: Gaussian ALE-style smoothing from coordinates, `fwhm=9.0 mm`
- target resolution: `4.0 mm`
- mask/crop: NeuroVLM brain mask resampled to 4 mm, cropped with slices `[[5, 41], [5, 50], [1, 39]]`
- output shape: `(36, 45, 38)`
- normalization: max-normalize each paper map
- values: clamped to `[0, 1]`
- dtype: `float16`

## HF-style atlas-free CNN pack

The packed mixed dataset is:

`experiments/3dcnn/atlas_free_cnn/cache/hf_atlas_free_cnn/atlas_free_cnn_volumes.pt`

The PubMed rows in that pack are not rebuilt or reprocessed. They are loaded
directly from the old ALE cache by `tensor_path`/`tensor_index`. A direct check
of the first 1000 PubMed rows found max absolute difference `0.0` against the
old cache, so PubMed image preprocessing is identical in the packed dataset.

The regenerated JSONL splits now point all sources at the packed shared CNN
tensor file and include all three intended sources:

- train: PubMed `24526`, NeuroVault `1779`, Nilearn `639`
- val: PubMed `3066`, NeuroVault `221`, Nilearn `79`
- test: PubMed `3066`, NeuroVault `202`, Nilearn `79`

## NeuroVault images

NeuroVault maps are already images, so no ALE kernel is applied. They are:

- loaded as NIfTI
- resampled to the same 4 mm NeuroVLM mask image used by the ALE cache
- cropped with the same brain crop convention to `(1, 36, 45, 38)`
- masked to brain voxels
- optionally clipped to positive values
- robust-percentile scaled to `[0, 1]`

This is intentionally different from PubMed coordinate processing because
NeuroVault inputs are statistical images, not coordinate lists.

## Nilearn images

Nilearn atlas/network maps are first generated as MNI152 2 mm NIfTI images.
When packed for CNN training, they are resampled to the same 4 mm NeuroVLM mask
and crop as the PubMed ALE cache. Binary atlas regions use nearest-neighbor
resampling; continuous/probabilistic maps use continuous interpolation. No ALE
kernel is applied.

## Current conclusion

The suspected PubMed preprocessing mismatch is not present in the packed CNN
data: PubMed volumes are byte-for-byte/effectively identical after tensor load.
The remaining preprocessing risk is source distribution and image-source
normalization, especially NeuroVault robust-percentile scaling producing much
denser maps than PubMed ALE maps.
