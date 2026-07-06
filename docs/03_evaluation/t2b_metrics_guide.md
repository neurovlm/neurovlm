# Text-to-Brain Evaluation Metrics

This document describes the metrics used in `22_text_to_brain_metrics_colab.ipynb`
for the Text -> Brain direction. The local notebook uses the same metric logic; the
Colab notebook is treated as the most up-to-date reference because it is configured
for the full evaluation run.

Text-to-brain metric primitives and workflows live in
`src/neurovlm/text_to_brain_metrics.py`. Shared retrieval-curve helpers live in
`src/neurovlm/retrieval_metrics.py`, and `src/neurovlm/metrics.py` remains only as
a backward-compatible import index for older code.

---

## Overview

| Metric | Type | Scale | What it asks |
|---|---|---|---|
| Pearson r | Whole-brain voxelwise correlation | -1 to 1 | Does the predicted map match the target map's voxelwise intensity pattern? |
| Pearson minus random | Dataset-specific baseline comparison | Usually centered near 0 | Is the prediction closer to its own target than to random targets from the same dataset? |
| Spearman rho | Whole-brain rank correlation | -1 to 1 | Does the predicted map put high and low voxels in the same rank order as the target? |
| Spearman minus random | Dataset-specific rank baseline comparison | Usually centered near 0 | Is the rank pattern more target-specific than random matched targets? |
| Dice pct | Top-activation overlap | 0 to 1 | Do the strongest predicted activation regions overlap the strongest target regions? |
| Spin p-value | Spatial null significance test | 0 to 1, lower is better | Is the surface Dice stronger than expected under spatial rotations? |
| Dice threshold sensitivity | Robustness check | Curves/tables | Are Dice conclusions stable across percentile thresholds? |

For Pearson, Spearman, and Dice, higher is better. For spin p-value, lower is
better, with `p < 0.05` used as the conventional significance cutoff.

---

## Evaluation Setup

The text-to-brain notebook predicts a brain map from reference text:

```text
input text
  -> NeuroVLM text encoder / projection
  -> text-to-brain head
  -> decoder
  -> predicted NIfTI brain map
```

The predicted brain map is transformed into the same masker voxel space as the
target brain map, then evaluated against the target.

The notebook evaluates three datasets when enabled:

| Dataset | Text input | Target brain map |
|---|---|---|
| Networks | Canonical network description | Canonical network latent decoded to a brain map |
| PubMed | Paper title/summary text | PubMed image latent decoded to a brain map |
| NeuroVault | Publication title/abstract text | Linked NeuroVault image latent decoded to a brain map |

For NeuroVault, a paper can have multiple linked maps. The current setting,
`NEUROVAULT_IMAGE_SELECTION = "argmax_pearson"`, generates one predicted brain map
from the paper text and selects the linked NeuroVault target map with the highest
Pearson correlation to that prediction. This makes the evaluation paper-level
instead of arbitrarily using the first uploaded NeuroVault image.

For PubMed, surface Dice and spin p-values are only used when the target activation
is sufficiently cortical. Mostly subcortical PubMed targets are still included in
whole-brain Pearson/Spearman, but excluded from surface Dice/spin summaries because
surface projection would not be a fair representation of those maps.

---

## 1. Pearson r

### What it measures

Pearson r measures linear voxelwise similarity between the predicted and target
brain maps. It asks whether high predicted activation tends to occur in voxels where
the target also has high activation, and low predicted activation tends to occur
where the target is low.

### How it is computed

```python
brain_pred = masker.transform(predicted_nifti).ravel()
brain_true = decoded_target_latent.ravel()
pearson_r = pearson_correlation(brain_true, brain_pred)
```

### How to interpret it

| Range | Interpretation |
|---|---|
| > 0.30 | Strong voxelwise alignment for this setting |
| 0.10 to 0.30 | Meaningful positive alignment |
| 0.00 to 0.10 | Weak alignment |
| < 0.00 | Anti-correlated or mismatched map |

Absolute values should not be read like generic image reconstruction scores. Brain
activation maps are sparse, noisy, and often have many plausible nearby patterns.
That is why the notebook also reports random-baseline comparisons.

---

## 2. Pearson Random Baseline

### What it measures

Pearson minus random asks whether the predicted map is more similar to its own
target than to other target maps from the same dataset.

This is important because a model could learn a generic "brain-looking" activation
pattern. Such a prediction might have nonzero Pearson correlation with many targets,
but it should not beat the correct target by much.

### How it is computed

For each prediction, the notebook samples random target maps from the same dataset
and computes Pearson correlation to those random targets. For speed, this baseline
uses a fixed voxel subset controlled by `T2B_RANDOM_BASELINE_SPEARMAN_MAX_VOXELS`.

The main output columns are:

| Column | Meaning |
|---|---|
| `pearson_baseline_actual` | Pearson between the prediction and its own target on the baseline voxel subset |
| `pearson_random_mean` | Mean Pearson against sampled random targets |
| `pearson_random_std` | Standard deviation of random-target Pearson scores |
| `pearson_minus_random` | `pearson_baseline_actual - pearson_random_mean` |
| `pearson_random_percentile` | Fraction of random targets scoring below the true target |

### How to interpret it

`pearson_minus_random > 0` means the prediction is closer to its own target than to
random targets from the same dataset. This is usually more interpretable than raw
Pearson alone.

`pearson_random_percentile` near 1.0 means the true target beats nearly all sampled
random targets. Values near 0.5 mean the true target is around random chance.

---

## 3. Spearman rho

### What it measures

Spearman rho measures rank-order similarity. Instead of comparing raw voxel values,
it compares whether the predicted and target maps rank voxels similarly from low to
high activation.

This can be useful when the predicted map gets the relative activation pattern right
but differs in scale or intensity calibration.

### How it is computed

```python
spearman_rho, _ = spearmanr(brain_true.ravel(), brain_pred.ravel())
```

### Random baseline columns

The notebook computes the same dataset-specific random-target baseline for Spearman:

| Column | Meaning |
|---|---|
| `spearman_baseline_actual` | Spearman between the prediction and its own target on the baseline voxel subset |
| `spearman_random_mean` | Mean Spearman against sampled random targets |
| `spearman_random_std` | Standard deviation of random-target Spearman scores |
| `spearman_minus_random` | `spearman_baseline_actual - spearman_random_mean` |
| `spearman_random_percentile` | Fraction of random targets scoring below the true target |

The main interpretation is the same: `spearman_minus_random > 0` means the predicted
rank pattern is more target-specific than random maps from the same dataset.

---

## 4. Dice pct

### What it measures

Dice pct measures overlap between the strongest activation regions in the predicted
and target maps. The notebook uses percentile thresholding:

```text
predicted top voxels = voxels above percentile DICE_PCT in predicted map
target top voxels    = voxels above percentile DICE_PCT in target map
dice_pct90           = overlap between those two binary maps
```

With the default `DICE_PCT = 90`, the metric compares the top 10% strongest voxels
in each map.

### How it is computed

```python
dice_val = dice_percentile(brain_pred, brain_true, pct=DICE_PCT)
```

When surface spin testing is enabled and the sample is surface-eligible, the maps
are projected to fsaverage and Dice is computed on the surface:

```text
NIfTI volume
  -> fsaverage surface projection
  -> percentile Dice
  -> spin-test null model
```

The output column is named according to the threshold, for example `dice_pct90`.

### How to interpret it

| Range | Interpretation |
|---|---|
| > 0.40 | Strong overlap of high-activation regions |
| 0.20 to 0.40 | Meaningful partial overlap |
| 0.05 to 0.20 | Weak but nonzero overlap |
| near 0.00 | Little overlap among strongest regions |

Dice is more spatially strict than Pearson/Spearman. A prediction can have decent
correlation but low Dice if it captures a broad pattern while missing the exact peak
regions.

---

## 5. Spin p-value

### What it measures

Spin p-value asks whether the observed surface Dice is stronger than expected under
a spatial null model. This matters because nearby cortical regions are spatially
autocorrelated; random voxel shuffling would be too optimistic.

The spin test rotates one surface map on a sphere many times, recomputes Dice for
each rotated null map, and asks how often the null overlap exceeds the observed
overlap.

### How it is computed

The Colab notebook uses:

```text
SPIN_USE_NEUROMAPS = True
SPIN_TEST_N_PERM = 1000
SPIN_FSAVERAGE_DENSITY = "41k"
SPIN_TRANSFORM_METHOD = "linear"
```

The main output columns are:

| Column | Meaning |
|---|---|
| `spin_p_value` | Permutation p-value for the observed Dice |
| `spin_significant` | Boolean, usually `spin_p_value < 0.05` |
| `spin_method` | Whether the spin test ran, or why it was skipped/failed |
| `surface_metric_eligible` | Whether Dice/spin should be interpreted for this row |
| `surface_metric_skip_reason` | Reason a row was excluded from surface metrics |

### How to interpret it

`spin_p_value < 0.05` means the high-activation overlap is unlikely under the
spatial rotation null. It is strongest when paired with a meaningful Dice value:
a tiny Dice score can be significant in some settings, and a moderate Dice score
can be nonsignificant if it is spatially easy to obtain by rotation.

---

## 6. Surface Eligibility

Surface Dice and spin tests are only meaningful when the target map can be fairly
projected to cortical surface space. The notebook builds a cortical eligibility mask
using Harvard-Oxford cortical regions when available, with an MNI cortical-shell
fallback if the atlas cannot be loaded.

For PubMed, the notebook checks how much of the top target activation mass is
cortical:

```text
cortical_top_mass_fraction = top activation mass inside cortical mask / total top activation mass
```

With the current default:

```text
PUBMED_MIN_CORTICAL_TOP_MASS_FRACTION = 0.50
```

PubMed samples below that threshold are excluded from Dice/spin interpretation but
kept for Pearson and Spearman. Networks and NeuroVault are treated as eligible unless
there is a technical failure.

---

## 7. Dice Threshold Sensitivity

Dice depends on the percentile threshold. The notebook therefore recomputes Dice and
spin significance across:

```text
DICE_SENSITIVITY_PCTS = [80, 85, 90, 95]
```

This produces a robustness check:

| Pattern | Interpretation |
|---|---|
| Dice stays positive across thresholds | The overlap conclusion is stable |
| Spin significance stays high across thresholds | The spatial result is robust |
| Result appears only at one threshold | The conclusion is threshold-sensitive |
| Dice drops sharply at high thresholds | The broad map is aligned, but exact peaks are less stable |

The notebook saves/plots this as `t2b_dice_pvalue_sensitivity.png`.

---

## Visualizations

The notebook includes four main visual checks:

| Plot | File | What it shows |
|---|---|---|
| Correlation random baseline plot | `t2b_correlation_random_baseline.png` | Actual Pearson/Spearman vs random-target baselines |
| Metric distribution plot | `t2b_metric_distributions.png` | Distribution of Pearson minus random, Spearman minus random, Dice, and spin p-values by dataset |
| Dice threshold sensitivity plot | `t2b_dice_pvalue_sensitivity.png` | Dice and spin-significant rate across thresholds |
| Dice vs spin significance | `t2b_dice_vs_pvalue.png` | Whether larger Dice also corresponds to stronger spatial significance |

---

## Output Files

The primary table is:

```text
text_to_brain_metrics_v2.csv
```

It contains one row per evaluated sample, excluding private in-memory array columns
such as `_brain_pred` and `_brain_true`.

Important columns include:

| Column | Meaning |
|---|---|
| `dataset` | `networks`, `pubmed`, or `neurovault` |
| `name` | Sample identifier |
| `text_input` | Truncated text input used to generate the brain map |
| `pearson_r` | Raw whole-brain Pearson correlation |
| `spearman_rho` | Raw whole-brain Spearman correlation |
| `pearson_minus_random` | Pearson improvement over random targets from the same dataset |
| `spearman_minus_random` | Spearman improvement over random targets from the same dataset |
| `dice_pct90` | Top-activation overlap at the default percentile threshold |
| `spin_p_value` | Spatial null p-value for surface Dice |
| `spin_significant` | Whether `spin_p_value < 0.05` |
| `surface_metric_eligible` | Whether Dice/spin should be interpreted |
| `surface_metric_skip_reason` | Why Dice/spin was skipped, if applicable |
| `neurovault_selection_mode` | NeuroVault target-map selection mode |
| `selected_candidate_pearson` | Pearson score of the selected NeuroVault target map |

---

## Summary: Which Metric To Lead With

| Situation | Primary metric | Supporting metrics |
|---|---|---|
| Whole-brain map similarity | **Pearson minus random** | Pearson r |
| Rank-pattern similarity independent of intensity scale | **Spearman minus random** | Spearman rho |
| Spatial overlap of strongest activation regions | **Dice pct** | Dice threshold sensitivity |
| Statistical evidence for spatial overlap | **Spin p-value** | Dice pct |
| Checking whether results are target-specific | **Random-baseline percentile / minus random** | Raw Pearson/Spearman |

For a single headline view, use **Pearson minus random** or **Spearman minus random**
for all evaluated samples, then use **Dice pct + spin p-value** as the spatial
overlap evidence for surface-eligible samples. Pearson/Spearman tell you whether the
full predicted map is target-specific; Dice/spin tell you whether the strongest
activation regions overlap in a spatially meaningful way.
