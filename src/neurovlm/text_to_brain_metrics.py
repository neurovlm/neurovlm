"""Text-to-brain evaluation metrics and notebook workflow helpers."""

from neurovlm.evaluation_notebook_utils import (
    add_random_correlation_baseline,
    build_surface_eligibility_mask,
    evaluate_t2b_sample,
    finite_box_values,
    finite_sem,
    make_t2b_runner,
    sensitivity_rows_for_df,
)
from neurovlm.metrics import (
    dice_percentile,
    mni152_to_fsaverage_arrays,
    nct_dice_spin_test_nifti,
    nct_dice_spin_test_surface,
    pearson_correlation,
)

__all__ = [
    "add_random_correlation_baseline",
    "build_surface_eligibility_mask",
    "dice_percentile",
    "evaluate_t2b_sample",
    "finite_box_values",
    "finite_sem",
    "make_t2b_runner",
    "mni152_to_fsaverage_arrays",
    "nct_dice_spin_test_nifti",
    "nct_dice_spin_test_surface",
    "pearson_correlation",
    "sensitivity_rows_for_df",
]
