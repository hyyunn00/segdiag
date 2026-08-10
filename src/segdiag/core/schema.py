"""Standardized long-format tables every downstream check reads.

These two dataclasses are the data-layer foundation for the collect/check/
writer pipeline: :func:`segdiag.core.pipeline.collect` is the only place that
scans folders, reads TIFFs, and runs the matching algorithm; everything else
(checks, reports, plots) queries the tables built from these records instead
of touching image files directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InstanceRecord:
    """One row = one GT or predicted instance, labeled once across the
    entire 3D (sample, model) volume - not once per 2D slice (see
    ``segdiag.core.pipeline.collect``).

    ``slice_name``/``z_index`` identify the single representative slice
    nearest this instance's 3D centroid (``round(centroid_z)``), not "the
    slice this instance was found on" - a 3D instance has no single owning
    slice, it just happens to be centered near one.

    This is the atomic unit every downstream check (FN characteristics, ROI,
    detection curves, FP root-cause...) is derived from. Nothing re-reads
    TIFFs after this table exists.
    """

    dataset: str
    sample: str
    model: str
    slice_name: str
    z_index: int
    role: str  # "gt" | "prediction"
    instance_id: int
    volume: int
    mean_intensity: Optional[float]
    centroid_z: float
    centroid_y: float
    centroid_x: float
    bbox_min_z: int
    bbox_min_row: int
    bbox_min_col: int
    bbox_max_z: int
    bbox_max_row: int
    bbox_max_col: int
    classification: str  # "true_positive" | "blind_fn" | "merged_fn" | "false_positive"
    best_iou: Optional[float]
    matched_instance_id: Optional[int]
    fp_subtype: Optional[str] = None  # filled in later by the fp_root_cause check
    #: Whether this instance is claimed under the position-lenient one-to-one
    #: match (``min_iou=core.matching.LOCATED_IOU_THRESHOLD``), independent
    #: of ``classification``/``matched_instance_id`` (the strict-IoU match).
    #: True for a "gt" row means this GT cell was really touched by *some*
    #: prediction, even if not closely enough to count as strict TP; True
    #: for a "prediction" row means this otherwise-FP prediction did land on
    #: a real GT cell rather than pure background noise/hallucination. See
    #: SEGDIAG_MARS_ALIGNMENT_COMPLETE.md Part 3 / checks.cell_count_agreement.
    located_matched: Optional[bool] = None
    #: For a "gt" row classified ``true_positive`` only: the Z-span (in
    #: slices) of the *prediction* instance that claimed this GT cell under
    #: the strict one-to-one match, i.e. ``matched_instance_id``'s Z extent.
    #: Filled directly here rather than requiring a join against a separate
    #: "prediction" row, because a matched/claimed prediction never gets its
    #: own row in this table - only unmatched predictions (false positives)
    #: do (see the ``fps``-only loop in ``core.pipeline._volume_instance_rows``).
    #: Used by ``checks.representative_case_gallery`` to flag GT cells the
    #: model only responded to on a single Z-slice despite spanning several
    #: (see SEGDIAG_REPRESENTATIVE_CASE_GALLERY.md section 2).
    matched_pred_z_span: Optional[int] = None
    #: For a "prediction" row only: how many local-background standard
    #: deviations this FP's mean intensity sits from the local background
    #: mean (``abs(mean_intensity - background_mean) / background_std``),
    #: i.e. the same ratio ``checks.fp_root_cause.classify_fp_subtype`` uses
    #: for its (looser, 1.5 sigma) ``noise_fp`` rule, but kept as a
    #: continuous value here instead of collapsing it into a threshold
    #: decision - ``checks.representative_case_gallery`` re-thresholds it at
    #: a stricter 2 sigma to find "missing_gt_annotation" candidates within
    #: ``hallucination_fp`` (see SEGDIAG_REPRESENTATIVE_CASE_GALLERY.md
    #: section 3). ``None`` whenever ``fp_root_cause.compute_local_background_stats``
    #: itself returned ``None`` (no raw image, or no true-background voxels
    #: in the padded neighbourhood).
    background_contrast_sigma: Optional[float] = None


@dataclass(frozen=True)
class ImageQualityRecord:
    """One row = one raw/GT slice's image-quality metrics."""

    dataset: str
    sample: str
    slice_name: str
    z_index: int
    source: str  # "raw" | "gt_mask" | "prediction"
    mean_intensity: float
    std_intensity: float
    snr_estimate: float
    saturation_pct: float  # % pixels at max dtype value
    laplacian_variance: float  # focus/blur proxy
    gt_instance_count: int
    gt_touching_border_pct: float  # possibly-truncated annotations
