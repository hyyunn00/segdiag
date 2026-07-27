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
