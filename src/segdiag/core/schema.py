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
    """One row = one GT or predicted instance on one slice.

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
    centroid_y: float
    centroid_x: float
    bbox_min_row: int
    bbox_min_col: int
    bbox_max_row: int
    bbox_max_col: int
    classification: str  # "true_positive" | "blind_fn" | "merged_fn" | "false_positive"
    best_iou: Optional[float]
    matched_instance_id: Optional[int]
    fp_subtype: Optional[str] = None  # filled in later by the fp_root_cause check


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
