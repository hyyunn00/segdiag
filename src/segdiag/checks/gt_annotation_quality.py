"""GT annotation quality: "is there a problem with the ground truth itself?"

Three checks on the GT population (independent of any model's predictions):

- **Volume outliers** (MAD rule on log-volume): abnormally small cells are
  often noise mislabeled as a cell; abnormally large ones are often two
  touching cells annotated as one (a labeling merge). Cell volume is
  heavily right-skewed (log-normal-ish, not normal), so the outlier rule
  runs on ``log(volume)`` rather than raw volume - an IQR/MAD computed on
  the raw scale is dominated by the long right tail and both under-flags
  subtle small-cell outliers and over-flags merely-large-but-typical cells.
  Uses the median absolute deviation (MAD), not IQR, for the same reason
  the density-anomaly check below does: a single robust statistic that
  isn't itself thrown off by the outliers it's trying to detect.
- **Border truncation**: a GT instance whose bounding box touches the image
  edge is very likely an incomplete cell (cut off by the field of view), and
  shouldn't be scored the same way as a fully-imaged cell.
- **Density anomalies**: a sample/slice whose GT cell count jumps sharply
  relative to its own sample's typical density can indicate an annotator
  switch or inconsistent labeling standards partway through a stack.
"""

from __future__ import annotations

import argparse
import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from segdiag.checks.base import Check
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)

#: Density-anomaly sensitivity: flag a slice when its GT count deviates from
#: its sample's median by more than this many median-absolute-deviations
#: (with a floor of 1 count, so a perfectly uniform sample never false-flags
#: on floating point noise).
DENSITY_ANOMALY_MAD_MULTIPLIER = 3.0

#: Volume-outlier sensitivity: flag a GT cell when log(volume)'s "modified
#: z-score" (Iglewicz & Hoaglin) exceeds this many MAD-scaled units from the
#: sample's median log-volume. 3.5 is the standard threshold from that
#: method - the 0.6745 constant in the modified z-score makes MAD
#: comparable to a normal distribution's standard deviation.
VOLUME_OUTLIER_MODIFIED_Z_THRESHOLD = 3.5
_MAD_CONSISTENCY_CONSTANT = 0.6745


class GtAnnotationQualityCheck(Check):
    name = "gt-annotation-quality"
    description = (
        "GT annotation quality: volume outliers, border-truncated cells, density anomalies"
    )

    def run(
        self, instances: pd.DataFrame, quality: pd.DataFrame, args: argparse.Namespace
    ) -> List[ReportArtifact]:
        gt = instances[instances["role"] == "gt"].copy()
        if gt.empty:
            logger.warning("No GT instances collected - nothing to report.")
            return []

        # --- volume outliers (MAD rule on log-volume) ---
        log_volume = np.log(gt["volume"].astype(float))
        median_log = log_volume.median()
        mad_log = (log_volume - median_log).abs().median()
        modified_z = _MAD_CONSISTENCY_CONSTANT * (log_volume - median_log) / max(mad_log, 1e-9)

        lower = float(
            np.exp(
                median_log
                - VOLUME_OUTLIER_MODIFIED_Z_THRESHOLD * mad_log / _MAD_CONSISTENCY_CONSTANT
            )
        )
        upper = float(
            np.exp(
                median_log
                + VOLUME_OUTLIER_MODIFIED_Z_THRESHOLD * mad_log / _MAD_CONSISTENCY_CONSTANT
            )
        )

        gt["volume_outlier"] = "normal"
        gt.loc[modified_z < -VOLUME_OUTLIER_MODIFIED_Z_THRESHOLD, "volume_outlier"] = "too_small"
        gt.loc[modified_z > VOLUME_OUTLIER_MODIFIED_Z_THRESHOLD, "volume_outlier"] = "too_large"
        outliers = gt.loc[
            gt["volume_outlier"] != "normal",
            [
                "dataset",
                "sample",
                "model",
                "slice_name",
                "z_index",
                "instance_id",
                "volume",
                "volume_outlier",
            ],
        ].reset_index(drop=True)

        # --- border truncation ---
        # Only the "near" border (row/col == 0) is checkable from the
        # instance table alone; the fully shape-aware check (including the
        # far border) already lives in quality_df's gt_touching_border_pct.
        gt["touches_near_border"] = (gt["bbox_min_row"] == 0) | (gt["bbox_min_col"] == 0)
        border_summary = (
            gt.groupby("sample")["touches_near_border"]
            .mean()
            .rename("near_border_pct")
            .reset_index()
        )
        if not quality.empty:
            full_border = (
                quality.groupby("sample")["gt_touching_border_pct"]
                .mean()
                .rename("gt_touching_border_pct")
                .reset_index()
            )
            border_summary = border_summary.merge(full_border, on="sample", how="left")

        # --- density anomalies ---
        density = (
            gt.groupby(["sample", "slice_name", "z_index"]).size().rename("gt_count").reset_index()
        )
        density = density.sort_values(["sample", "z_index"])

        anomaly_frames = []
        for _, group in density.groupby("sample"):
            median = group["gt_count"].median()
            mad = (group["gt_count"] - median).abs().median()
            threshold = max(mad * DENSITY_ANOMALY_MAD_MULTIPLIER, 1.0)
            flagged = group[(group["gt_count"] - median).abs() > threshold].copy()
            flagged["sample_median"] = median
            anomaly_frames.append(flagged)
        density_anomalies = (
            pd.concat(anomaly_frames, ignore_index=True)
            if anomaly_frames
            else pd.DataFrame(
                columns=["sample", "slice_name", "z_index", "gt_count", "sample_median"]
            )
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].hist(gt["volume"], bins=30, color="gray")
        axes[0].axvline(lower, color="r", linestyle="--", label=f"MAD(log) lower={lower:.1f}")
        axes[0].axvline(upper, color="r", linestyle="--", label=f"MAD(log) upper={upper:.1f}")
        axes[0].set_title("GT Volume Distribution with Outlier Bounds")
        axes[0].set_xlabel("Volume (voxels)")
        axes[0].legend()

        for sample_name, group in density.groupby("sample"):
            axes[1].plot(group["z_index"], group["gt_count"], marker="o", label=sample_name)
        axes[1].set_title("GT Density per Slice (looking for cliffs)")
        axes[1].set_xlabel("Z index")
        axes[1].set_ylabel("GT cell count")
        axes[1].legend()

        plt.tight_layout()

        return [
            ReportArtifact(
                name="gt_annotation_quality_outliers",
                table=outliers,
                figure=fig,
                metadata={"log_volume_mad_bounds": {"lower": lower, "upper": upper}},
            ),
            ReportArtifact(name="gt_annotation_quality_border", table=border_summary),
            ReportArtifact(name="gt_annotation_quality_density_anomalies", table=density_anomalies),
        ]
