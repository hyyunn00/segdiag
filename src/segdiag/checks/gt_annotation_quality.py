"""GT annotation quality: "is there a problem with the ground truth itself?"

Three checks on the GT population (independent of any model's predictions):

- **Volume outliers** (IQR rule): abnormally small cells are often noise
  mislabeled as a cell; abnormally large ones are often two touching cells
  annotated as one (a labeling merge).
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
import pandas as pd

from segdiag.checks.base import Check
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)

#: Density-anomaly sensitivity: flag a slice when its GT count deviates from
#: its sample's median by more than this many median-absolute-deviations
#: (with a floor of 1 count, so a perfectly uniform sample never false-flags
#: on floating point noise).
DENSITY_ANOMALY_MAD_MULTIPLIER = 3.0


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

        # --- volume outliers (IQR rule) ---
        q1, q3 = gt["volume"].quantile(0.25), gt["volume"].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        gt["volume_outlier"] = "normal"
        gt.loc[gt["volume"] < lower, "volume_outlier"] = "too_small"
        gt.loc[gt["volume"] > upper, "volume_outlier"] = "too_large"
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
        axes[0].axvline(lower, color="r", linestyle="--", label=f"IQR lower={lower:.1f}")
        axes[0].axvline(upper, color="r", linestyle="--", label=f"IQR upper={upper:.1f}")
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
                metadata={"iqr_bounds": {"lower": float(lower), "upper": float(upper)}},
            ),
            ReportArtifact(name="gt_annotation_quality_border", table=border_summary),
            ReportArtifact(name="gt_annotation_quality_density_anomalies", table=density_anomalies),
        ]
