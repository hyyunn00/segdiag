"""Raw input-image quality vs. recall/FP rate.

Answers: "is low recall a modeling problem, or is the raw image itself
low-quality (noisy, blurry, saturated, low-contrast)?"

:func:`compute_image_quality_metrics` is a pure function with no CLI/report
dependencies, so :func:`segdiag.core.pipeline.collect` can call it directly
while scanning to populate ``ImageQualityRecord`` rows - the check itself
only aggregates and plots the table collect() already built.
"""

from __future__ import annotations

import argparse
import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import laplace

from segdiag.checks.base import Check
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)


def compute_image_quality_metrics(gt_arr: np.ndarray, raw_arr: np.ndarray) -> dict:
    """Compute the per-slice raw-image-quality metrics backing
    ``ImageQualityRecord`` (everything except the dataset/sample/slice/z
    identity columns, which the caller already knows).

    - ``snr_estimate``: foreground (GT-covered) mean intensity over
      background (outside GT) standard deviation - a simple, annotation-
      informed signal-to-noise proxy.
    - ``laplacian_variance``: variance of the Laplacian of the raw image;
      lower means blurrier.
    - ``saturation_pct``: percentage of pixels sitting at the image's own
      max value (saturated/clipped).
    - ``gt_instance_count`` / ``gt_touching_border_pct``: how many GT
      instances are on this slice, and what fraction touch the image
      border (and are therefore possibly truncated annotations).
    """
    raw_float = raw_arr.astype(np.float64)
    fg_mask = gt_arr > 0
    bg_mask = ~fg_mask

    fg_mean = float(raw_float[fg_mask].mean()) if fg_mask.any() else float("nan")
    bg_std = float(raw_float[bg_mask].std()) if bg_mask.any() else float("nan")
    if bg_mask.any() and bg_std > 1e-9 and not np.isnan(fg_mean):
        snr_estimate = float(fg_mean / bg_std)
    else:
        snr_estimate = 0.0

    max_val = raw_float.max()
    saturation_pct = float((raw_float == max_val).mean() * 100.0)
    laplacian_variance = float(laplace(raw_float).var())

    from skimage.measure import label as sk_label
    from skimage.measure import regionprops

    gt_labels = sk_label(gt_arr > 0, connectivity=None)
    props = regionprops(gt_labels)
    gt_instance_count = len(props)
    if gt_instance_count:
        h, w = gt_arr.shape
        touching = sum(
            1 for p in props if p.bbox[0] == 0 or p.bbox[1] == 0 or p.bbox[2] == h or p.bbox[3] == w
        )
        gt_touching_border_pct = float(touching / gt_instance_count * 100.0)
    else:
        gt_touching_border_pct = 0.0

    return {
        "mean_intensity": float(raw_float.mean()),
        "std_intensity": float(raw_float.std()),
        "snr_estimate": snr_estimate,
        "saturation_pct": saturation_pct,
        "laplacian_variance": laplacian_variance,
        "gt_instance_count": gt_instance_count,
        "gt_touching_border_pct": gt_touching_border_pct,
    }


class RawImageQualityCheck(Check):
    name = "raw-image-quality"
    description = "Raw input image quality vs. recall/FP rate (is low recall a data problem?)"

    def run(
        self, instances: pd.DataFrame, quality: pd.DataFrame, args: argparse.Namespace
    ) -> List[ReportArtifact]:
        if quality.empty:
            logger.warning(
                "No image-quality data collected (no raw images found) - nothing to report."
            )
            return []

        gt = instances[instances["role"] == "gt"]
        fp = instances[
            (instances["role"] == "prediction") & (instances["classification"] == "false_positive")
        ]

        per_sample_quality = (
            quality.groupby("sample")
            .agg(
                mean_snr_estimate=("snr_estimate", "mean"),
                mean_laplacian_variance=("laplacian_variance", "mean"),
                mean_saturation_pct=("saturation_pct", "mean"),
                mean_intensity=("mean_intensity", "mean"),
            )
            .reset_index()
        )

        recall_by_sample = (
            gt.groupby("sample")["classification"]
            .apply(lambda s: (s == "true_positive").mean())
            .rename("recall")
        )
        gt_count_by_sample = gt.groupby("sample").size().rename("gt_count")
        fp_count_by_sample = (
            fp.groupby("sample").size().rename("fp_count")
            if not fp.empty
            else pd.Series(dtype="int64", name="fp_count")
        )

        summary = per_sample_quality.set_index("sample").join(
            [recall_by_sample, gt_count_by_sample, fp_count_by_sample], how="left"
        )
        summary["fp_count"] = summary["fp_count"].fillna(0)
        summary["recall"] = summary["recall"].fillna(0.0)
        summary["fp_rate"] = (summary["fp_count"] / summary["gt_count"].replace(0, np.nan)).fillna(
            0.0
        )
        summary = summary.reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(summary["mean_snr_estimate"], summary["recall"])
        for _, row in summary.iterrows():
            axes[0].annotate(row["sample"], (row["mean_snr_estimate"], row["recall"]))
        axes[0].set_xlabel("Mean SNR estimate")
        axes[0].set_ylabel("Recall")
        axes[0].set_title("Recall vs. raw image SNR")

        axes[1].scatter(summary["mean_laplacian_variance"], summary["fp_rate"])
        for _, row in summary.iterrows():
            axes[1].annotate(row["sample"], (row["mean_laplacian_variance"], row["fp_rate"]))
        axes[1].set_xlabel("Mean Laplacian variance (focus)")
        axes[1].set_ylabel("FP rate")
        axes[1].set_title("FP rate vs. raw image blur")

        plt.tight_layout()

        return [
            ReportArtifact(name="raw_image_quality_per_sample", table=summary, figure=fig),
            ReportArtifact(
                name="raw_image_quality_per_slice", table=quality.reset_index(drop=True)
            ),
        ]
