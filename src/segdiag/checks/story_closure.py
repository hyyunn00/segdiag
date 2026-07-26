"""Step 6: The Final Story Closure.

1. Plots a Volume x Intensity detection-rate heatmap (checks for interaction
   effects between the two variables).
2. Plots the best-IoU distribution stratified by volume bin, to distinguish
   "never detected" cells from "detected but poorly matched" cells.

Reads straight from ``collect()``'s instances table - no TIFFs are read
here.
"""

from __future__ import annotations

import argparse
import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from segdiag.checks.base import Check
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)


class StoryClosureCheck(Check):
    name = "story-closure"
    description = "Step 6: Final heatmap + stratified IoU story-closure report"

    def run(
        self, instances: pd.DataFrame, quality: pd.DataFrame, args: argparse.Namespace
    ) -> List[ReportArtifact]:
        gt = instances[instances["role"] == "gt"].copy()
        if gt.empty:
            logger.error("No data collected.")
            return []

        gt = gt.rename(columns={"mean_intensity": "intensity"})
        gt["is_tp"] = (gt["classification"] == "true_positive").astype(int)

        if gt["model"].nunique() > 1:
            logger.info("Multiple models included in this run: %s", sorted(gt["model"].unique()))

        vol_bins = [0, 50, 100, 150, np.inf]
        vol_labels = ["<50", "50-100", "100-150", ">150"]
        gt["volume_bin"] = pd.cut(gt["volume"], bins=vol_bins, labels=vol_labels, right=False)

        # Quartile intensity bins ensure every bucket has enough cells.
        gt["intensity_bin"] = pd.qcut(
            gt["intensity"], q=4, labels=["Dark", "Med-Dark", "Med-Bright", "Bright"]
        )

        fig = plt.figure(figsize=(20, 12))

        ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
        pivot_table = gt.pivot_table(
            values="is_tp",
            index="volume_bin",
            columns="intensity_bin",
            aggfunc="mean",
            observed=True,
        )
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".1%",
            cmap="YlGnBu",
            cbar_kws={"label": "Detection Rate (Recall)"},
            ax=ax1,
        )
        ax1.set_title(
            "Detection Rate: Volume vs. Intensity Interaction", fontsize=16, fontweight="bold"
        )
        ax1.set_xlabel("Intensity Quartiles", fontsize=12)
        ax1.set_ylabel("Volume Bins (Voxels)", fontsize=12)

        axes_iou = [
            plt.subplot2grid((2, 4), (1, 0)),
            plt.subplot2grid((2, 4), (1, 1)),
            plt.subplot2grid((2, 4), (1, 2)),
            plt.subplot2grid((2, 4), (1, 3)),
        ]

        colors = ["#ff9999", "#ffcc99", "#99ccff", "#66b3ff"]

        for i, (vol_label, color) in enumerate(zip(vol_labels, colors)):
            ax = axes_iou[i]
            subset = gt[gt["volume_bin"] == vol_label]["best_iou"]

            sns.histplot(subset, bins=20, binrange=(0, 1), color=color, ax=ax, stat="percent")
            ax.axvline(x=0.5, color="r", linestyle="--", linewidth=2)

            zero_iou_pct = (subset < 0.05).mean() * 100 if len(subset) else 0.0

            ax.set_title(
                f"Volume: {vol_label}\n(IoU~0: {zero_iou_pct:.1f}%)", fontsize=14, fontweight="bold"
            )
            ax.set_xlabel("Best IoU", fontsize=12)
            ax.set_ylabel("Percentage of Cells (%)" if i == 0 else "")
            ax.set_xlim(-0.05, 1.05)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        detail_table = gt[
            [
                "dataset",
                "sample",
                "model",
                "slice_name",
                "z_index",
                "instance_id",
                "volume",
                "intensity",
                "best_iou",
                "is_tp",
                "volume_bin",
                "intensity_bin",
            ]
        ].reset_index(drop=True)

        return [ReportArtifact(name="step6_story_closure", table=detail_table, figure=fig)]
