"""Step 4: Dataset Bias and Detection-Probability Curves.

1. Plots the volume/intensity distribution of the GT population (dataset bias).
2. Plots the empirical detection rate P(TP) as a function of volume and
   intensity, binned into deciles.

Reads straight from ``collect()``'s instances table - no TIFFs are read
here.
"""

from __future__ import annotations

import argparse
import logging
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from segdiag.checks.base import Check
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)


class DetectionProbabilityCheck(Check):
    name = "detection-probability"
    description = "Step 4: Dataset bias & detection-probability curves"

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
            print("\n--- Per-model detection rate ---")
            print(gt.groupby("model")["is_tp"].mean().rename("recall").round(4))

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        sns.histplot(
            data=gt,
            x="volume",
            log_scale=True,
            bins=30,
            ax=axes[0, 0],
            color="gray",
            stat="percent",
        )
        axes[0, 0].set_title("Ground Truth Distribution: Volume (Log Scale)", fontsize=14)
        axes[0, 0].set_xlabel("Volume (voxels)")
        axes[0, 0].set_ylabel("Percentage of Total Cells (%)")

        sns.histplot(data=gt, x="intensity", bins=30, ax=axes[0, 1], color="gray", stat="percent")
        axes[0, 1].set_title("Ground Truth Distribution: Intensity", fontsize=14)
        axes[0, 1].set_xlabel("Mean intensity")
        axes[0, 1].set_ylabel("Percentage of Total Cells (%)")

        gt["volume_bin"] = pd.qcut(gt["volume"], q=10, duplicates="drop")
        vol_prob = gt.groupby("volume_bin", observed=True)["is_tp"].mean().reset_index()
        vol_prob["volume_mid"] = vol_prob["volume_bin"].apply(lambda x: x.mid)

        axes[1, 0].plot(
            vol_prob["volume_mid"],
            vol_prob["is_tp"],
            marker="o",
            linestyle="-",
            color="b",
            linewidth=2,
        )
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].set_title("Detection Probability: P(TP) vs Volume", fontsize=14)
        axes[1, 0].set_xlabel("Cell Volume (Voxels)")
        axes[1, 0].set_ylabel("Detection Rate (P(TP))")

        gt["intensity_bin"] = pd.qcut(gt["intensity"], q=10, duplicates="drop")
        int_prob = gt.groupby("intensity_bin", observed=True)["is_tp"].mean().reset_index()
        int_prob["intensity_mid"] = int_prob["intensity_bin"].apply(lambda x: x.mid)

        axes[1, 1].plot(
            int_prob["intensity_mid"],
            int_prob["is_tp"],
            marker="o",
            linestyle="-",
            color="r",
            linewidth=2,
        )
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].set_title("Detection Probability: P(TP) vs Intensity", fontsize=14)
        axes[1, 1].set_xlabel("Mean Intensity")
        axes[1, 1].set_ylabel("Detection Rate (P(TP))")

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
                "is_tp",
            ]
        ].reset_index(drop=True)

        return [ReportArtifact(name="step4_detection_probability", table=detail_table, figure=fig)]
