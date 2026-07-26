"""Step 5: FN/FP Contribution and Stratified (ROI) Analysis.

For volume bins (<50, 50-100, 100-150, >150 voxels), computes:

1. The bin's share of the total GT population.
2. The bin's instance recall.
3. The bin's contribution to the total False Negative count -
   i.e. "is chasing small cells actually worth the effort?"
4. The same size-bucket breakdown for False Positives (spurious
   predictions with no real GT partner) - i.e. "are the model's
   hallucinated detections concentrated in the small/noisy end, or is it
   inventing large, plausible-looking cells?"

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

VOLUME_BIN_EDGES = [0, 50, 100, 150, np.inf]
VOLUME_BIN_LABELS = ["<50", "50-100", "100-150", ">150"]


class FnContributionCheck(Check):
    name = "fn-contribution"
    description = "Step 5: Stratified FN/FP contribution / ROI analysis by cell size"

    def run(
        self, instances: pd.DataFrame, quality: pd.DataFrame, args: argparse.Namespace
    ) -> List[ReportArtifact]:
        gt = instances[instances["role"] == "gt"].copy()
        fp = instances[
            (instances["role"] == "prediction") & (instances["classification"] == "false_positive")
        ].copy()

        if gt.empty:
            logger.error("No data collected.")
            return []

        gt["is_fn"] = (gt["classification"] != "true_positive").astype(int)
        gt["is_tp"] = (gt["classification"] == "true_positive").astype(int)

        if gt["model"].nunique() > 1:
            logger.info("Multiple models included in this run: %s", sorted(gt["model"].unique()))

        gt["volume_bin"] = pd.cut(
            gt["volume"], bins=VOLUME_BIN_EDGES, labels=VOLUME_BIN_LABELS, right=False
        )
        if not fp.empty:
            fp["volume_bin"] = pd.cut(
                fp["volume"], bins=VOLUME_BIN_EDGES, labels=VOLUME_BIN_LABELS, right=False
            )

        total_gt = len(gt)
        total_fn = gt["is_fn"].sum()
        total_fp = len(fp)

        summary = (
            gt.groupby("volume_bin", observed=True)
            .agg(
                total_cells=("is_fn", "count"),
                fn_count=("is_fn", "sum"),
                tp_count=("is_tp", "sum"),
            )
            .reset_index()
        )

        summary["pct_of_total_gt"] = (summary["total_cells"] / total_gt) * 100
        summary["recall_pct"] = (summary["tp_count"] / summary["total_cells"]) * 100
        summary["pct_contribution_to_total_fn"] = (
            (summary["fn_count"] / total_fn) * 100 if total_fn else 0.0
        )

        if not fp.empty:
            fp_summary = (
                fp.groupby("volume_bin", observed=True)
                .size()
                .reindex(VOLUME_BIN_LABELS, fill_value=0)
                .rename("fp_count")
                .reset_index()
            )
            fp_summary["pct_contribution_to_total_fp"] = (
                (fp_summary["fp_count"] / total_fp) * 100 if total_fp else 0.0
            )
        else:
            fp_summary = pd.DataFrame(
                {
                    "volume_bin": VOLUME_BIN_LABELS,
                    "fp_count": 0,
                    "pct_contribution_to_total_fp": 0.0,
                }
            )

        print("\n" + "=" * 80)
        print(" ROI check: is chasing small cells actually worth the effort?")
        print("=" * 80)
        print(summary.to_string(index=False, float_format="%.2f"))
        print(f"\nFalse positives (spurious predictions, no GT partner): {total_fp}")
        print(fp_summary.to_string(index=False, float_format="%.2f"))
        print("=" * 80 + "\n")

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].pie(
            summary["total_cells"],
            labels=summary["volume_bin"],
            autopct="%1.1f%%",
            colors=sns.color_palette("pastel"),
        )
        axes[0, 0].set_title("1. Ground Truth Composition", fontsize=14, fontweight="bold")

        sns.barplot(
            data=summary,
            x="volume_bin",
            y="recall_pct",
            hue="volume_bin",
            palette="Blues_d",
            legend=False,
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("2. Instance Recall by Volume", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Volume bin")
        axes[0, 1].set_ylabel("Recall (%)")
        axes[0, 1].set_ylim(0, 100)
        for i, v in enumerate(summary["recall_pct"]):
            axes[0, 1].text(i, v + 2, f"{v:.1f}%", color="black", ha="center")

        fn_counts = summary["fn_count"].values
        if np.sum(fn_counts) > 0:
            axes[1, 0].pie(
                fn_counts,
                labels=summary["volume_bin"],
                autopct="%1.1f%%",
                colors=sns.color_palette("Reds"),
            )
        axes[1, 0].set_title(
            "3. Contribution to Total False Negatives", fontsize=14, fontweight="bold"
        )

        fp_counts = fp_summary["fp_count"].values
        if np.sum(fp_counts) > 0:
            axes[1, 1].pie(
                fp_counts,
                labels=fp_summary["volume_bin"],
                autopct="%1.1f%%",
                colors=sns.color_palette("Purples"),
            )
        else:
            axes[1, 1].text(0.5, 0.5, "No false positives", ha="center", va="center")
            axes[1, 1].axis("off")
        axes[1, 1].set_title(
            "4. Contribution to Total False Positives", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()

        return [
            ReportArtifact(
                name="step5_fn_contribution",
                table=summary,
                figure=fig,
                metadata={"total_fp": int(total_fp)},
            ),
            ReportArtifact(name="step5_fn_contribution_fp_summary", table=fp_summary),
        ]
