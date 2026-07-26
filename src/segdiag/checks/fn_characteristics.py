"""Step 3: Quantify False-Negative *and* False-Positive Characteristics.

Compares four buckets - blind FN (best IoU < 0.05), merged FN (unmatched
but touched by something), true positive, and false positive (a spurious
prediction with no real GT partner) - along volume, mean intensity, and
Z-depth, to see whether misses are systematically smaller/dimmer/deeper,
and whether spurious detections look like noise or like real cells (a
direct read on over-segmentation/over-fragmentation).

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
from segdiag.core.matching import CLASSIFICATION_LABELS
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)

CLASS_ORDER = ["blind_fn", "merged_fn", "true_positive", "false_positive"]
CLASS_PALETTE = {
    "blind_fn": "#ff9999",
    "merged_fn": "#ffcc99",
    "true_positive": "#66b3ff",
    "false_positive": "#b399ff",
}


class FnCharacteristicsCheck(Check):
    name = "fn-characteristics"
    description = "Step 3: Quantify FN/FP characteristics (volume/intensity/depth)"

    def run(
        self, instances: pd.DataFrame, quality: pd.DataFrame, args: argparse.Namespace
    ) -> List[ReportArtifact]:
        if instances.empty:
            logger.error("No data collected.")
            return []

        gt = instances[instances["role"] == "gt"].copy()
        fp = instances[
            (instances["role"] == "prediction") & (instances["classification"] == "false_positive")
        ].copy()

        # NOTE: "z_depth" here is centroid_y (the per-slice row coordinate),
        # not a literal Z slice index - preserved verbatim from the
        # pre-refactor step to keep output values unchanged.
        gt["z_depth"] = gt["centroid_y"]
        fp["z_depth"] = fp["centroid_y"]

        df = pd.concat([gt, fp], ignore_index=True, sort=False)

        logger.info("=== Characteristics Summary ===")
        summary = (
            df.groupby("classification")
            .agg({"volume": ["count", "median", "mean"], "mean_intensity": ["median", "mean"]})
            .round(2)
        )
        print(summary)

        confusion_rows = []
        for model_name, group in df.groupby("model"):
            counts = group["classification"].value_counts()
            tp = int(counts.get("true_positive", 0))
            fp_n = int(counts.get("false_positive", 0))
            fn = int(counts.get("blind_fn", 0) + counts.get("merged_fn", 0))
            precision = tp / (tp + fp_n) if (tp + fp_n) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
            confusion_rows.append(
                {
                    "model": model_name,
                    "tp": tp,
                    "fp": fp_n,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        confusion_df = pd.DataFrame(confusion_rows)
        print("\n--- Aligned object-level confusion (per model) ---")
        print(confusion_df.to_string(index=False))

        if df["model"].nunique() > 1:
            logger.info("Multiple models included in this run: %s", sorted(df["model"].unique()))

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        display_order = [c for c in CLASS_ORDER if c in df["classification"].unique()]
        display_labels = [CLASSIFICATION_LABELS[c] for c in display_order]

        sns.boxplot(
            data=df,
            x="classification",
            y="volume",
            order=display_order,
            hue="classification",
            palette=CLASS_PALETTE,
            legend=False,
            ax=axes[0],
        )
        axes[0].set_yscale("log")
        axes[0].set_title("Cell Volume Distribution (Log Scale)", fontsize=14)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Volume (voxels)")
        axes[0].set_xticks(range(len(display_order)))
        axes[0].set_xticklabels(display_labels, rotation=15, ha="right")

        sns.boxplot(
            data=df,
            x="classification",
            y="mean_intensity",
            order=display_order,
            hue="classification",
            palette=CLASS_PALETTE,
            legend=False,
            ax=axes[1],
        )
        axes[1].set_title("Cell Mean Intensity Distribution", fontsize=14)
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Mean intensity")
        axes[1].set_xticks(range(len(display_order)))
        axes[1].set_xticklabels(display_labels, rotation=15, ha="right")

        sns.kdeplot(
            data=df,
            x="z_depth",
            hue="classification",
            hue_order=display_order,
            fill=True,
            common_norm=False,
            palette=CLASS_PALETTE,
            ax=axes[2],
            alpha=0.5,
        )
        axes[2].set_title("Z-Depth Distribution", fontsize=14)
        axes[2].set_xlabel("Z depth")
        if axes[2].get_legend() is not None:
            for text, label in zip(axes[2].get_legend().get_texts(), display_labels):
                text.set_text(label)

        plt.tight_layout()

        return [
            ReportArtifact(name="step3_fn_characteristics", table=df, figure=fig),
            ReportArtifact(name="step3_fn_characteristics_confusion", table=confusion_df),
        ]
