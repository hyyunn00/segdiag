"""Step 1: GT Best-IoU Distribution Analysis.

Diagnoses low instance recall at its root: for every ground-truth cell,
its best IoU against any predicted instance, then plots the distribution. A
large mass of cells sitting near IoU = 0 points at cells the model never
detects at all, as opposed to cells it detects but under-segments.

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

from segdiag.checks.base import Check
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)


class IouDistributionCheck(Check):
    name = "iou-distribution"
    description = "Step 1: Best-IoU distribution histogram (why cells are missed)"

    def run(
        self, instances: pd.DataFrame, quality: pd.DataFrame, args: argparse.Namespace
    ) -> List[ReportArtifact]:
        gt = instances[instances["role"] == "gt"]
        if gt.empty:
            logger.warning("No GT instances in the collected data - nothing to plot.")
            return []

        tp_ious = gt.loc[gt["classification"] == "true_positive", "best_iou"].tolist()
        fn_ious = gt.loc[gt["classification"].isin(["blind_fn", "merged_fn"]), "best_iou"].tolist()

        total_tp = len(tp_ious)
        total_fn = len(fn_ious)
        total_fp = int(
            (
                (instances["role"] == "prediction")
                & (instances["classification"] == "false_positive")
            ).sum()
        )
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        logger.info("Models included in this run: %s", sorted(gt["model"].unique()))
        logger.info(
            "Aligned object-level counts: tp=%d fp=%d fn=%d | precision=%.3f recall=%.3f f1=%.3f",
            total_tp,
            total_fp,
            total_fn,
            precision,
            recall,
            f1,
        )

        fig = plt.figure(figsize=(10, 6))
        bins = np.linspace(0, 1, 21)  # 0.05 per bin
        plt.hist(
            [fn_ious, tp_ious],
            bins=bins,  # type: ignore[arg-type]
            stacked=True,
            color=["#ff9999", "#66b3ff"],
            label=["False Negative (IoU < 0.5)", "True Positive (IoU >= 0.5)"],
        )
        plt.axvline(x=0.5, color="r", linestyle="--", linewidth=2, label="IoU = 0.5 Threshold")
        plt.title(
            "Best IoU Distribution of Ground Truth Cells\n(Why are we missing cells?)", fontsize=16
        )
        plt.xlabel("Best IoU Score", fontsize=14)
        plt.ylabel("Number of GT Cells", fontsize=14)
        plt.xticks(np.arange(0, 1.05, 0.1))
        plt.legend(fontsize=12)
        plt.grid(axis="y", alpha=0.3)

        detail_table = gt[
            [
                "dataset",
                "sample",
                "model",
                "slice_name",
                "z_index",
                "instance_id",
                "volume",
                "best_iou",
                "classification",
            ]
        ].reset_index(drop=True)
        summary_table = pd.DataFrame(
            [
                {
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            ]
        )

        return [
            ReportArtifact(
                name="step1_iou_distribution",
                table=detail_table,
                figure=fig,
                metadata={"summary": summary_table.to_dict("records")[0]},
            ),
            ReportArtifact(name="step1_iou_distribution_summary", table=summary_table),
        ]
