"""FP root-cause classification.

Splits false positives - previously just "an unmatched prediction" - into
three actionable buckets:

- ``noise_fp``: small and background-level intensity - probably a noise
  speckle, not a real detection problem.
- ``boundary_split_fp``: sits right next to a real GT cell - probably the
  same cell over-segmented into two pieces, not a hallucination.
- ``hallucination_fp``: volume/intensity look like a real cell, but there's
  no GT partner nearby at all - the most concerning category, worth
  investigating first.

:func:`classify_fp_subtype` is a pure, rule-based function with no CLI/report
dependencies, so :func:`segdiag.core.pipeline.collect` can call it directly
while scanning to populate ``InstanceRecord.fp_subtype`` - the check itself
only aggregates and plots the column collect() already filled in.
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from segdiag.checks.base import Check
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)

#: Below this volume (voxels), a false positive is a candidate for
#: "noise_fp" - small enough that it plausibly isn't a real structure.
SMALL_VOLUME_THRESHOLD = 30

#: How many background standard deviations a FP's mean intensity may sit
#: away from the local background mean and still count as "background-like"
#: (i.e. indistinguishable from noise) for the noise_fp rule.
NOISE_INTENSITY_TOLERANCE_STDS = 1.5

#: Below this centroid-to-nearest-GT distance (pixels), an unmatched
#: prediction is treated as sitting right next to a real cell - i.e. a
#: probable over-segmentation split of that cell rather than a fabricated
#: detection.
BOUNDARY_SPLIT_DISTANCE_THRESHOLD = 20.0

FP_SUBTYPE_ORDER = ["noise_fp", "boundary_split_fp", "hallucination_fp"]


def classify_fp_subtype(
    fp_volume: int,
    fp_intensity: Optional[float],
    nearest_gt_distance: Optional[float],
    background_mean: Optional[float],
    background_std: Optional[float],
) -> str:
    """Rule-based FP root-cause classification (see module docstring for the
    three buckets). Falls through to ``"hallucination_fp"`` whenever there
    isn't enough information (no raw image, no GT on the slice) to confirm
    the more specific noise/boundary-split explanations - the conservative
    choice, since hallucinations are the category most worth a human's
    attention anyway.
    """
    if (
        fp_volume < SMALL_VOLUME_THRESHOLD
        and fp_intensity is not None
        and background_mean is not None
        and background_std is not None
        and abs(fp_intensity - background_mean)
        <= NOISE_INTENSITY_TOLERANCE_STDS * max(background_std, 1e-6)
    ):
        return "noise_fp"

    if nearest_gt_distance is not None and nearest_gt_distance < BOUNDARY_SPLIT_DISTANCE_THRESHOLD:
        return "boundary_split_fp"

    return "hallucination_fp"


class FpRootCauseCheck(Check):
    name = "fp-root-cause"
    description = "FP root-cause breakdown (noise / boundary-split / hallucination)"

    def run(
        self, instances: pd.DataFrame, quality: pd.DataFrame, args: argparse.Namespace
    ) -> List[ReportArtifact]:
        fp = instances[
            (instances["role"] == "prediction") & (instances["classification"] == "false_positive")
        ].copy()
        if fp.empty:
            logger.warning("No false positives in the collected data - nothing to report.")
            return []

        summary = (
            fp.groupby("fp_subtype")
            .agg(
                count=("volume", "count"),
                median_volume=("volume", "median"),
                mean_volume=("volume", "mean"),
                median_intensity=("mean_intensity", "median"),
                mean_intensity=("mean_intensity", "mean"),
            )
            .reset_index()
        )

        crosstab = pd.crosstab([fp["sample"], fp["model"]], fp["fp_subtype"]).reset_index()

        order = [s for s in FP_SUBTYPE_ORDER if s in fp["fp_subtype"].unique()]

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.boxplot(
            data=fp,
            x="fp_subtype",
            y="volume",
            order=order,
            hue="fp_subtype",
            legend=False,
            ax=axes[0],
        )
        axes[0].set_yscale("log")
        axes[0].set_title("FP Volume by Root-Cause Subtype")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Volume (voxels)")

        sns.countplot(
            data=fp, x="fp_subtype", order=order, hue="fp_subtype", legend=False, ax=axes[1]
        )
        axes[1].set_title("FP Count by Root-Cause Subtype")
        axes[1].set_xlabel("")

        plt.tight_layout()

        return [
            ReportArtifact(name="fp_root_cause_summary", table=summary, figure=fig),
            ReportArtifact(name="fp_root_cause_by_sample_model", table=crosstab),
        ]
