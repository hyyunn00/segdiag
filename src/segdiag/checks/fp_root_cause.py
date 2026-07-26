"""FP root-cause classification.

Splits false positives - previously just "an unmatched prediction" - into
three actionable buckets:

- ``noise_fp``: intensity is statistically indistinguishable from the
  *local* background around it - probably a noise speckle, not a real
  detection problem.
- ``boundary_split_fp``: sits right next to a real GT cell - probably the
  same cell over-segmented into two pieces, not a hallucination.
- ``hallucination_fp``: has real local contrast against its background, but
  there's no GT partner nearby at all - the most concerning category, worth
  investigating first.

Volume is deliberately *not* one of the signals here: a false positive's
size doesn't actually tell you why it's spurious (both a large noise blob
and a small genuine hallucination exist), whereas local contrast and
distance-to-nearest-GT are the two properties that directly correspond to
the two real explanations ("is it just noise" / "is it a fragment of a real
cell").

:func:`classify_fp_subtype` is a pure, rule-based function with no CLI/report
dependencies, so :func:`segdiag.core.pipeline.collect` can call it directly
while scanning to populate ``InstanceRecord.fp_subtype``, using
:func:`compute_local_background_stats` (also here, since it feeds directly
into that rule) to estimate the background around each FP - the check
itself only aggregates and plots the column collect() already filled in.
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from segdiag.checks.base import Check
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)

#: How many pixels of context around a FP's bbox to sample when estimating
#: its *local* background - large enough to gather a stable mean/std, small
#: enough that a genuinely local neighbourhood doesn't drift into an
#: unrelated part of the slice.
LOCAL_BACKGROUND_PAD = 20

#: How many local-background standard deviations a FP's mean intensity may
#: sit away from the local background mean and still count as
#: "background-like" (i.e. indistinguishable from noise) for the noise_fp
#: rule.
NOISE_CONTRAST_STD_THRESHOLD = 1.5

#: Below this centroid-to-nearest-GT distance (pixels), an unmatched
#: prediction is treated as sitting right next to a real cell - i.e. a
#: probable over-segmentation split of that cell rather than a fabricated
#: detection.
BOUNDARY_SPLIT_DISTANCE_THRESHOLD = 20.0

FP_SUBTYPE_ORDER = ["noise_fp", "boundary_split_fp", "hallucination_fp"]


def compute_local_background_stats(
    raw_arr: np.ndarray,
    gt_arr: np.ndarray,
    pr_arr: np.ndarray,
    bbox: Tuple[int, ...],
    pad: int = LOCAL_BACKGROUND_PAD,
) -> Tuple[Optional[float], Optional[float]]:
    """Mean/std raw intensity of the true-background voxels in a
    ``pad``-voxel neighbourhood around a false positive's ``bbox``.

    Dimension-agnostic: ``bbox`` is ``(*mins, *maxs)`` (length ``2 * ndim``,
    e.g. ``(min_row, min_col, max_row, max_col)`` for a 2D array or
    ``(min_z, min_row, min_col, max_z, max_row, max_col)`` for a 3D volume),
    matching whatever ``raw_arr.ndim`` the caller passes in.

    "True background" here means voxels that are part of neither *any* GT
    cell nor *any* predicted instance in that neighbourhood - so neighbouring
    real cells and the FP itself never contaminate the estimate. Returns
    ``(None, None)`` if the padded neighbourhood happens to have no such
    voxels at all (e.g. an unusually crowded field).
    """
    ndim = raw_arr.ndim
    mins, maxs = bbox[:ndim], bbox[ndim:]
    window = tuple(
        slice(max(0, lo - pad), min(size, hi + pad))
        for lo, hi, size in zip(mins, maxs, raw_arr.shape)
    )

    bg_mask = (gt_arr[window] == 0) & (pr_arr[window] == 0)
    if not bg_mask.any():
        return None, None

    bg_values = raw_arr[window].astype(np.float64)[bg_mask]
    return float(bg_values.mean()), float(bg_values.std())


def classify_fp_subtype(
    fp_intensity: Optional[float],
    nearest_gt_distance: Optional[float],
    background_mean: Optional[float],
    background_std: Optional[float],
) -> str:
    """Rule-based FP root-cause classification (see module docstring for the
    three buckets). ``background_mean``/``background_std`` should come from
    :func:`compute_local_background_stats` (a neighbourhood around this
    specific FP), not a whole-slice average. Falls through to
    ``"hallucination_fp"`` whenever there isn't enough information (no raw
    image, no local background, no GT on the slice) to confirm the more
    specific noise/boundary-split explanations - the conservative choice,
    since hallucinations are the category most worth a human's attention
    anyway.
    """
    if (
        fp_intensity is not None
        and background_mean is not None
        and background_std is not None
        and abs(fp_intensity - background_mean)
        <= NOISE_CONTRAST_STD_THRESHOLD * max(background_std, 1e-6)
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
