"""Tests for the gt-annotation-quality check."""

from __future__ import annotations

import argparse

import numpy as np
import tifffile

from segdiag.checks.gt_annotation_quality import GtAnnotationQualityCheck
from segdiag.core.pipeline import collect


def test_gt_annotation_quality_flags_a_genuine_log_volume_outlier(tmp_path):
    # A dedicated dataset (not the shared collected_dataset fixture) since
    # the MAD-on-log-volume rule needs a population where the outlier is
    # genuinely far in *log* space - the shared fixture's tightly-packed
    # grid of same-size-bucket cells doesn't leave room to place one without
    # it overlapping its neighbours.
    shape = (200, 200)
    sample_dir = tmp_path / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    gt_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)

    gt = np.zeros(shape, dtype=np.uint8)
    pr = np.zeros(shape, dtype=np.uint8)

    # A cluster of typically-sized cells (volumes 64..400)...
    for i, size in enumerate((8, 10, 12, 14, 16, 18, 20)):
        r, c = 10, 10 + i * 30
        gt[r : r + size, c : c + size] = 1
        pr[r : r + size, c : c + size] = 1

    # ...and one dramatically larger "two cells merged into one label"
    # outlier, isolated from the cluster above and matched exactly (so it
    # isn't also flagged as a miss).
    gt[100:160, 100:160] = 1  # volume 3600
    pr[100:160, 100:160] = 1

    tifffile.imwrite(gt_dir / "slice_0000.tif", gt)
    tifffile.imwrite(pred_dir / "slice_0000.tif", pr)

    instances_df, quality_df = collect(tmp_path)
    args = argparse.Namespace(
        root=tmp_path,
        sample=None,
        model=None,
        output_dir=tmp_path / "out",
        mask_name=None,
        raw_name=None,
        dark_name="Flatten_561_dark",
        num_samples=5,
    )

    artifacts = GtAnnotationQualityCheck().run(instances_df, quality_df, args)
    by_name = {a.name: a for a in artifacts}

    outliers = by_name["gt_annotation_quality_outliers"].table
    assert (outliers["volume"] == 3600).any()
    assert (outliers.loc[outliers["volume"] == 3600, "volume_outlier"] == "too_large").all()
    assert by_name["gt_annotation_quality_outliers"].figure is not None


def test_gt_annotation_quality_flags_the_planted_border_cell(collected_dataset):
    instances_df, quality_df, _, args = collected_dataset

    artifacts = GtAnnotationQualityCheck().run(instances_df, quality_df, args)
    by_name = {a.name: a for a in artifacts}

    border = by_name["gt_annotation_quality_border"].table
    row = border[border["sample"] == "case01"].iloc[0]
    assert row["near_border_pct"] > 0.0
    # quality_df's fully shape-aware count should agree it's non-zero too.
    assert row["gt_touching_border_pct"] > 0.0


def test_gt_annotation_quality_flags_the_planted_density_anomaly(collected_dataset):
    instances_df, quality_df, _, args = collected_dataset

    artifacts = GtAnnotationQualityCheck().run(instances_df, quality_df, args)
    by_name = {a.name: a for a in artifacts}

    anomalies = by_name["gt_annotation_quality_density_anomalies"].table
    # The fixture makes z_index 2 far denser (20 cells) than the other
    # slices (6-9 cells).
    assert 2 in set(anomalies["z_index"])


def test_gt_annotation_quality_returns_empty_when_no_gt(collected_dataset):
    import pandas as pd

    _, quality_df, _, args = collected_dataset
    empty_instances = pd.DataFrame(columns=["role", "volume", "sample"])
    assert GtAnnotationQualityCheck().run(empty_instances, quality_df, args) == []
