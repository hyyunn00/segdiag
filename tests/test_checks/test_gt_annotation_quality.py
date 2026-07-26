"""Tests for the gt-annotation-quality check."""

from __future__ import annotations

from segdiag.checks.gt_annotation_quality import GtAnnotationQualityCheck


def test_gt_annotation_quality_flags_the_planted_volume_outlier(collected_dataset):
    instances_df, quality_df, _, args = collected_dataset

    artifacts = GtAnnotationQualityCheck().run(instances_df, quality_df, args)
    by_name = {a.name: a for a in artifacts}

    outliers = by_name["gt_annotation_quality_outliers"].table
    # The fixture plants exactly one much-larger-than-typical GT cell
    # (volume 900 vs. the rest in the 64-529 range).
    assert (outliers["volume"] == 900).any()
    assert (outliers.loc[outliers["volume"] == 900, "volume_outlier"] == "too_large").all()
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
