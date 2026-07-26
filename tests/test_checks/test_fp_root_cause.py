"""Tests for the fp-root-cause check and its pure classification function."""

from __future__ import annotations

import numpy as np

from segdiag.checks.fp_root_cause import (
    FpRootCauseCheck,
    classify_fp_subtype,
    compute_local_background_stats,
)


def test_classify_fp_subtype_background_like_is_noise_regardless_of_volume():
    # classify_fp_subtype no longer takes a volume argument at all - a
    # *large* FP with background-level local contrast is still noise_fp,
    # not automatically promoted to hallucination_fp just because it's big.
    assert (
        classify_fp_subtype(
            fp_intensity=20.2,
            nearest_gt_distance=100.0,
            background_mean=20.0,
            background_std=1.0,
        )
        == "noise_fp"
    )


def test_classify_fp_subtype_near_a_gt_cell_is_boundary_split():
    assert (
        classify_fp_subtype(
            fp_intensity=80.0,
            nearest_gt_distance=5.0,
            background_mean=20.0,
            background_std=1.0,
        )
        == "boundary_split_fp"
    )


def test_classify_fp_subtype_far_and_bright_is_hallucination():
    assert (
        classify_fp_subtype(
            fp_intensity=90.0,
            nearest_gt_distance=100.0,
            background_mean=20.0,
            background_std=1.0,
        )
        == "hallucination_fp"
    )


def test_classify_fp_subtype_falls_back_to_hallucination_without_raw_data():
    # No background stats and no GT on the slice - the conservative default.
    assert (
        classify_fp_subtype(
            fp_intensity=None,
            nearest_gt_distance=None,
            background_mean=None,
            background_std=None,
        )
        == "hallucination_fp"
    )


def test_compute_local_background_stats_excludes_gt_and_prediction_pixels():
    shape = (60, 60)
    raw = np.full(shape, 20.0, dtype=np.float32)
    gt = np.zeros(shape, dtype=np.uint8)
    pr = np.zeros(shape, dtype=np.uint8)

    # A real GT cell just outside the FP's padded neighbourhood window...
    gt[5:10, 5:10] = 1
    raw[5:10, 5:10] = 200.0  # would badly skew the mean if it leaked in

    # ...and the FP itself, whose own pixels must also be excluded.
    fp_bbox = (30, 30, 40, 40)
    pr[30:40, 30:40] = 1
    raw[30:40, 30:40] = 90.0

    mean, std = compute_local_background_stats(raw, gt, pr, fp_bbox, pad=20)

    assert mean == 20.0
    assert std == 0.0


def test_compute_local_background_stats_returns_none_when_no_background_pixels():
    shape = (20, 20)
    raw = np.full(shape, 20.0, dtype=np.float32)
    gt = np.ones(shape, dtype=np.uint8)  # entire crop is GT - no background at all
    pr = np.zeros(shape, dtype=np.uint8)

    mean, std = compute_local_background_stats(raw, gt, pr, (5, 5, 10, 10), pad=20)

    assert mean is None
    assert std is None


def test_fp_root_cause_check_matches_planted_subtypes(collected_dataset):
    instances_df, quality_df, _, args = collected_dataset

    artifacts = FpRootCauseCheck().run(instances_df, quality_df, args)
    by_name = {a.name: a for a in artifacts}

    summary = by_name["fp_root_cause_summary"].table
    assert set(summary["fp_subtype"]) == {"noise_fp", "boundary_split_fp", "hallucination_fp"}
    assert (summary["count"] == 1).all()  # the fixture plants exactly one of each

    crosstab = by_name["fp_root_cause_by_sample_model"].table
    assert (crosstab["sample"] == "case01").all()
    assert (crosstab["model"] == "unet_v9").all()


def test_fp_root_cause_check_returns_empty_when_no_false_positives(collected_dataset):
    instances_df, quality_df, _, args = collected_dataset
    no_fp = instances_df[instances_df["role"] == "gt"]

    assert FpRootCauseCheck().run(no_fp, quality_df, args) == []
