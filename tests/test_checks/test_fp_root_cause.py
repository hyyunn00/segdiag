"""Tests for the fp-root-cause check and its pure classification function."""

from __future__ import annotations

from segdiag.checks.fp_root_cause import FpRootCauseCheck, classify_fp_subtype


def test_classify_fp_subtype_small_background_like_is_noise():
    assert (
        classify_fp_subtype(
            fp_volume=10,
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
            fp_volume=200,
            fp_intensity=80.0,
            nearest_gt_distance=5.0,
            background_mean=20.0,
            background_std=1.0,
        )
        == "boundary_split_fp"
    )


def test_classify_fp_subtype_large_far_and_bright_is_hallucination():
    assert (
        classify_fp_subtype(
            fp_volume=400,
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
            fp_volume=10,
            fp_intensity=None,
            nearest_gt_distance=None,
            background_mean=None,
            background_std=None,
        )
        == "hallucination_fp"
    )


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
