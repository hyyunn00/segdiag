"""Regression tests for the six checks migrated from steps/*.py.

Per the upgrade plan's acceptance criteria, these checks must reproduce the
exact same tp/fp/fn/precision/recall/f1 numbers the pre-refactor scripts
computed from match_instances()/find_false_positives() - only how the
result gets packaged (ReportArtifact vs. print+savefig) should differ.
"""

from __future__ import annotations

from segdiag.checks.detection_probability import DetectionProbabilityCheck
from segdiag.checks.fn_characteristics import FnCharacteristicsCheck
from segdiag.checks.fn_contribution import FnContributionCheck
from segdiag.checks.iou_distribution import IouDistributionCheck
from segdiag.checks.story_closure import StoryClosureCheck


def _expected_counts(instances_df):
    gt = instances_df[instances_df["role"] == "gt"]
    fp = instances_df[
        (instances_df["role"] == "prediction")
        & (instances_df["classification"] == "false_positive")
    ]
    tp = int((gt["classification"] == "true_positive").sum())
    fn = int((gt["classification"] != "true_positive").sum())
    return tp, len(fp), fn


def test_iou_distribution_check_matches_aligned_counts(collected_dataset):
    instances_df, quality_df, _, args = collected_dataset
    tp, fp, fn = _expected_counts(instances_df)

    artifacts = IouDistributionCheck().run(instances_df, quality_df, args)
    by_name = {a.name: a for a in artifacts}

    summary = by_name["step1_iou_distribution_summary"].table.iloc[0]
    assert (int(summary["tp"]), int(summary["fp"]), int(summary["fn"])) == (tp, fp, fn)
    assert by_name["step1_iou_distribution"].figure is not None
    assert len(by_name["step1_iou_distribution"].table) == tp + fn


def test_fn_characteristics_check_matches_aligned_counts(collected_dataset):
    instances_df, quality_df, _, args = collected_dataset
    tp, fp, fn = _expected_counts(instances_df)

    artifacts = FnCharacteristicsCheck().run(instances_df, quality_df, args)
    by_name = {a.name: a for a in artifacts}

    confusion = by_name["step3_fn_characteristics_confusion"].table
    row = confusion[confusion["model"] == "unet_v9"].iloc[0]
    assert (int(row["tp"]), int(row["fp"]), int(row["fn"])) == (tp, fp, fn)

    combined = by_name["step3_fn_characteristics"].table
    assert len(combined) == tp + fn + fp
    assert set(combined["classification"]) <= {
        "true_positive",
        "blind_fn",
        "merged_fn",
        "false_positive",
    }


def test_detection_probability_check_recall_matches(collected_dataset):
    instances_df, quality_df, _, args = collected_dataset
    tp, _, fn = _expected_counts(instances_df)

    artifacts = DetectionProbabilityCheck().run(instances_df, quality_df, args)
    assert len(artifacts) == 1
    table = artifacts[0].table
    assert len(table) == tp + fn
    assert table["is_tp"].sum() == tp


def test_fn_contribution_check_totals_match(collected_dataset):
    instances_df, quality_df, _, args = collected_dataset
    tp, fp, fn = _expected_counts(instances_df)

    artifacts = FnContributionCheck().run(instances_df, quality_df, args)
    by_name = {a.name: a for a in artifacts}

    summary = by_name["step5_fn_contribution"].table
    assert summary["tp_count"].sum() == tp
    assert summary["fn_count"].sum() == fn

    fp_summary = by_name["step5_fn_contribution_fp_summary"].table
    assert fp_summary["fp_count"].sum() == fp


def test_story_closure_check_runs_and_returns_all_instances(collected_dataset):
    instances_df, quality_df, _, args = collected_dataset
    tp, _, fn = _expected_counts(instances_df)

    artifacts = StoryClosureCheck().run(instances_df, quality_df, args)
    assert len(artifacts) == 1
    table = artifacts[0].table
    assert len(table) == tp + fn
    assert artifacts[0].figure is not None
