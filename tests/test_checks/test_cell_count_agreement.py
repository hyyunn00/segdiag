"""Tests for the cell-count-agreement check
(SEGDIAG_MARS_ALIGNMENT_COMPLETE.md Part 3).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import tifffile

from segdiag.checks.cell_count_agreement import CellCountAgreementCheck, _safe_f1
from segdiag.core.pipeline import collect


def test_safe_f1_is_zero_not_nan_when_precision_and_recall_are_both_zero():
    assert _safe_f1(0.0, 0.0) == 0.0


def test_safe_f1_is_nan_when_undefined():
    assert pd.isna(_safe_f1(float("nan"), 0.5))


def test_safe_f1_normal_case():
    assert _safe_f1(1.0, 1.0) == 1.0
    assert _safe_f1(0.5, 0.5) == 0.5


def test_cell_count_agreement_check_returns_empty_when_no_instances():
    empty = pd.DataFrame(columns=["role"])
    args = argparse.Namespace()

    assert CellCountAgreementCheck().run(empty, empty, args) == []


def _write_perfectly_matched_dataset(root) -> None:
    """3 GT cells, all exactly matched by predictions - the "everything
    agrees" baseline: gt_count == pr_count and located_count_f1 == 1.0.
    """
    shape = (100, 100)
    sample_dir = root / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    gt_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)

    gt = np.zeros(shape, dtype=np.uint8)
    pr = np.zeros(shape, dtype=np.uint8)
    for i in range(3):
        r, c = 10, 10 + i * 30
        gt[r : r + 10, c : c + 10] = 1
        pr[r : r + 10, c : c + 10] = 1

    tifffile.imwrite(gt_dir / "slice_0000.tif", gt)
    tifffile.imwrite(pred_dir / "slice_0000.tif", pr)


def _write_disjoint_dataset(root) -> None:
    """3 GT cells and 3 predictions of the same count - so a naive
    ``min(gt_count, pr_count)``-style metric would score this ~1.0 - but
    placed at completely disjoint locations with zero overlap, so none of
    them "really" agree. This is the exact scenario
    SEGDIAG_MARS_ALIGNMENT_COMPLETE.md Part 3.0 argues against: a model
    that hallucinates predictions in empty background must not score well
    just because the totals happen to match.
    """
    shape = (100, 100)
    sample_dir = root / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    gt_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)

    gt = np.zeros(shape, dtype=np.uint8)
    pr = np.zeros(shape, dtype=np.uint8)
    for i in range(3):
        r, c = 10, 10 + i * 30
        gt[r : r + 10, c : c + 10] = 1
        pr[r + 50 : r + 60, c : c + 10] = 1  # same count, disjoint location

    tifffile.imwrite(gt_dir / "slice_0000.tif", gt)
    tifffile.imwrite(pred_dir / "slice_0000.tif", pr)


def _run_check(tmp_path, dataset_writer):
    dataset_writer(tmp_path)
    instances_df, quality_df = collect(tmp_path)
    args = argparse.Namespace(root=tmp_path)
    artifacts = CellCountAgreementCheck().run(instances_df, quality_df, args)
    return {a.name: a for a in artifacts}


def test_cell_count_agreement_check_returns_all_four_artifacts(tmp_path):
    by_name = _run_check(tmp_path, _write_perfectly_matched_dataset)

    assert set(by_name) == {
        "cell_count_agreement_summary",
        "cell_count_agreement_per_sample",
        "cell_count_agreement_scatter",
        "cell_count_agreement_bland_altman",
    }
    assert by_name["cell_count_agreement_scatter"].figure is not None
    assert by_name["cell_count_agreement_bland_altman"].figure is not None


def test_located_count_f1_is_one_when_everything_matches(tmp_path):
    by_name = _run_check(tmp_path, _write_perfectly_matched_dataset)

    per_sample = by_name["cell_count_agreement_per_sample"].table
    row = per_sample.iloc[0]
    assert row["gt_count"] == 3
    assert row["pr_count"] == 3
    assert row["located_count_f1"] == 1.0

    summary = by_name["cell_count_agreement_summary"].table
    assert summary.iloc[0]["located_count_f1_macro"] == 1.0


def test_located_count_f1_is_zero_when_counts_match_but_locations_dont(tmp_path):
    """The key Part 3 regression case: gt_count == pr_count (a naive
    total-count metric would score this ~1.0), but located_count_f1 must be
    ~0 since none of the predictions actually overlap any GT cell.
    """
    by_name = _run_check(tmp_path, _write_disjoint_dataset)

    per_sample = by_name["cell_count_agreement_per_sample"].table
    row = per_sample.iloc[0]
    assert row["gt_count"] == row["pr_count"] == 3  # naive count agreement
    assert row["located_tp"] == 0
    assert row["located_count_f1"] == 0.0  # but positionally, none of it is real

    summary = by_name["cell_count_agreement_summary"].table
    assert summary.iloc[0]["located_count_f1_macro"] == 0.0
