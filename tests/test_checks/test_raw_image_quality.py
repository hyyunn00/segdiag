"""Tests for the raw-image-quality check and its pure metric function."""

from __future__ import annotations

import numpy as np

from segdiag.checks.raw_image_quality import RawImageQualityCheck, compute_image_quality_metrics


def test_compute_image_quality_metrics_flat_image_is_fully_saturated():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:10, 5:10] = 1
    raw = np.full((20, 20), 42.0, dtype=np.float32)

    metrics = compute_image_quality_metrics(gt, raw)

    assert metrics["saturation_pct"] == 100.0
    assert metrics["mean_intensity"] == 42.0
    assert metrics["std_intensity"] == 0.0
    assert metrics["gt_instance_count"] == 1
    assert metrics["gt_touching_border_pct"] == 0.0


def test_compute_image_quality_metrics_flags_border_touching_instance():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[0:5, 0:5] = 1  # touches row 0 and col 0
    raw = np.full((20, 20), 10.0, dtype=np.float32)

    metrics = compute_image_quality_metrics(gt, raw)

    assert metrics["gt_instance_count"] == 1
    assert metrics["gt_touching_border_pct"] == 100.0


def test_compute_image_quality_metrics_snr_reflects_foreground_background_gap():
    gt = np.zeros((40, 40), dtype=np.uint8)
    gt[10:20, 10:20] = 1
    raw = np.random.default_rng(0).normal(10.0, 1.0, (40, 40)).astype(np.float32)
    raw[10:20, 10:20] = 100.0  # bright foreground vs noisy background

    metrics = compute_image_quality_metrics(gt, raw)

    assert metrics["snr_estimate"] > 10  # foreground mean way above background std


def test_raw_image_quality_check_returns_empty_when_no_quality_data(collected_dataset):
    import pandas as pd

    instances_df, _, _, args = collected_dataset
    artifacts = RawImageQualityCheck().run(instances_df, pd.DataFrame(), args)
    assert artifacts == []


def test_raw_image_quality_check_produces_per_sample_and_per_slice_tables(collected_dataset):
    instances_df, quality_df, _, args = collected_dataset

    artifacts = RawImageQualityCheck().run(instances_df, quality_df, args)
    by_name = {a.name: a for a in artifacts}

    per_sample = by_name["raw_image_quality_per_sample"].table
    assert set(per_sample["sample"]) == {"case01"}
    assert "recall" in per_sample.columns
    assert "fp_rate" in per_sample.columns
    assert by_name["raw_image_quality_per_sample"].figure is not None

    per_slice = by_name["raw_image_quality_per_slice"].table
    assert len(per_slice) == len(quality_df)
