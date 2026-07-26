"""Tests for segdiag.core.pipeline.collect().

The critical property this module must preserve is: the per-instance rows
collect() produces must carry exactly the same volume/mean_intensity/
best_iou/classification values that directly calling
match_instances()/find_false_positives() on the same arrays would - those
two functions are the single source of truth every check (old or new)
ultimately derives its numbers from. The first test group below
cross-validates segdiag.core.pipeline._slice_instance_rows() directly against
those primitives on identical synthetic arrays. The second group exercises
collect()'s folder scanning, filtering, and parquet caching end to end on a
small synthetic dataset tree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from segdiag.core.matching import find_false_positives, match_instances
from segdiag.core.pipeline import _slice_instance_rows, collect
from segdiag.core.schema import ImageQualityRecord, InstanceRecord


def _make_square_mask(shape, top_left, size, value=1):
    arr = np.zeros(shape, dtype=np.uint8)
    r, c = top_left
    arr[r : r + size, c : c + size] = value
    return arr


def _synthetic_slice(shape=(50, 80)):
    """One slice with a TP, a blind FN, and a hallucination FP - exercises
    every classification bucket in one array pair.
    """
    gt = np.zeros(shape, dtype=np.uint8)
    gt[10:20, 10:20] = 1  # matched by pred below -> true_positive, area 100
    gt[30:38, 30:38] = 1  # untouched -> blind_fn, area 64

    pr = np.zeros(shape, dtype=np.uint8)
    pr[10:20, 10:20] = 1  # matches the first GT square
    pr[5:11, 60:66] = 1  # no GT partner -> false_positive, area 36

    raw = np.full(shape, 42.0, dtype=np.float32)
    return gt, pr, raw


def _synthetic_merged_fn_slice(shape=(50, 80)):
    """One slice with a single partial-overlap (merged FN) pair."""
    gt = _make_square_mask(shape, (10, 10), 10)  # area 100
    pr = _make_square_mask(shape, (15, 15), 10)  # shifted, partial overlap
    raw = np.full(shape, 77.0, dtype=np.float32)
    return gt, pr, raw


# --- cross-validation against the matching primitives directly -------------


def test_slice_instance_rows_matches_match_instances_directly():
    gt, pr, raw = _synthetic_slice()

    expected = match_instances(gt, pr, raw_arr=raw)
    expected_tuples = sorted(
        (m.classify(), m.volume, m.mean_intensity, m.best_iou) for m in expected
    )

    actual = _slice_instance_rows(
        gt, pr, raw, dataset="ds", sample="case01", model="unet_v9", slice_name="s.tif", z_index=0
    )
    gt_rows = [r for r in actual if r["role"] == "gt"]
    actual_tuples = sorted(
        (r["classification"], r["volume"], r["mean_intensity"], r["best_iou"]) for r in gt_rows
    )

    assert actual_tuples == expected_tuples


def test_slice_instance_rows_matches_find_false_positives_directly():
    gt, pr, raw = _synthetic_slice()

    expected = find_false_positives(gt, pr, raw_arr=raw)
    expected_volumes = sorted((fp.volume, fp.mean_intensity) for fp in expected)

    actual = _slice_instance_rows(
        gt, pr, raw, dataset="ds", sample="case01", model="unet_v9", slice_name="s.tif", z_index=0
    )
    fp_rows = [r for r in actual if r["role"] == "prediction"]
    actual_volumes = sorted((r["volume"], r["mean_intensity"]) for r in fp_rows)

    assert actual_volumes == expected_volumes
    assert all(r["classification"] == "false_positive" for r in fp_rows)


def test_slice_instance_rows_records_matched_instance_id_pairing():
    gt, pr, raw = _synthetic_slice()

    rows = _slice_instance_rows(
        gt, pr, raw, dataset="ds", sample="case01", model="unet_v9", slice_name="s.tif", z_index=0
    )
    tp_rows = [r for r in rows if r["classification"] == "true_positive"]
    blind_fn_rows = [r for r in rows if r["classification"] == "blind_fn"]

    assert len(tp_rows) == 1
    assert tp_rows[0]["matched_instance_id"] is not None
    assert len(blind_fn_rows) == 1
    assert blind_fn_rows[0]["matched_instance_id"] is None


def test_slice_instance_rows_merged_fn_has_no_matched_instance_id():
    """The partially-overlapping prediction fails to claim the GT (best IoU
    is between the blind and TP thresholds), so under one-to-one matching
    it is *also* unmatched and counted as its own false positive - both
    rows carry no matched_instance_id.
    """
    gt, pr, raw = _synthetic_merged_fn_slice()

    rows = _slice_instance_rows(
        gt, pr, raw, dataset="ds", sample="case01", model="unet_v9", slice_name="s.tif", z_index=0
    )
    assert len(rows) == 2
    by_role = {r["role"]: r for r in rows}
    assert by_role["gt"]["classification"] == "merged_fn"
    assert by_role["gt"]["matched_instance_id"] is None
    assert by_role["prediction"]["classification"] == "false_positive"
    assert by_role["prediction"]["matched_instance_id"] is None


def test_slice_instance_rows_columns_match_instance_record_schema():
    gt, pr, raw = _synthetic_slice()
    rows = _slice_instance_rows(
        gt, pr, raw, dataset="ds", sample="case01", model="unet_v9", slice_name="s.tif", z_index=0
    )
    expected_columns = {f for f in InstanceRecord.__dataclass_fields__}
    assert set(rows[0].keys()) == expected_columns


# --- end-to-end collect() over a synthetic dataset tree ---------------------


def _write_dataset(root) -> None:
    """Build a minimal analysis.py-style dataset tree with two samples."""
    shape = (50, 80)

    for sample, raw_value in (("case01", 42.0), ("case02", 99.0)):
        sample_dir = root / sample
        gt_dir = sample_dir / "Flatten_561_mask"
        raw_dir = sample_dir / "Flatten_561"
        pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
        for d in (gt_dir, raw_dir, pred_dir):
            d.mkdir(parents=True)

        gt0, pr0, _ = _synthetic_slice(shape)
        gt1, pr1, _ = _synthetic_merged_fn_slice(shape)
        raw0 = np.full(shape, raw_value, dtype=np.float32)
        raw1 = np.full(shape, raw_value + 1, dtype=np.float32)

        tifffile.imwrite(gt_dir / "slice_0000.tif", gt0)
        tifffile.imwrite(gt_dir / "slice_0001.tif", gt1)
        tifffile.imwrite(pred_dir / "slice_0000.tif", pr0)
        tifffile.imwrite(pred_dir / "slice_0001.tif", pr1)
        tifffile.imwrite(raw_dir / "slice_0000.tif", raw0)
        tifffile.imwrite(raw_dir / "slice_0001.tif", raw1)


def test_collect_returns_expected_shape_and_columns(tmp_path):
    _write_dataset(tmp_path)

    instances_df, quality_df = collect(tmp_path)

    assert list(instances_df.columns) == [f for f in InstanceRecord.__dataclass_fields__]
    assert list(quality_df.columns) == [f for f in ImageQualityRecord.__dataclass_fields__]
    # Every synthetic slice has a raw image, so quality rows are populated
    # (one per GT slice, independent of how many models are evaluated).
    assert len(quality_df) == 4  # 2 samples x 2 slices each
    assert set(quality_df["source"]) == {"raw"}


def test_collect_finds_both_samples_and_all_classifications(tmp_path):
    _write_dataset(tmp_path)

    instances_df, _ = collect(tmp_path)

    assert set(instances_df["sample"]) == {"case01", "case02"}
    assert set(instances_df["dataset"]) == {tmp_path.name}
    assert set(instances_df["model"]) == {"unet_v9"}
    assert set(instances_df["classification"]) == {
        "true_positive",
        "blind_fn",
        "false_positive",
        "merged_fn",
    }


def test_collect_matches_direct_match_instances_totals(tmp_path):
    _write_dataset(tmp_path)

    instances_df, _ = collect(tmp_path)
    case01 = instances_df[instances_df["sample"] == "case01"]

    gt_rows = case01[case01["role"] == "gt"]
    fp_rows = case01[case01["role"] == "prediction"]

    # Slice 0: 1 TP + 1 blind FN + 1 hallucination FP (no GT partner).
    # Slice 1: 1 merged FN whose partially-overlapping prediction also
    # fails to claim it, so it counts as a second FP.
    assert (gt_rows["classification"] == "true_positive").sum() == 1
    assert (gt_rows["classification"] == "blind_fn").sum() == 1
    assert (gt_rows["classification"] == "merged_fn").sum() == 1
    assert len(fp_rows) == 2
    assert sorted(fp_rows["volume"]) == [36, 100]


def test_collect_sample_filter_restricts_scan(tmp_path):
    _write_dataset(tmp_path)

    instances_df, _ = collect(tmp_path, sample_filter="case01")

    assert set(instances_df["sample"]) == {"case01"}


def test_collect_model_filter_restricts_scan(tmp_path):
    _write_dataset(tmp_path)

    instances_df, _ = collect(tmp_path, model_filter="does-not-exist")

    assert instances_df.empty


def _write_dataset_with_sibling_models(root) -> None:
    """Two samples, each with a "v12_unet_baseline" model *and* a sibling
    "v12_unet_baseline_dark" model whose name contains the first as a
    substring - the exact scenario --model-exact needs to tell apart.
    """
    shape = (50, 80)
    for sample in ("case01", "case02"):
        sample_dir = root / sample
        gt_dir = sample_dir / "Flatten_561_mask"
        raw_dir = sample_dir / "Flatten_561"
        gt_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)
        gt0, pr0, raw0 = _synthetic_slice(shape)
        tifffile.imwrite(gt_dir / "slice_0000.tif", gt0)
        tifffile.imwrite(raw_dir / "slice_0000.tif", raw0)

        for model_dir_name in (
            "Flatten_561_v12_unet_baseline_mask.scroll-tif",
            "Flatten_561_v12_unet_baseline_dark_mask.scroll-tif",
        ):
            pred_dir = sample_dir / model_dir_name
            pred_dir.mkdir(parents=True)
            tifffile.imwrite(pred_dir / "slice_0000.tif", pr0)


def test_collect_model_exact_excludes_substring_sibling(tmp_path):
    _write_dataset_with_sibling_models(tmp_path)

    instances_df, _ = collect(tmp_path, model_exact="v12_unet_baseline")

    assert set(instances_df["model"]) == {"v12_unet_baseline"}


def test_collect_model_filter_substring_would_include_the_sibling(tmp_path):
    """Sanity check that the scenario above is real: the existing substring
    `model_filter` *does* pick up both models, which is exactly why
    `model_exact` exists.
    """
    _write_dataset_with_sibling_models(tmp_path)

    instances_df, _ = collect(tmp_path, model_filter="v12_unet_baseline")

    assert set(instances_df["model"]) == {"v12_unet_baseline", "v12_unet_baseline_dark"}


def test_collect_max_slices_caps_total_slices_scanned(tmp_path):
    _write_dataset(tmp_path)

    # Each sample has 2 slices; capping at 1 total slice across the whole
    # scan should only ever process the first sample's first slice.
    instances_df, quality_df = collect(tmp_path, max_slices=1)

    assert set(instances_df["sample"]) == {"case01"}
    assert set(instances_df["slice_name"]) == {"slice_0000.tif"}
    assert set(quality_df["slice_name"]) == {"slice_0000.tif"}


def test_collect_uses_cache_when_present(tmp_path):
    _write_dataset(tmp_path)
    cache_path = tmp_path / "cache" / "run"

    first_df, _ = collect(tmp_path, cache_path=cache_path)
    assert cache_path.with_suffix(".instances.parquet").exists()
    assert cache_path.with_suffix(".quality.parquet").exists()

    # Corrupt the underlying dataset; a cached read must not reflect this.
    (tmp_path / "case01" / "Flatten_561_mask" / "slice_0000.tif").unlink()

    cached_df, _ = collect(tmp_path, cache_path=cache_path)
    pd.testing.assert_frame_equal(first_df, cached_df)


def test_collect_force_refresh_bypasses_cache(tmp_path):
    _write_dataset(tmp_path)
    cache_path = tmp_path / "cache" / "run"

    first_df, _ = collect(tmp_path, cache_path=cache_path)
    (tmp_path / "case02" / "Flatten_561_mask" / "slice_0000.tif").unlink()

    refreshed_df, _ = collect(tmp_path, cache_path=cache_path, force_refresh=True)

    assert len(refreshed_df) < len(first_df)


def test_collect_returns_empty_instances_df_when_no_gt_folders(tmp_path):
    instances_df, quality_df = collect(tmp_path)
    assert instances_df.empty
    assert quality_df.empty
    assert list(instances_df.columns) == [f for f in InstanceRecord.__dataclass_fields__]


# --- fp_subtype (fp_root_cause) and image-quality population ---------------


def test_collect_populates_fp_subtype_for_every_false_positive(tmp_path):
    # A dedicated dataset with real background variation (not the flat-raw
    # _write_dataset fixture above) - fp_subtype now hinges on local
    # background *contrast*, so a perfectly flat raw image makes every FP
    # indistinguishable from noise_fp regardless of what it's meant to test.
    shape = (60, 90)
    sample_dir = tmp_path / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    raw_dir = sample_dir / "Flatten_561"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    for d in (gt_dir, raw_dir, pred_dir):
        d.mkdir(parents=True)

    rng = np.random.default_rng(0)
    gt = np.zeros(shape, dtype=np.uint8)
    pr = np.zeros(shape, dtype=np.uint8)
    raw = rng.normal(20.0, 1.0, shape).astype(np.float32)

    # A matched TP cell, so the slice isn't otherwise degenerate.
    gt[5:15, 5:15] = 1
    pr[5:15, 5:15] = 1
    raw[5:15, 5:15] = 60.0

    # boundary_split_fp: a merged-FN GT cell whose prediction is shifted
    # just enough to fail one-to-one matching, at a clearly non-background
    # intensity - sits right next to a real cell, so it's an over-
    # segmentation split, not noise or a hallucination.
    gt[30:40, 30:40] = 1
    pr[33:43, 33:43] = 1  # area 100, centroid ~4px from the GT above
    raw[33:43, 33:43] = 65.0

    # hallucination_fp: far from any GT, bright/non-background.
    pr[5:11, 70:76] = 1  # area 36
    raw[5:11, 70:76] = 90.0

    # noise_fp: far from any GT, left at the ambient background level.
    pr[50:55, 5:10] = 1  # area 25 (distinct from the other two FPs' volumes)

    tifffile.imwrite(gt_dir / "slice_0000.tif", gt)
    tifffile.imwrite(pred_dir / "slice_0000.tif", pr)
    tifffile.imwrite(raw_dir / "slice_0000.tif", raw)

    instances_df, _ = collect(tmp_path)
    fp_rows = instances_df[instances_df["role"] == "prediction"]

    assert len(fp_rows) == 3
    by_volume = {int(row["volume"]): row["fp_subtype"] for _, row in fp_rows.iterrows()}
    assert by_volume == {36: "hallucination_fp", 100: "boundary_split_fp", 25: "noise_fp"}


def test_collect_gt_role_rows_never_have_fp_subtype(tmp_path):
    _write_dataset(tmp_path)

    instances_df, _ = collect(tmp_path)
    gt_rows = instances_df[instances_df["role"] == "gt"]

    assert gt_rows["fp_subtype"].isna().all()


def test_collect_populates_quality_metrics_per_slice(tmp_path):
    _write_dataset(tmp_path)

    _, quality_df = collect(tmp_path)

    assert (quality_df["gt_instance_count"] >= 1).all()
    assert (quality_df["mean_intensity"] > 0).all()
    # The synthetic raw arrays are perfectly flat, so every pixel is at the
    # (only) max value.
    assert (quality_df["saturation_pct"] == 100.0).all()
