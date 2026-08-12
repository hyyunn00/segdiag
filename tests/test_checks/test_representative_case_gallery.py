"""Tests for the representative-case-gallery check
(SEGDIAG_REPRESENTATIVE_CASE_GALLERY.md + the two-panel figure-legend
redesign it now follows), covering:

1. filter selection is reproducible
2. sampling (which single candidate gets picked) is reproducible for a
   fixed seed
3. z_discontinuity boundary (GT Z-span >= 3, matched prediction Z-span == 1)
4. missing_gt_annotation boundary (background_contrast_sigma >= 2.0)
5. intensity-window consistency (one shared vmin/vmax per figure)
6. a missing pattern degrades gracefully (warns, doesn't error, the other
   panel still renders)
7. the fixed 64x64 crop window is centered on the instance's centroid and
   identical across patterns
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest
import tifffile

from segdiag.checks.representative_case_gallery import (
    FIXED_CROP_SIZE,
    RepresentativeCaseGalleryCheck,
    _crop_fixed_window,
    _resolve_intensity_window,
    _sample_cases,
    _select_candidates,
)
from segdiag.core.pipeline import collect

_COLUMNS = [
    "dataset",
    "sample",
    "model",
    "slice_name",
    "z_index",
    "role",
    "instance_id",
    "volume",
    "mean_intensity",
    "classification",
    "best_iou",
    "fp_subtype",
    "bbox_min_z",
    "bbox_max_z",
    "matched_pred_z_span",
    "background_contrast_sigma",
    "bbox_min_row",
    "bbox_min_col",
    "bbox_max_row",
    "bbox_max_col",
    "centroid_y",
    "centroid_x",
]


def _row(**overrides) -> dict:
    base = {c: None for c in _COLUMNS}
    base.update(
        dataset="ds",
        sample="case01",
        model="unet_v9",
        slice_name="slice_0000.tif",
        z_index=0,
        role="gt",
        instance_id=1,
        volume=100,
        mean_intensity=40.0,
        classification="true_positive",
        best_iou=0.9,
        bbox_min_row=0,
        bbox_min_col=0,
        bbox_max_row=10,
        bbox_max_col=10,
        centroid_y=5.0,
        centroid_x=5.0,
    )
    base.update(overrides)
    return base


def _instances(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_COLUMNS)


# --- 1. filter selection is reproducible -----------------------------------


def test_select_candidates_is_reproducible_for_every_pattern():
    instances = _instances(
        [
            _row(role="gt", classification="true_positive", best_iou=0.55),
            _row(role="gt", classification="blind_fn", best_iou=0.0),
            _row(
                role="gt",
                classification="true_positive",
                bbox_min_z=0,
                bbox_max_z=3,
                matched_pred_z_span=1,
            ),
            _row(role="prediction", classification="false_positive", fp_subtype="noise_fp"),
            _row(
                role="prediction",
                classification="false_positive",
                fp_subtype="hallucination_fp",
                background_contrast_sigma=3.0,
            ),
        ]
    )
    for pattern in [
        "contour_underestimate",
        "no_response",
        "z_discontinuity",
        "background_noise",
        "missing_gt_annotation",
    ]:
        first = _select_candidates(instances, pattern)
        second = _select_candidates(instances, pattern)
        pd.testing.assert_frame_equal(first, second)


def test_select_candidates_rejects_unknown_pattern():
    with pytest.raises(ValueError):
        _select_candidates(_instances([]), "not_a_real_pattern")


def test_contour_underestimate_filters_by_iou_half_open_range():
    instances = _instances(
        [
            _row(instance_id=1, best_iou=0.49),  # below range
            _row(instance_id=2, best_iou=0.50),  # inclusive lower bound
            _row(instance_id=3, best_iou=0.55),  # inside
            _row(instance_id=4, best_iou=0.60),  # exclusive upper bound
            _row(instance_id=5, classification="blind_fn", best_iou=0.55),  # wrong classification
        ]
    )
    selected = _select_candidates(instances, "contour_underestimate")
    assert sorted(selected["instance_id"]) == [2, 3]


def test_no_response_selects_only_blind_fn_gt_rows():
    instances = _instances(
        [
            _row(instance_id=1, classification="blind_fn"),
            _row(instance_id=2, classification="merged_fn"),
            _row(instance_id=3, role="prediction", classification="false_positive"),
        ]
    )
    selected = _select_candidates(instances, "no_response")
    assert list(selected["instance_id"]) == [1]


def test_background_noise_selects_only_noise_fp_predictions():
    instances = _instances(
        [
            _row(
                instance_id=1,
                role="prediction",
                classification="false_positive",
                fp_subtype="noise_fp",
            ),
            _row(
                instance_id=2,
                role="prediction",
                classification="false_positive",
                fp_subtype="boundary_split_fp",
            ),
            _row(instance_id=3, role="gt", classification="blind_fn"),
        ]
    )
    selected = _select_candidates(instances, "background_noise")
    assert list(selected["instance_id"]) == [1]


# --- 3. z_discontinuity boundary --------------------------------------------


def test_z_discontinuity_selects_z_span_3_but_not_z_span_2():
    instances = _instances(
        [
            # GT Z-span == 3 (>=3, qualifies), matched prediction Z-span == 1.
            _row(instance_id=1, bbox_min_z=0, bbox_max_z=3, matched_pred_z_span=1),
            # GT Z-span == 2 (< 3, does not qualify) - the exact "one below
            # the boundary" case section 7 item 3 calls for.
            _row(instance_id=2, bbox_min_z=0, bbox_max_z=2, matched_pred_z_span=1),
            # GT Z-span == 3 but the matched prediction itself spans >1 slice
            # - not a discontinuity, must not be selected.
            _row(instance_id=3, bbox_min_z=0, bbox_max_z=3, matched_pred_z_span=2),
            # Not a strict match at all - irrelevant regardless of spans.
            _row(
                instance_id=4,
                classification="merged_fn",
                bbox_min_z=0,
                bbox_max_z=5,
                matched_pred_z_span=None,
            ),
        ]
    )
    selected = _select_candidates(instances, "z_discontinuity")
    assert list(selected["instance_id"]) == [1]


# --- 4. missing_gt_annotation boundary --------------------------------------


def test_missing_gt_annotation_includes_sigma_exactly_two():
    instances = _instances(
        [
            _row(
                instance_id=1,
                role="prediction",
                classification="false_positive",
                fp_subtype="hallucination_fp",
                background_contrast_sigma=2.0,  # inclusive boundary
            ),
            _row(
                instance_id=2,
                role="prediction",
                classification="false_positive",
                fp_subtype="hallucination_fp",
                background_contrast_sigma=1.999999,  # just below - excluded
            ),
            _row(
                instance_id=3,
                role="prediction",
                classification="false_positive",
                fp_subtype="noise_fp",
                background_contrast_sigma=5.0,  # right subtype but wrong fp_subtype
            ),
        ]
    )
    selected = _select_candidates(instances, "missing_gt_annotation")
    assert list(selected["instance_id"]) == [1]


# --- 2. sampling is reproducible ---------------------------------------------


def test_sample_cases_is_reproducible_for_a_fixed_seed():
    candidates = _instances([_row(instance_id=i) for i in range(20)])
    first = _sample_cases(candidates, "contour_underestimate", n=1, seed=42)
    second = _sample_cases(candidates, "contour_underestimate", n=1, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_sample_cases_differs_across_seeds_with_enough_candidates():
    candidates = _instances([_row(instance_id=i) for i in range(20)])
    a = _sample_cases(candidates, "contour_underestimate", n=1, seed=1)
    b = _sample_cases(candidates, "contour_underestimate", n=1, seed=2)
    assert list(a["instance_id"]) != list(b["instance_id"])


def test_sample_cases_handles_a_too_small_candidate_pool_without_erroring():
    candidates = _instances([_row(instance_id=1)])
    sampled = _sample_cases(candidates, "no_response", n=1, seed=42)
    assert len(sampled) == 1


def test_sample_cases_returns_empty_when_no_candidates():
    empty = _instances([])
    assert _sample_cases(empty, "no_response", n=1, seed=42).empty


# --- 5. intensity-window consistency ----------------------------------------


def test_resolve_intensity_window_uses_pooled_0_1_99_9_percentile():
    rng = np.random.default_rng(0)
    arrays = [rng.normal(50, 5, size=(20, 20)) for _ in range(4)]
    vmin, vmax = _resolve_intensity_window(arrays, None, None)

    pooled = np.concatenate([a.ravel() for a in arrays])
    assert vmin == pytest.approx(np.percentile(pooled, 0.1))
    assert vmax == pytest.approx(np.percentile(pooled, 99.9))
    assert vmin < vmax


def test_resolve_intensity_window_explicit_bounds_win_over_percentiles():
    rng = np.random.default_rng(0)
    arrays = [rng.normal(50, 5, size=(20, 20))]
    vmin, vmax = _resolve_intensity_window(arrays, 10.0, 90.0)
    assert (vmin, vmax) == (10.0, 90.0)


def test_resolve_intensity_window_degrades_gracefully_with_no_arrays():
    vmin, vmax = _resolve_intensity_window([], None, None)
    assert vmin < vmax


# --- 7. fixed 64x64 crop window ----------------------------------------------


def test_crop_fixed_window_is_always_the_configured_size_away_from_edges():
    arr = np.arange(300 * 300).reshape(300, 300)
    crop = _crop_fixed_window(arr, center_row=150, center_col=150)
    assert crop.shape == (FIXED_CROP_SIZE, FIXED_CROP_SIZE)


def test_crop_fixed_window_is_centered_on_the_given_centroid():
    arr = np.zeros((300, 300), dtype=np.uint8)
    arr[150, 150] = 1  # a single marker voxel at the intended center
    crop = _crop_fixed_window(arr, center_row=150, center_col=150)
    assert crop.shape == (FIXED_CROP_SIZE, FIXED_CROP_SIZE)
    # The marker should land exactly in the middle of the crop.
    marker_row, marker_col = np.argwhere(crop == 1)[0]
    assert marker_row == FIXED_CROP_SIZE // 2
    assert marker_col == FIXED_CROP_SIZE // 2


def test_crop_fixed_window_slides_to_stay_in_bounds_near_an_edge():
    arr = np.zeros((300, 300), dtype=np.uint8)
    crop = _crop_fixed_window(arr, center_row=2, center_col=2)
    assert crop.shape == (FIXED_CROP_SIZE, FIXED_CROP_SIZE)


# --- End-to-end wiring: collect() -> check, on a small synthetic 3D dataset -


def _build_gallery_dataset(root) -> None:
    """Plants one case of each of the five patterns in a single 3D (sample,
    model) volume, so a full ``collect()`` -> check run exercises the real
    ``core.pipeline`` wiring (not just the pure filter functions above) for
    ``matched_pred_z_span``/``background_contrast_sigma``.
    """
    shape = (300, 300)
    # z_discontinuity needs a full Z-2..Z+2 window around its center slice,
    # so at least 5 slices are required with the discontinuity's center at
    # index 2.
    num_z = 5
    sample_dir = root / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    raw_dir = sample_dir / "Flatten_561"
    dark_dir = sample_dir / "Flatten_561_dark"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    for d in (gt_dir, raw_dir, dark_dir, pred_dir):
        d.mkdir(parents=True)

    rng = np.random.default_rng(0)
    background_mean = 20.0

    gt = np.zeros((num_z, *shape), dtype=np.uint8)
    pr = np.zeros((num_z, *shape), dtype=np.uint8)
    raw = rng.normal(background_mean, 1.0, (num_z, *shape)).astype(np.float32)

    # z_discontinuity: GT spans z=1..3 (a 10x10 blob at z=2, plus single
    # "trailing" voxels at z=1/z=3 that extend the bbox without adding much
    # volume), matched almost exactly by a prediction confined to z=2 alone.
    # Centered at z=2 so the Z-2..Z+2 context window (z=0..4) fits inside
    # this dataset's 5 slices.
    gt[2, 10:20, 10:20] = 1
    gt[1, 15, 15] = 1
    gt[3, 15, 15] = 1
    pr[2, 10:20, 10:20] = 1

    # contour_underestimate: two 10x10 squares offset by 3 columns -> IoU =
    # 70 / 130 ~= 0.538, inside [0.50, 0.60).
    gt[0, 50:60, 50:60] = 1
    pr[0, 50:60, 53:63] = 1

    # no_response: an isolated GT cell with zero prediction anywhere near it.
    gt[0, 100:110, 100:110] = 1

    # background_noise: an isolated, unmatched prediction at background
    # intensity (7x7 = 49 voxels, clears the default MARS min_volume=40).
    pr[0, 150:157, 150:157] = 1
    raw[0, 150:157, 150:157] = background_mean

    # missing_gt_annotation: an isolated, unmatched prediction with strong
    # local contrast (10x10 = 100 voxels), far from any GT.
    pr[0, 200:210, 200:210] = 1
    raw[0, 200:210, 200:210] = background_mean + 20.0

    dark = np.full((num_z, *shape), 5.0, dtype=np.float32)

    for z in range(num_z):
        tifffile.imwrite(gt_dir / f"slice_{z:04d}.tif", gt[z])
        tifffile.imwrite(pred_dir / f"slice_{z:04d}.tif", pr[z])
        tifffile.imwrite(raw_dir / f"slice_{z:04d}.tif", raw[z])
        tifffile.imwrite(dark_dir / f"slice_{z:04d}.tif", dark[z])


@pytest.fixture
def gallery_dataset(tmp_path):
    _build_gallery_dataset(tmp_path)
    instances_df, quality_df = collect(tmp_path)
    args = argparse.Namespace(
        root=tmp_path,
        sample=None,
        model=None,
        output_dir=tmp_path / "out",
        mask_name=None,
        raw_name=None,
        dark_name="Flatten_561_dark",
        gallery_seed=42,
        intensity_vmin=None,
        intensity_vmax=None,
        voxel_size_um=1.82,
    )
    return instances_df, quality_df, args


def test_pipeline_populates_matched_pred_z_span_and_background_contrast_sigma(gallery_dataset):
    instances_df, _quality_df, _args = gallery_dataset

    tp_rows = instances_df[
        (instances_df["role"] == "gt") & (instances_df["classification"] == "true_positive")
    ]
    z_span_rows = tp_rows[tp_rows["bbox_max_z"] - tp_rows["bbox_min_z"] >= 3]
    assert len(z_span_rows) == 1
    assert z_span_rows.iloc[0]["matched_pred_z_span"] == 1

    hallucination = instances_df[instances_df["fp_subtype"] == "hallucination_fp"]
    assert len(hallucination) == 1
    assert hallucination.iloc[0]["background_contrast_sigma"] >= 2.0


def test_representative_case_gallery_check_produces_both_panels(gallery_dataset):
    instances_df, quality_df, args = gallery_dataset

    artifacts = RepresentativeCaseGalleryCheck().run(instances_df, quality_df, args)
    by_name = {a.name: a for a in artifacts}

    assert "case_gallery_sampling_summary" in by_name
    summary = by_name["case_gallery_sampling_summary"].table
    assert set(summary["pattern"]) == {
        "contour_underestimate",
        "no_response",
        "z_discontinuity",
        "background_noise",
        "missing_gt_annotation",
    }
    # Every planted pattern has exactly one candidate in this dataset, and
    # every one of them should have been renderable (folders/slices exist).
    assert (summary["candidate_count"] >= 1).all()
    assert (summary["sampled_count"] == 1).all()

    assert "case_gallery_panel_a_general" in by_name
    panel_a = by_name["case_gallery_panel_a_general"]
    assert panel_a.figure is not None
    # One row per pattern - exactly one example each, not a multi-case gallery.
    assert len(panel_a.table) == 4
    assert set(panel_a.table["classification"]).issubset(
        {"true_positive", "blind_fn", "false_positive"}
    )
    assert panel_a.metadata["crop_size"] == FIXED_CROP_SIZE

    assert "case_gallery_panel_b_z_discontinuity" in by_name
    panel_b = by_name["case_gallery_panel_b_z_discontinuity"]
    assert panel_b.figure is not None
    assert len(panel_b.table) == 1
    assert panel_b.metadata["crop_size"] == FIXED_CROP_SIZE


def test_representative_case_gallery_check_degrades_gracefully_when_one_pattern_is_missing(
    tmp_path,
):
    """No hallucination_fp planted at all -> missing_gt_annotation has zero
    candidates. Panel A must still render for the other three patterns, and
    Panel B must still render, instead of the whole check failing.
    """
    shape = (300, 300)
    num_z = 5
    sample_dir = tmp_path / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    raw_dir = sample_dir / "Flatten_561"
    dark_dir = sample_dir / "Flatten_561_dark"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    for d in (gt_dir, raw_dir, dark_dir, pred_dir):
        d.mkdir(parents=True)

    rng = np.random.default_rng(0)
    gt = np.zeros((num_z, *shape), dtype=np.uint8)
    pr = np.zeros((num_z, *shape), dtype=np.uint8)
    raw = rng.normal(20.0, 1.0, (num_z, *shape)).astype(np.float32)
    dark = np.full((num_z, *shape), 5.0, dtype=np.float32)

    gt[2, 10:20, 10:20] = 1
    gt[1, 15, 15] = 1
    gt[3, 15, 15] = 1
    pr[2, 10:20, 10:20] = 1

    gt[0, 50:60, 50:60] = 1
    pr[0, 50:60, 53:63] = 1

    gt[0, 100:110, 100:110] = 1

    pr[0, 150:157, 150:157] = 1
    raw[0, 150:157, 150:157] = 20.0

    for z in range(num_z):
        tifffile.imwrite(gt_dir / f"slice_{z:04d}.tif", gt[z])
        tifffile.imwrite(pred_dir / f"slice_{z:04d}.tif", pr[z])
        tifffile.imwrite(raw_dir / f"slice_{z:04d}.tif", raw[z])
        tifffile.imwrite(dark_dir / f"slice_{z:04d}.tif", dark[z])

    instances_df, quality_df = collect(tmp_path)
    args = argparse.Namespace(
        root=tmp_path,
        sample=None,
        model=None,
        output_dir=tmp_path / "out",
        mask_name=None,
        raw_name=None,
        dark_name="Flatten_561_dark",
        gallery_seed=42,
        intensity_vmin=None,
        intensity_vmax=None,
        voxel_size_um=1.82,
    )

    artifacts = RepresentativeCaseGalleryCheck().run(instances_df, quality_df, args)
    by_name = {a.name: a for a in artifacts}

    summary = by_name["case_gallery_sampling_summary"].table
    missing_row = summary[summary["pattern"] == "missing_gt_annotation"].iloc[0]
    assert missing_row["candidate_count"] == 0
    assert missing_row["sampled_count"] == 0

    panel_a = by_name["case_gallery_panel_a_general"]
    assert panel_a.figure is not None
    assert len(panel_a.table) == 3  # missing_gt_annotation excluded, three others present

    assert by_name["case_gallery_panel_b_z_discontinuity"].figure is not None


def test_representative_case_gallery_check_returns_empty_when_no_instances():
    empty = pd.DataFrame(columns=["role"])
    args = argparse.Namespace(root=None)
    assert RepresentativeCaseGalleryCheck().run(empty, empty, args) == []
