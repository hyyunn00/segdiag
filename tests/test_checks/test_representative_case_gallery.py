"""Tests for the representative-case-gallery check
(SEGDIAG_REPRESENTATIVE_CASE_GALLERY.md + the two-panel figure-legend
redesign it now follows), covering:

1. filter selection is reproducible
2. sampling (which single candidate gets picked) is reproducible for a
   fixed seed
3. z_discontinuity boundary (GT Z-span >= 3, matched prediction Z-span at
   most Z_DISCONTINUITY_COVERAGE_RATIO of the GT's Z-span)
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
    Z_DISCONTINUITY_COVERAGE_RATIO,
    Z_DISCONTINUITY_MIN_GT_Z_SPAN,
    RepresentativeCaseGalleryCheck,
    _crop_fixed_window,
    _pick_and_load,
    _render_panel_a,
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


def test_z_discontinuity_coverage_ratio_constants_are_what_the_tests_below_assume():
    """Pin the current threshold values - a deliberate change to either
    constant should force an update here (and in the boundary math below),
    not silently drift.
    """
    assert Z_DISCONTINUITY_COVERAGE_RATIO == 0.3
    assert Z_DISCONTINUITY_MIN_GT_Z_SPAN == 3


def test_z_discontinuity_selects_by_coverage_ratio_not_a_hard_span_of_one():
    """Regression for switching from the original spec's hard
    ``matched_pred_z_span == 1`` to a proportional rule: a real trained
    model's matched (TP) predictions essentially never come out exactly
    1-slice-thick in 3D (confirmed on an actual dataset - see
    Z_DISCONTINUITY_COVERAGE_RATIO's docstring), so the literal rule always
    yielded zero candidates. GT Z-span 10, ratio 0.3 -> boundary at
    matched_pred_z_span == 3 (inclusive).
    """
    instances = _instances(
        [
            # matched/GT ratio exactly 3/10 == 0.3 - inclusive boundary, qualifies.
            _row(instance_id=1, bbox_min_z=0, bbox_max_z=10, matched_pred_z_span=3),
            # matched/GT ratio 4/10 == 0.4 - just over the boundary, excluded.
            _row(instance_id=2, bbox_min_z=0, bbox_max_z=10, matched_pred_z_span=4),
            # GT Z-span 2 (< Z_DISCONTINUITY_MIN_GT_Z_SPAN=3) - even a
            # generous ratio (1/2 == 0.5) doesn't count on a cell this short.
            _row(instance_id=3, bbox_min_z=0, bbox_max_z=2, matched_pred_z_span=1),
            # Not a strict match at all - irrelevant regardless of spans.
            _row(
                instance_id=4,
                classification="merged_fn",
                bbox_min_z=0,
                bbox_max_z=14,
                matched_pred_z_span=None,
            ),
            # matched_pred_z_span missing (NaN) despite an otherwise-TP row
            # with a tall GT - excluded, can't compute a ratio.
            _row(instance_id=5, bbox_min_z=0, bbox_max_z=14, matched_pred_z_span=None),
        ]
    )
    selected = _select_candidates(instances, "z_discontinuity")
    assert list(selected["instance_id"]) == [1]


def test_z_discontinuity_matches_the_real_data_case_that_motivated_the_ratio_switch():
    """The exact shape of case the diagnosis surfaced: a GT cell spanning
    14 slices, matched by a prediction spanning only 3 - a real
    Z-underestimation the original literal ``== 1`` rule could never select.
    """
    instances = _instances(
        [_row(instance_id=1, bbox_min_z=0, bbox_max_z=14, matched_pred_z_span=3)]
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


# --- retry: fall back through the pool if the first candidate fails to load -


def test_pick_and_load_falls_back_to_the_next_candidate_when_the_first_fails():
    """The key regression this mechanism exists for: a candidate whose
    images can't actually be loaded (stale cache, missing files, too close
    to a volume edge...) must not silently drop the whole pattern when a
    different candidate in the same pool would have worked.
    """
    candidates = _instances(
        [_row(instance_id=1, sample="bad_sample"), _row(instance_id=2, sample="good_sample")]
    )

    def folders_for(sample, model):
        return ("gt_dir", "raw_dir", "dark_dir", "pred_dir", [])

    def loader(case, folders):
        return None if case["sample"] == "bad_sample" else {"raw": [np.zeros((4, 4))]}

    case, loaded = _pick_and_load(
        candidates, "z_discontinuity", seed=42, folders_for=folders_for, loader=loader
    )

    assert loaded is not None
    assert case["sample"] == "good_sample"


def test_pick_and_load_is_reproducible_for_a_fixed_seed():
    candidates = _instances([_row(instance_id=i, sample=f"s{i}") for i in range(10)])
    calls = []

    def folders_for(sample, model):
        return ("gt_dir", "raw_dir", "dark_dir", "pred_dir", [])

    def loader(case, folders):
        calls.append(case["sample"])
        return {"raw": [np.zeros((4, 4))]}  # first attempt always succeeds

    case_a, _ = _pick_and_load(
        candidates, "no_response", seed=7, folders_for=folders_for, loader=loader
    )
    case_b, _ = _pick_and_load(
        candidates, "no_response", seed=7, folders_for=folders_for, loader=loader
    )
    assert case_a["sample"] == case_b["sample"]


def test_pick_and_load_returns_none_when_every_candidate_fails():
    candidates = _instances([_row(instance_id=1), _row(instance_id=2)])
    case, loaded = _pick_and_load(
        candidates,
        "z_discontinuity",
        seed=42,
        folders_for=lambda sample, model: ("gt", "raw", "dark", "pred", []),
        loader=lambda case, folders: None,
    )
    assert case is None
    assert loaded is None


def test_pick_and_load_returns_none_for_an_empty_pool():
    empty = _instances([])
    case, loaded = _pick_and_load(
        empty,
        "z_discontinuity",
        seed=42,
        folders_for=lambda sample, model: None,
        loader=lambda case, folders: None,
    )
    assert case is None
    assert loaded is None


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


# --- Regression: Dark Sectioning must not inherit Raw's intensity window ----


def test_render_panel_a_gives_dark_column_its_own_intensity_window():
    """Dark Sectioning is a different imaging channel from Raw, with its own
    intensity range. Reusing Raw's vmin/vmax for it (the original bug)
    blows the Dark column out whenever Dark's pixel values sit outside
    Raw's window - e.g. here Raw ~20, Dark ~200.
    """
    raw = np.full((FIXED_CROP_SIZE, FIXED_CROP_SIZE), 20.0)
    dark = np.full((FIXED_CROP_SIZE, FIXED_CROP_SIZE), 200.0)
    gt = np.zeros((FIXED_CROP_SIZE, FIXED_CROP_SIZE), dtype=np.uint8)
    pr = np.zeros((FIXED_CROP_SIZE, FIXED_CROP_SIZE), dtype=np.uint8)
    loaded = {"no_response": {"raw": raw, "dark": dark, "gt": gt, "pr": pr}}

    fig = _render_panel_a(
        loaded, vmin=0.0, vmax=50.0, dark_vmin=150.0, dark_vmax=250.0, voxel_size_um=1.82
    )

    raw_ax, dark_ax = fig.axes[0], fig.axes[1]
    assert raw_ax.images[0].get_clim() == (0.0, 50.0)
    assert dark_ax.images[0].get_clim() == (150.0, 250.0)


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

    # z_discontinuity: GT spans all 5 z-slices (a 10x10 blob at z=2, plus
    # single "trailing" voxels at z=0/1/3/4 that extend the bbox without
    # adding much volume: GT z-span=5), matched by a prediction confined to
    # z=2 alone (matched z-span=1, ratio 1/5=0.2 <= Z_DISCONTINUITY_COVERAGE_RATIO).
    # Centered at z=2 so the Z-2..Z+2 context window (z=0..4) fits inside
    # this dataset's 5 slices.
    gt[2, 10:20, 10:20] = 1
    gt[0, 15, 15] = 1
    gt[1, 15, 15] = 1
    gt[3, 15, 15] = 1
    gt[4, 15, 15] = 1
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
    gt[0, 15, 15] = 1
    gt[1, 15, 15] = 1
    gt[3, 15, 15] = 1
    gt[4, 15, 15] = 1
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


def test_representative_case_gallery_check_falls_back_past_an_edge_candidate_for_panel_b(tmp_path):
    """End-to-end version of the retry regression: two genuine
    z_discontinuity candidates in the same dataset, one centered too close
    to its sample's volume edge for a full Z-2..Z+2 window (unrenderable)
    and one safely in the middle (renderable). Panel B must still render,
    using the renderable one, regardless of which one a given seed tries
    first.
    """
    shape = (300, 300)
    num_z = 7  # only z_index in [2, num_z-3] = [2, 4] has room for Z-2..Z+2
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

    # Candidate 1 (unrenderable): centered at z=1, which is < 2 - too close
    # to the z=0 edge for a Z-2..Z+2 window. GT z-span=4 (z=0..3), matched
    # z-span=1 (z=1 only) -> ratio 0.25 <= Z_DISCONTINUITY_COVERAGE_RATIO.
    gt[1, 10:20, 10:20] = 1
    gt[0, 15, 15] = 1
    gt[2, 15, 15] = 1
    gt[3, 15, 15] = 1
    pr[1, 10:20, 10:20] = 1

    # Candidate 2 (renderable): centered at z=4, the last index with room
    # for a full Z-2..Z+2 window in a 7-slice volume (z=2..6). GT z-span=4
    # (z=2..5), matched z-span=1 (z=4 only) -> same 0.25 ratio.
    gt[4, 50:60, 50:60] = 1
    gt[2, 55, 55] = 1
    gt[3, 55, 55] = 1
    gt[5, 55, 55] = 1
    pr[4, 50:60, 50:60] = 1

    for z in range(num_z):
        tifffile.imwrite(gt_dir / f"slice_{z:04d}.tif", gt[z])
        tifffile.imwrite(pred_dir / f"slice_{z:04d}.tif", pr[z])
        tifffile.imwrite(raw_dir / f"slice_{z:04d}.tif", raw[z])
        tifffile.imwrite(dark_dir / f"slice_{z:04d}.tif", dark[z])

    instances_df, quality_df = collect(tmp_path)
    z_candidates = _select_candidates(instances_df, "z_discontinuity")
    assert len(z_candidates) == 2  # both planted cells qualify as candidates

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

    panel_b = by_name["case_gallery_panel_b_z_discontinuity"]
    assert panel_b.figure is not None
    assert int(panel_b.table.iloc[0]["z_index"]) == 4  # the renderable candidate, not z=1

    summary = by_name["case_gallery_sampling_summary"].table
    z_row = summary[summary["pattern"] == "z_discontinuity"].iloc[0]
    assert z_row["candidate_count"] == 2
    assert z_row["sampled_count"] == 1


def test_representative_case_gallery_check_returns_empty_when_no_instances():
    empty = pd.DataFrame(columns=["role"])
    args = argparse.Namespace(root=None)
    assert RepresentativeCaseGalleryCheck().run(empty, empty, args) == []
