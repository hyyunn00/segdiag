"""Unit tests for segdiag.core.matching.

These tests use small synthetic 2D arrays instead of real microscopy TIFFs,
so they run instantly and require no test data.
"""

import numpy as np
import pytest
from skimage.measure import label

from segdiag.core.matching import (
    LOCATED_IOU_THRESHOLD,
    cc3d_to_skimage_connectivity,
    filter_labels_by_volume,
    find_false_positives,
    find_fn_bboxes,
    match_and_find_false_positives,
    match_instances,
)


def _make_square_mask(shape, top_left, size, value=1):
    arr = np.zeros(shape, dtype=np.uint8)
    r, c = top_left
    arr[r : r + size, c : c + size] = value
    return arr


def test_perfect_overlap_is_true_positive():
    gt = _make_square_mask((50, 50), (10, 10), 10)
    pr = _make_square_mask((50, 50), (10, 10), 10)

    matches = match_instances(gt, pr)

    assert len(matches) == 1
    assert matches[0].best_iou == 1.0
    assert matches[0].is_tp
    assert matches[0].matched
    assert matches[0].classify() == "true_positive"


def test_no_overlap_is_blind_false_negative():
    gt = _make_square_mask((50, 50), (5, 5), 8)
    pr = _make_square_mask((50, 50), (30, 30), 8)

    matches = match_instances(gt, pr)

    assert len(matches) == 1
    assert matches[0].best_iou == 0.0
    assert matches[0].is_fn
    assert not matches[0].matched
    assert matches[0].classify() == "blind_fn"


def test_partial_overlap_is_merged_false_negative():
    gt = _make_square_mask((50, 50), (10, 10), 10)  # area 100
    pr = _make_square_mask((50, 50), (15, 15), 10)  # shifted, partial overlap

    matches = match_instances(gt, pr)

    assert len(matches) == 1
    assert 0.05 <= matches[0].best_iou < 0.5
    assert not matches[0].matched
    assert matches[0].classify() == "merged_fn"


def test_empty_gt_returns_no_matches():
    gt = np.zeros((20, 20), dtype=np.uint8)
    pr = _make_square_mask((20, 20), (2, 2), 5)

    assert match_instances(gt, pr) == []


def test_intensity_is_populated_when_raw_array_given():
    gt = _make_square_mask((20, 20), (2, 2), 5)  # area 25, below MARS_MIN_VOLUME
    pr = _make_square_mask((20, 20), (2, 2), 5)
    raw = np.full((20, 20), 42.0, dtype=np.float32)

    matches = match_instances(gt, pr, raw_arr=raw, min_volume=None, max_volume=None)

    assert matches[0].mean_intensity == 42.0


def test_find_fn_bboxes_filters_by_threshold():
    gt = _make_square_mask((50, 50), (5, 5), 8)
    pr = _make_square_mask((50, 50), (30, 30), 8)

    bboxes = find_fn_bboxes(gt, pr, threshold=0.05)

    assert len(bboxes) == 1


# --- one-to-one matching alignment with metrics.compute_object_metrics ------


def test_extra_prediction_with_no_gt_partner_is_a_false_positive():
    gt = _make_square_mask((50, 50), (5, 5), 8)
    pr = np.zeros((50, 50), dtype=np.uint8)
    pr[5:13, 5:13] = 1  # matches the GT cell
    pr[30:38, 30:38] = 1  # spurious extra detection, no GT nearby

    matches = match_instances(gt, pr)
    fps = find_false_positives(gt, pr)

    assert len(matches) == 1 and matches[0].matched
    assert len(fps) == 1
    assert fps[0].volume == 64  # 8x8 square
    assert fps[0].classification == "false_positive"


def test_no_false_positives_when_every_prediction_is_claimed():
    gt = _make_square_mask((50, 50), (10, 10), 10)
    pr = _make_square_mask((50, 50), (10, 10), 10)

    assert find_false_positives(gt, pr) == []


def test_contention_leaves_the_losing_gt_unmatched_even_with_high_best_iou():
    """Two GT cells both individually clear the IoU threshold against the
    *same* single prediction (a merged/under-segmented blob), but since
    there's only one prediction, only one GT can be claimed under
    one-to-one matching - exactly what compute_object_metrics would also
    report, and the scenario the old "best IoU >= 0.5" rule (without
    one-to-one matching) used to get wrong.

    Constructing this with real connected-component labels needs precise,
    degenerate geometry (the merged prediction must equal the two GT areas
    combined with zero connecting "bridge" pixels, or the IoU arithmetic
    can never let both clear 0.5 simultaneously) - so this test drives the
    matching primitive directly with hand-built label arrays instead of
    relying on ``skimage.measure.label`` to happen to produce that shape.
    """
    from segdiag.core.matching import _one_to_one_match

    gt_labels = np.zeros((10, 20), dtype=np.int32)
    gt_labels[:, :10] = 1  # GT 1, area 100
    gt_labels[:, 10:] = 2  # GT 2, area 100, touching GT 1 (would merge if
    #                        labeled together - hence building this by hand)

    pr_labels = np.zeros((10, 20), dtype=np.int32)
    pr_labels[:, :] = 1  # a single merged prediction covering both exactly,
    #                      area 200 - the only geometry where both GTs can
    #                      individually clear IoU 0.5 against one prediction

    matched_gt_ids, matched_pr_ids, pairs = _one_to_one_match(gt_labels, pr_labels)

    # Only one of the two GTs can be claimed - one prediction, one claim -
    # even though both touch it generously enough to individually qualify.
    assert matched_gt_ids == {1}
    assert matched_pr_ids == {1}
    assert pairs == {1: 1}


def test_false_positive_mean_intensity_uses_raw_array():
    gt = np.zeros((30, 30), dtype=np.uint8)
    pr = np.zeros((30, 30), dtype=np.uint8)
    pr[5:13, 5:13] = 1
    raw = np.full((30, 30), 17.0, dtype=np.float32)

    fps = find_false_positives(gt, pr, raw_arr=raw)

    assert len(fps) == 1
    assert fps[0].mean_intensity == 17.0


# --- 3D volume labeling (SEGDIAG_3D_INSTANCE_FIX.md regression) -------------


def test_match_instances_is_shape_agnostic_and_works_on_3d_volumes():
    """match_instances()/find_false_positives() themselves never assumed a
    2D input - passing a 3D volume should "just work" via skimage.measure.
    label's own default (full/26-connected) 3D connectivity. This is what
    lets segdiag.core.pipeline.collect() stack a sample's slices into one
    3D volume and match it once, instead of once per 2D slice.
    """
    shape = (4, 20, 20)
    gt_vol = np.zeros(shape, dtype=np.uint8)
    gt_vol[1:3, 5:10, 5:10] = 1  # one cell spanning z = 1, 2
    pr_vol = np.zeros(shape, dtype=np.uint8)
    pr_vol[1:3, 5:10, 5:10] = 1  # exact match

    matches = match_instances(gt_vol, pr_vol)

    assert len(matches) == 1
    assert matches[0].matched
    assert matches[0].volume == 2 * 5 * 5
    assert len(matches[0].centroid) == 3
    assert len(matches[0].bbox) == 6


def test_3d_volume_labeling_counts_cross_slice_cell_once_not_once_per_slice():
    """The regression scenario from SEGDIAG_3D_INSTANCE_FIX.md: a single
    cell spanning 4 z-slices (``gt[1:5, 5:10, 5:10] = 1``) must be counted
    as exactly 1 instance when the whole volume is labeled at once - the new
    (correct) behavior. Labeling each 2D slice independently and summing -
    the old, buggy per-slice pipeline behavior - instead counts it 4 times,
    once per slice it touches.
    """
    shape = (6, 20, 20)
    gt_vol = np.zeros(shape, dtype=np.uint8)
    gt_vol[1:5, 5:10, 5:10] = 1  # spans z = 1, 2, 3, 4 -> 4 slices
    pr_vol = np.zeros(shape, dtype=np.uint8)

    # Per-slice footprint is only 5x5=25 voxels (below MARS_MIN_VOLUME) even
    # though the combined 3D volume (100) clears it - disable the volume
    # filter here since this test is about 3D-vs-per-slice labeling, not
    # about the volume filter itself.
    matches_3d = match_instances(gt_vol, pr_vol, min_volume=None, max_volume=None)
    assert len(matches_3d) == 1
    assert matches_3d[0].volume == 4 * 5 * 5

    per_slice_counts = sum(
        len(match_instances(gt_vol[z], pr_vol[z], min_volume=None, max_volume=None))
        for z in range(shape[0])
    )
    assert per_slice_counts == 4


def test_matching_handles_many_disjoint_3d_instances_correctly():
    """Regression test for the bounding-box-cropped rewrite of
    _one_to_one_match()/_build_instance_matches(): on a full 3D volume, both
    now restrict every mask/overlap computation to each instance's own
    bounding box (via scipy.ndimage.find_objects / regionprops' prop.slice)
    instead of scanning the whole array per instance, which is what makes
    matching a stacked 3D volume fast instead of prohibitively slow. This
    scatters several same-sized cells at disjoint 3D locations - some
    matched, some blind FN, some hallucination FP - so a bounding-box
    indexing mistake (wrong axis order, off-by-one crop, a match leaking
    from a neighbouring instance's box) would show up as a wrong
    classification or IoU here.
    """
    shape = (20, 60, 60)
    gt = np.zeros(shape, dtype=np.uint8)
    pr = np.zeros(shape, dtype=np.uint8)

    # Matched pair, far from everything else.
    gt[1:5, 5:10, 5:10] = 1
    pr[1:5, 5:10, 5:10] = 1

    # A second matched pair, at a different z-range and xy location.
    gt[10:14, 30:35, 10:15] = 1
    pr[10:14, 30:35, 10:15] = 1

    # Blind FN: nothing overlapping anywhere near it.
    gt[5:9, 40:45, 40:45] = 1

    # Hallucination FP: a prediction with no GT partner nearby.
    pr[15:19, 5:10, 40:45] = 1

    matches = match_instances(gt, pr)
    fps = find_false_positives(gt, pr)

    assert len(matches) == 3
    tp_count = sum(1 for m in matches if m.classify() == "true_positive")
    blind_fn_count = sum(1 for m in matches if m.classify() == "blind_fn")
    assert tp_count == 2
    assert blind_fn_count == 1
    assert all(m.best_iou == 1.0 for m in matches if m.classify() == "true_positive")
    assert all(m.volume == 4 * 5 * 5 for m in matches)

    assert len(fps) == 1
    assert fps[0].volume == 4 * 5 * 5


# --- cc3d/MARS connectivity alignment (SEGDIAG_MARS_CONNECTIVITY.md) -------


def test_cc3d_to_skimage_connectivity_mapping():
    assert cc3d_to_skimage_connectivity(6) == 1
    assert cc3d_to_skimage_connectivity(18) == 2
    assert cc3d_to_skimage_connectivity(26) == 3
    assert cc3d_to_skimage_connectivity(None) is None
    with pytest.raises(ValueError):
        cc3d_to_skimage_connectivity(2)  # 不是合法的 cc3d 3D connectivity


def test_edge_adjacent_voxels_connect_at_18_but_not_at_6():
    """cc3d connectivity=18（MARS 用的）應該連通『邊相鄰但非面相鄰』的體素，
    但 connectivity=6 不應該。"""
    vol = np.zeros((3, 3, 3), dtype=np.uint8)
    vol[1, 1, 1] = 1
    vol[2, 2, 1] = 1  # 邊相鄰（兩個座標各差 1，非面、非純角）

    labels_6 = label(vol, connectivity=cc3d_to_skimage_connectivity(6))
    labels_18 = label(vol, connectivity=cc3d_to_skimage_connectivity(18))

    assert labels_6.max() == 2  # connectivity=6：視為兩個物件
    assert labels_18.max() == 1  # connectivity=18：視為同一個物件


# --- MARS 體積過濾對齊 (SEGDIAG_MARS_ALIGNMENT_COMPLETE.md Part 2) ----------


def test_filter_labels_by_volume_is_an_open_interval_at_both_boundaries():
    # Labels built directly by voxel count, not spatial connectivity -
    # filter_labels_by_volume only ever looks at np.bincount(labels), so this
    # is equivalent to (and much smaller than) building real 40/41/9999/10000
    # -voxel connected components.
    label_arr = np.array([1] * 40 + [2] * 41 + [3] * 9999 + [4] * 10000, dtype=np.int32)

    filtered = filter_labels_by_volume(label_arr, min_volume=40, max_volume=10000)

    # Exactly-40 and exactly-10000 are dropped (open interval, not <=/>=);
    # 41 and 9999 survive.
    assert set(np.unique(filtered)) == {0, 1, 2}
    assert (filtered == 1).sum() == 41
    assert (filtered == 2).sum() == 9999


def test_filter_labels_by_volume_relabels_surviving_component_to_one():
    # Three components (volumes 5 / 50 / 50000); only the middle one (50)
    # falls inside the default (40, 10000) window and must survive, renumbered
    # to 1 (not its original label, 2) - no gaps left for downstream
    # np.bincount/regionprops to waste space on.
    label_arr = np.array([1] * 5 + [2] * 50 + [3] * 50000, dtype=np.int32)

    filtered = filter_labels_by_volume(label_arr)

    assert set(np.unique(filtered)) == {0, 1}
    assert (filtered == 1).sum() == 50


def test_filter_labels_by_volume_none_on_both_sides_is_a_no_op():
    label_arr = np.array([1] * 5 + [2] * 50000, dtype=np.int32)

    filtered = filter_labels_by_volume(label_arr, min_volume=None, max_volume=None)

    assert np.array_equal(filtered, label_arr)


def test_match_instances_default_filters_out_small_volume_instances():
    # area 25 < MARS_MIN_VOLUME(40) - both default-filtered away entirely.
    gt = _make_square_mask((30, 30), (2, 2), 5)
    pr = _make_square_mask((30, 30), (2, 2), 5)

    assert match_instances(gt, pr) == []


def test_match_instances_default_filters_out_oversized_merged_blob():
    # A single connected blob of 101x101=10201 voxels clears
    # MARS_MIN_VOLUME(40) but exceeds MARS_MAX_VOLUME(10000) - over-
    # segmentation/merging artifacts this large shouldn't count as "one
    # cell" either, per SEGDIAG_MARS_ALIGNMENT_COMPLETE.md Part 2.5.
    gt = _make_square_mask((120, 120), (2, 2), 101)
    pr = gt.copy()

    assert match_instances(gt, pr) == []


def test_find_false_positives_respects_volume_filter():
    gt = np.zeros((30, 30), dtype=np.uint8)
    pr = _make_square_mask((30, 30), (2, 2), 5)  # area 25, below default min

    assert find_false_positives(gt, pr) == []
    assert find_false_positives(gt, pr, min_volume=None, max_volume=None) != []


# --- 位置寬鬆版配對 (SEGDIAG_MARS_ALIGNMENT_COMPLETE.md Part 3) -------------


def test_min_iou_none_default_preserves_tp_iou_threshold_behavior():
    """Not passing min_iou at all must behave identically to before this
    parameter existed - the merged_fn scenario from
    test_partial_overlap_is_merged_false_negative still fails to match.
    """
    gt = _make_square_mask((50, 50), (10, 10), 10)  # area 100
    pr = _make_square_mask((50, 50), (15, 15), 10)  # shifted, partial overlap

    matches = match_instances(gt, pr, min_volume=None, max_volume=None)

    assert not matches[0].matched
    assert matches[0].classify() == "merged_fn"


def test_min_iou_lenient_threshold_claims_a_merged_fn_as_matched():
    """The same partial-overlap pair, but with min_iou lowered to
    LOCATED_IOU_THRESHOLD (0.05, same "did it touch anything real" bar as
    BLIND_FN_IOU_THRESHOLD) - the prediction now claims the GT instance.
    """
    gt = _make_square_mask((50, 50), (10, 10), 10)
    pr = _make_square_mask((50, 50), (15, 15), 10)

    matches = match_instances(
        gt, pr, min_iou=LOCATED_IOU_THRESHOLD, min_volume=None, max_volume=None
    )

    assert matches[0].matched


def test_match_and_find_false_positives_forwards_min_iou():
    gt = _make_square_mask((50, 50), (10, 10), 10)
    pr = _make_square_mask((50, 50), (15, 15), 10)

    matches, fps, pairs = match_and_find_false_positives(
        gt, pr, min_iou=LOCATED_IOU_THRESHOLD, min_volume=None, max_volume=None
    )

    assert matches[0].matched
    assert fps == []
    assert pairs == {1: 1}
