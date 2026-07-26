"""Unit tests for segdiag.core.matching.

These tests use small synthetic 2D arrays instead of real microscopy TIFFs,
so they run instantly and require no test data.
"""

import numpy as np

from segdiag.core.matching import find_false_positives, find_fn_bboxes, match_instances


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
    gt = _make_square_mask((20, 20), (2, 2), 5)
    pr = _make_square_mask((20, 20), (2, 2), 5)
    raw = np.full((20, 20), 42.0, dtype=np.float32)

    matches = match_instances(gt, pr, raw_arr=raw)

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

    matches_3d = match_instances(gt_vol, pr_vol)
    assert len(matches_3d) == 1
    assert matches_3d[0].volume == 4 * 5 * 5

    per_slice_counts = sum(len(match_instances(gt_vol[z], pr_vol[z])) for z in range(shape[0]))
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
