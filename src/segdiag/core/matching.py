"""Instance-level ground-truth vs. prediction matching.

This module is the single source of truth for connected-component instance
matching used by steps 1-6. Its TP/FN/FP membership is computed with the
**same one-to-one greedy matching algorithm, iteration order, and IoU
threshold** as ``compute_object_metrics`` in the lab's own ``metrics.py``.
That means a GT/prediction pair evaluated here and the same pair evaluated
with the lab's ``compute_object_metrics`` always agree on which cells are
TP/FN and which predictions are FP - the same "which objects did the model
get wrong" answer, just exposed at per-instance granularity here
(volume/intensity/depth per cell) instead of aggregate counts.

Two complementary views are exposed:

- :func:`match_instances` - one :class:`InstanceMatch` per **ground-truth**
  instance (was it matched, and to what best IoU). Drives recall-side
  diagnostics: which real cells did the model miss, and why.
- :func:`find_false_positives` - one :class:`FalsePositive` per
  **prediction** instance that never got claimed as any GT's match.
  Drives precision-side diagnostics: which detections are spurious (e.g.
  noise from over-fragmentation / over-segmentation), and what do they
  look like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops

#: IoU threshold at/above which a prediction claims a GT instance as a
#: match. Identical to the ``min_iou`` default in
#: ``metrics.compute_object_metrics``.
TP_IOU_THRESHOLD = 0.5

#: Best-IoU threshold below which an *unmatched* GT instance is considered
#: a "blind" miss (essentially zero overlap with anything), as opposed to
#: a "merged" miss where some prediction touches it but either the overlap
#: falls short of TP_IOU_THRESHOLD, or that overlap was already claimed by
#: another GT instance under the one-to-one matching rule.
BLIND_FN_IOU_THRESHOLD = 0.05

#: 位置寬鬆版一對一配對的 IoU 門檻（見 SEGDIAG_MARS_ALIGNMENT_COMPLETE.md
#: Part 3）：只要求「有真的碰到東西」（不是純雜訊/純幻覺），不要求
#: TP_IOU_THRESHOLD 那種嚴格形狀吻合度。刻意重用 BLIND_FN_IOU_THRESHOLD 的
#: 數值——同一把尺，不是另一個獨立的門檻家族。
LOCATED_IOU_THRESHOLD = BLIND_FN_IOU_THRESHOLD

#: MARS 3Dfilter（filter_annotation.py::AnnotationAnalyzer.get_points()）用來
#: 決定「這個連通元件算不算一顆細胞」的體積門檻（voxel 數，開區間）。上限
#: 10000 抄自原始判斷式 `if 25 < MarkerS < 10000:`；下限則不是抄程式碼裡的
#: 25——對 TH 染色，實務操作值是 40（來自實驗室經驗，非 MARS 原始碼），此處
#: 採用 40 作為預設值。若未來要支援其他標記，下限可能需要不同數字，呼叫端
#: 可覆寫，不要假設這個預設值放諸四海皆準。
MARS_MIN_VOLUME = 40  # TH 標記的實務下限（經驗值，非 MARS 程式碼字面值）
MARS_MAX_VOLUME = 10000  # 與 MARS 原始碼判斷式上限一致

#: cc3d(MARS 的 3Dfilter 用的套件)用鄰居數表示 3D 連通程度(6/18/26);
#: skimage.measure.label 用 1..ndim 表示。兩者對 3D 資料指向完全相同的三種
#: 物理連通程度,這裡是唯一負責轉換的地方——下游一律用 cc3d 的慣例
#: (6/18/26),不要在別處直接手寫 skimage 的 1/2/3,避免兩種數字系統混用
#: 卻沒有任何錯誤訊息提醒。
CC3D_TO_SKIMAGE_CONNECTIVITY_3D: Dict[int, int] = {6: 1, 18: 2, 26: 3}


def cc3d_to_skimage_connectivity(cc3d_connectivity: Optional[int]) -> Optional[int]:
    """把 cc3d 風格的 3D connectivity(6/18/26，MARS `3Dfilter` 用的慣例）
    轉成 ``skimage.measure.label`` 吃的 1/2/3。``None`` 原樣通過（等同
    skimage 自己的滿連通預設值）。
    """
    if cc3d_connectivity is None:
        return None
    try:
        return CC3D_TO_SKIMAGE_CONNECTIVITY_3D[cc3d_connectivity]
    except KeyError:
        raise ValueError(
            f"connectivity 必須是 {sorted(CC3D_TO_SKIMAGE_CONNECTIVITY_3D)} "
            f"其中之一（cc3d/MARS 慣例），收到 {cc3d_connectivity}"
        )


def filter_labels_by_volume(
    labels: np.ndarray,
    min_volume: Optional[int] = MARS_MIN_VOLUME,
    max_volume: Optional[int] = MARS_MAX_VOLUME,
) -> np.ndarray:
    """把體積不落在 ``(min_volume, max_volume)`` 開區間內的連通元件從
    ``labels`` 中清除（設回 0），其餘元件重新編號為連續的 1..N（不留
    「洞」，避免下游 ``np.bincount``/``regionprops`` 浪費空間處理不存在的
    label）。``None`` 代表該側不設限。

    在呼叫 :func:`match_instances`/:func:`find_false_positives`/
    :func:`match_and_find_false_positives` 之前，對 ``gt_labels``/
    ``pr_labels`` **都要**套用同一組門檻——MARS 的實際流程也是對稱套用
    （不分這個 marker 來自人工標註還是模型偵測），segdiag 對齊的是這個對稱
    行為。
    """
    if min_volume is None and max_volume is None:
        return labels

    areas = np.bincount(labels.ravel())
    keep = np.ones(len(areas), dtype=bool)
    keep[0] = False
    if min_volume is not None:
        keep &= areas > min_volume
    if max_volume is not None:
        keep &= areas < max_volume

    lut = np.zeros(len(areas), dtype=labels.dtype)
    lut[keep] = np.arange(1, keep.sum() + 1, dtype=labels.dtype)
    return lut[labels]


def label_and_filter(
    gt_arr: np.ndarray,
    pr_arr: np.ndarray,
    connectivity: Optional[int] = None,
    min_volume: Optional[int] = MARS_MIN_VOLUME,
    max_volume: Optional[int] = MARS_MAX_VOLUME,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label ``gt_arr``/``pr_arr`` and apply :func:`filter_labels_by_volume`
    to both - the single shared labeling step behind
    :func:`match_instances`/:func:`find_false_positives`/
    :func:`match_and_find_false_positives`, and reusable by
    :mod:`segdiag.core.pipeline` when it needs the same labeled/filtered
    arrays for a second one-to-one matching pass at a different IoU
    threshold (see :data:`LOCATED_IOU_THRESHOLD`) without paying for
    ``skimage.measure.label`` twice on the same 3D volume.
    """
    gt_labels = label(gt_arr > 0, connectivity=connectivity)
    pr_labels = label(pr_arr > 0, connectivity=connectivity)
    gt_labels = filter_labels_by_volume(gt_labels, min_volume, max_volume)
    pr_labels = filter_labels_by_volume(pr_labels, min_volume, max_volume)
    return gt_labels, pr_labels


#: Machine-readable classification codes returned by
#: :meth:`InstanceMatch.classify`, and their human-readable chart labels.
#: Kept as a separate mapping (rather than baking the pretty text into the
#: code itself) so DataFrame columns/groupby keys stay short, stable,
#: snake_case identifiers while plots can still show a descriptive label.
CLASSIFICATION_LABELS: Dict[str, str] = {
    "blind_fn": "Blind FN (best IoU < 0.05)",
    "merged_fn": "Merged FN (unmatched, best IoU >= 0.05)",
    "true_positive": "True Positive (matched)",
    "false_positive": "False Positive (unmatched prediction)",
}


@dataclass
class InstanceMatch:
    """One ground-truth instance: its best available IoU against any
    overlapping prediction, and whether it was actually claimed as a match
    under the shared one-to-one matching algorithm.
    """

    gt_id: int
    volume: int
    best_iou: float
    centroid: Tuple[float, ...]
    bbox: Tuple[int, ...]
    mean_intensity: Optional[float] = None
    #: True iff a prediction claimed this GT instance under the same
    #: one-to-one greedy algorithm as metrics.compute_object_metrics - i.e.
    #: this is exactly what that function would count as a TP. This can
    #: differ from ``best_iou >= TP_IOU_THRESHOLD`` when two GT instances
    #: both overlap the same prediction well enough, but only one can claim
    #: it (a real "contention" case worth flagging, not a bug).
    matched: bool = False

    @property
    def is_tp(self) -> bool:
        """Whether this GT instance counts as a True Positive - identical
        definition to ``compute_object_metrics``'s TP count.
        """
        return self.matched

    @property
    def is_fn(self) -> bool:
        """Whether this GT instance counts as a False Negative - identical
        definition to ``compute_object_metrics``'s FN count.
        """
        return not self.matched

    def classify(self, blind_threshold: float = BLIND_FN_IOU_THRESHOLD) -> str:
        """Classify this instance into one of the three GT-side buckets
        used throughout the report figures. Returns a short snake_case
        code; look it up in :data:`CLASSIFICATION_LABELS` for a
        human-readable chart label.
        """
        if self.matched:
            return "true_positive"
        if self.best_iou < blind_threshold:
            return "blind_fn"
        return "merged_fn"


@dataclass
class FalsePositive:
    """A predicted instance that never got claimed as any GT's match under
    the shared one-to-one matching algorithm - exactly what
    ``compute_object_metrics`` counts toward its FP total.
    """

    pr_id: int
    volume: int
    centroid: Tuple[float, ...]
    bbox: Tuple[int, ...]
    mean_intensity: Optional[float] = None

    @property
    def classification(self) -> str:
        return "false_positive"


def _mean_intensity(prop) -> float:
    """Return a region's mean intensity, compatible with both the modern
    ``intensity_mean`` attribute (skimage >= 0.26) and the older
    ``mean_intensity`` attribute it replaced.
    """
    value = getattr(prop, "intensity_mean", None)
    if value is None:
        value = prop.mean_intensity
    return float(value)


def _one_to_one_match(
    gt_labels: np.ndarray, pr_labels: np.ndarray, min_iou: float = TP_IOU_THRESHOLD
) -> Tuple[Set[int], Set[int], Dict[int, int]]:
    """Greedy one-to-one GT<->prediction matching - same algorithm,
    iteration order, and threshold as the lab's own
    ``metrics.compute_object_metrics``: walk predictions in
    label order, and for each one claim the first not-yet-claimed GT
    instance whose IoU meets ``min_iou``.

    Returns ``(matched_gt_ids, matched_pr_ids, pairs)``, where ``pairs``
    maps ``gt_id -> pr_id`` for every claimed pair - callers that need the
    pairing (not just membership) can read it straight off this single pass
    instead of re-running the algorithm a second time.

    Every mask/overlap computation below is restricted to a prediction
    instance's own bounding box (via ``scipy.ndimage.find_objects``, one
    ``O(N)`` pass) rather than indexing the full ``gt_labels``/``pr_labels``
    arrays per instance - since a match can only ever occur inside that
    prediction's own extent, this gives identical results at a fraction of
    the cost once ``label_arr`` is a whole stacked 3D volume instead of one
    2D slice (an ``== pr_id`` scan over the *entire* volume, repeated once
    per prediction, is what made 3D collection so much slower than the old
    per-slice matching before this optimization).
    """
    num_pr = int(pr_labels.max())
    if num_pr == 0:
        return set(), set(), {}

    pr_areas = np.bincount(pr_labels.ravel())
    gt_areas = np.bincount(gt_labels.ravel())
    pr_bboxes = ndimage.find_objects(pr_labels)

    matched_gt_ids: Set[int] = set()
    matched_pr_ids: Set[int] = set()
    pairs: Dict[int, int] = {}

    for pr_id in range(1, num_pr + 1):
        bbox = pr_bboxes[pr_id - 1]
        if bbox is None:  # no voxels with this label (can't happen for a
            continue  # dense 1..num_pr range, but find_objects allows gaps)

        pr_crop = pr_labels[bbox]
        gt_crop = gt_labels[bbox]
        pr_mask = pr_crop == pr_id
        overlapping_gt_ids = np.unique(gt_crop[pr_mask])
        for gt_id in overlapping_gt_ids:
            if gt_id == 0 or gt_id in matched_gt_ids:
                continue
            gt_mask = gt_crop == gt_id
            intersection = np.logical_and(pr_mask, gt_mask).sum()
            union = pr_areas[pr_id] + gt_areas[gt_id] - intersection
            iou = intersection / union if union else 0.0
            if iou >= min_iou:
                matched_gt_ids.add(int(gt_id))
                matched_pr_ids.add(pr_id)
                pairs[int(gt_id)] = pr_id
                break

    return matched_gt_ids, matched_pr_ids, pairs


def match_instances(
    gt_arr: np.ndarray,
    pr_arr: np.ndarray,
    raw_arr: Optional[np.ndarray] = None,
    connectivity: Optional[int] = None,
    min_iou: Optional[float] = None,
    min_volume: Optional[int] = MARS_MIN_VOLUME,
    max_volume: Optional[int] = MARS_MAX_VOLUME,
) -> List[InstanceMatch]:
    """Label connected components in ``gt_arr`` and ``pr_arr`` and compute,
    for every ground-truth instance, its best (maximum) IoU against any
    overlapping predicted instance, plus whether it was actually claimed as
    a match under the shared one-to-one algorithm (see module docstring).

    Parameters
    ----------
    gt_arr:
        Binary (or label) ground-truth array for a single slice/volume.
    pr_arr:
        Binary (or label) prediction array, same shape as ``gt_arr``.
    raw_arr:
        Optional raw intensity image (same shape) used to compute the mean
        intensity of each GT instance. When omitted, ``mean_intensity`` is
        left as ``None`` on every result.
    connectivity:
        Connectivity used for ``skimage.measure.label``. Defaults to
        ``None`` (skimage's own default, i.e. full/maximal connectivity -
        8-connected in 2D, 26-connected in 3D), matching the default used
        by ``compute_object_metrics`` in the lab's own ``metrics.py``.
    min_iou:
        IoU threshold a prediction must clear to claim a GT instance under
        the one-to-one matching algorithm. ``None`` (default) keeps the
        existing behavior of :data:`TP_IOU_THRESHOLD` - pass
        :data:`LOCATED_IOU_THRESHOLD` for the position-lenient count
        agreement used by ``checks.cell_count_agreement``.
    min_volume, max_volume:
        Open-interval voxel-count bounds applied to both ``gt_arr`` and
        ``pr_arr`` after labeling (see :func:`filter_labels_by_volume`).
        Default to the MARS TH-marker thresholds (:data:`MARS_MIN_VOLUME`/
        :data:`MARS_MAX_VOLUME`) - pass ``None`` for either side to see the
        raw, unfiltered connected-component counts.

    Returns
    -------
    A list of :class:`InstanceMatch`, one per ground-truth instance.
    """
    gt_labels, pr_labels = label_and_filter(gt_arr, pr_arr, connectivity, min_volume, max_volume)

    if gt_labels.max() == 0:
        return []

    matched_gt_ids, _, _ = _one_to_one_match(
        gt_labels, pr_labels, min_iou if min_iou is not None else TP_IOU_THRESHOLD
    )
    return _build_instance_matches(gt_labels, pr_labels, matched_gt_ids, raw_arr)


def _build_instance_matches(
    gt_labels: np.ndarray,
    pr_labels: np.ndarray,
    matched_gt_ids: Set[int],
    raw_arr: Optional[np.ndarray],
) -> List[InstanceMatch]:
    """Build one :class:`InstanceMatch` per GT instance from already-labeled
    arrays and an already-computed ``matched_gt_ids`` set - the shared body
    of :func:`match_instances`, factored out so
    :func:`match_and_find_false_positives` can reuse it without labeling or
    running :func:`_one_to_one_match` a second time.

    Like :func:`_one_to_one_match`, every mask/overlap computation is
    restricted to the GT instance's own bounding box - ``regionprops``
    already computes this for free as ``prop.slice``/``prop.image``, so
    reusing them (instead of an ``gt_labels == prop.label`` scan of the
    *entire* array per instance) is what keeps this cheap on a full 3D
    volume instead of one 2D slice.
    """
    pr_areas = np.bincount(pr_labels.ravel()) if pr_labels.max() > 0 else np.array([0])

    if raw_arr is not None:
        props = regionprops(gt_labels, intensity_image=raw_arr)
    else:
        props = regionprops(gt_labels)

    results: List[InstanceMatch] = []
    for prop in props:
        gt_mask = prop.image  # already cropped to prop.slice's bounding box
        pr_crop = pr_labels[prop.slice]
        overlapping_pr_ids = np.unique(pr_crop[gt_mask])

        max_iou = 0.0
        for pr_id in overlapping_pr_ids:
            if pr_id == 0:
                continue
            pr_mask = pr_crop == pr_id
            intersection = np.logical_and(pr_mask, gt_mask).sum()
            union = prop.area + pr_areas[pr_id] - intersection
            iou = intersection / union
            if iou > max_iou:
                max_iou = iou

        results.append(
            InstanceMatch(
                gt_id=prop.label,
                volume=int(prop.area),
                best_iou=float(max_iou),
                centroid=tuple(prop.centroid),
                bbox=tuple(prop.bbox),
                mean_intensity=_mean_intensity(prop) if raw_arr is not None else None,
                matched=prop.label in matched_gt_ids,
            )
        )

    return results


def find_false_positives(
    gt_arr: np.ndarray,
    pr_arr: np.ndarray,
    raw_arr: Optional[np.ndarray] = None,
    connectivity: Optional[int] = None,
    min_iou: Optional[float] = None,
    min_volume: Optional[int] = MARS_MIN_VOLUME,
    max_volume: Optional[int] = MARS_MAX_VOLUME,
) -> List[FalsePositive]:
    """Return every predicted instance that never got claimed as any GT's
    match under the shared one-to-one algorithm - i.e. exactly the
    predictions ``compute_object_metrics`` would count toward its FP total.

    Parameters mirror :func:`match_instances`; ``raw_arr`` (if given) is
    used to compute each false positive's mean intensity, letting you check
    whether spurious detections tend to be dim/small (noise) or
    real-looking (a genuine over-segmentation problem).
    """
    gt_labels, pr_labels = label_and_filter(gt_arr, pr_arr, connectivity, min_volume, max_volume)

    if pr_labels.max() == 0:
        return []

    _, matched_pr_ids, _ = _one_to_one_match(
        gt_labels, pr_labels, min_iou if min_iou is not None else TP_IOU_THRESHOLD
    )
    return _build_false_positives(pr_labels, matched_pr_ids, raw_arr)


def _build_false_positives(
    pr_labels: np.ndarray,
    matched_pr_ids: Set[int],
    raw_arr: Optional[np.ndarray],
) -> List[FalsePositive]:
    """Build the list of unclaimed predicted instances from an
    already-labeled ``pr_labels`` array and an already-computed
    ``matched_pr_ids`` set - the shared body of :func:`find_false_positives`,
    factored out for reuse by :func:`match_and_find_false_positives`.
    """
    if raw_arr is not None:
        props = regionprops(pr_labels, intensity_image=raw_arr)
    else:
        props = regionprops(pr_labels)

    results: List[FalsePositive] = []
    for prop in props:
        if prop.label in matched_pr_ids:
            continue
        results.append(
            FalsePositive(
                pr_id=prop.label,
                volume=int(prop.area),
                centroid=tuple(prop.centroid),
                bbox=tuple(prop.bbox),
                mean_intensity=_mean_intensity(prop) if raw_arr is not None else None,
            )
        )

    return results


def match_and_find_false_positives(
    gt_arr: np.ndarray,
    pr_arr: np.ndarray,
    raw_arr: Optional[np.ndarray] = None,
    connectivity: Optional[int] = None,
    min_iou: Optional[float] = None,
    min_volume: Optional[int] = MARS_MIN_VOLUME,
    max_volume: Optional[int] = MARS_MAX_VOLUME,
) -> Tuple[List[InstanceMatch], List[FalsePositive], Dict[int, int]]:
    """Combined, single-pass equivalent of calling :func:`match_instances`
    and :func:`find_false_positives` back to back on the same arrays, plus
    the ``gt_id -> pr_id`` pairing for every claimed match.

    Calling those two functions separately labels ``gt_arr``/``pr_arr`` and
    re-runs :func:`_one_to_one_match` twice each (once per function) - fine
    in isolation, but ``collect()`` needs all three results (matches, FPs,
    *and* the pairing) for every slice, which used to mean labeling twice
    and running the greedy matcher three times per slice (once inside each
    of ``match_instances``/``find_false_positives``, plus a third time in
    pipeline.py just to recover the pairing). This does it once.
    """
    gt_labels, pr_labels = label_and_filter(gt_arr, pr_arr, connectivity, min_volume, max_volume)

    matched_gt_ids, matched_pr_ids, pairs = _one_to_one_match(
        gt_labels, pr_labels, min_iou if min_iou is not None else TP_IOU_THRESHOLD
    )

    matches = (
        _build_instance_matches(gt_labels, pr_labels, matched_gt_ids, raw_arr)
        if gt_labels.max() > 0
        else []
    )
    false_positives = (
        _build_false_positives(pr_labels, matched_pr_ids, raw_arr) if pr_labels.max() > 0 else []
    )

    return matches, false_positives, pairs


def find_fn_bboxes(
    gt_arr: np.ndarray,
    pr_arr: np.ndarray,
    threshold: float = BLIND_FN_IOU_THRESHOLD,
    connectivity: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    """Convenience wrapper returning only the bounding boxes of GT instances
    whose best IoU falls below ``threshold`` (used by the visualization step
    to locate "ghost cell" crops - cells with essentially no overlap at
    all, a stricter net than the general FN definition above).
    """
    matches = match_instances(gt_arr, pr_arr, connectivity=connectivity)
    return [m.bbox for m in matches if m.best_iou < threshold]
