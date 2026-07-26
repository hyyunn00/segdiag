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
) -> Tuple[Set[int], Set[int]]:
    """Greedy one-to-one GT<->prediction matching - same algorithm,
    iteration order, and threshold as the lab's own
    ``metrics.compute_object_metrics``: walk predictions in
    label order, and for each one claim the first not-yet-claimed GT
    instance whose IoU meets ``min_iou``.

    Returns ``(matched_gt_ids, matched_pr_ids)``.
    """
    num_pr = int(pr_labels.max())
    if num_pr == 0:
        return set(), set()

    pr_areas = np.bincount(pr_labels.ravel())
    gt_areas = np.bincount(gt_labels.ravel())

    matched_gt_ids: Set[int] = set()
    matched_pr_ids: Set[int] = set()

    for pr_id in range(1, num_pr + 1):
        pr_mask = pr_labels == pr_id
        overlapping_gt_ids = np.unique(gt_labels[pr_mask])
        for gt_id in overlapping_gt_ids:
            if gt_id == 0 or gt_id in matched_gt_ids:
                continue
            gt_mask = gt_labels == gt_id
            intersection = np.logical_and(pr_mask, gt_mask).sum()
            union = pr_areas[pr_id] + gt_areas[gt_id] - intersection
            iou = intersection / union if union else 0.0
            if iou >= min_iou:
                matched_gt_ids.add(int(gt_id))
                matched_pr_ids.add(pr_id)
                break

    return matched_gt_ids, matched_pr_ids


def match_instances(
    gt_arr: np.ndarray,
    pr_arr: np.ndarray,
    raw_arr: Optional[np.ndarray] = None,
    connectivity: Optional[int] = None,
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

    Returns
    -------
    A list of :class:`InstanceMatch`, one per ground-truth instance.
    """
    gt_labels = label(gt_arr > 0, connectivity=connectivity)
    pr_labels = label(pr_arr > 0, connectivity=connectivity)

    num_gt = gt_labels.max()
    if num_gt == 0:
        return []

    pr_areas = np.bincount(pr_labels.ravel()) if pr_labels.max() > 0 else np.array([0])
    matched_gt_ids, _ = _one_to_one_match(gt_labels, pr_labels)

    if raw_arr is not None:
        props = regionprops(gt_labels, intensity_image=raw_arr)
    else:
        props = regionprops(gt_labels)

    results: List[InstanceMatch] = []
    for prop in props:
        gt_mask = gt_labels == prop.label
        overlapping_pr_ids = np.unique(pr_labels[gt_mask])

        max_iou = 0.0
        for pr_id in overlapping_pr_ids:
            if pr_id == 0:
                continue
            pr_mask = pr_labels == pr_id
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
) -> List[FalsePositive]:
    """Return every predicted instance that never got claimed as any GT's
    match under the shared one-to-one algorithm - i.e. exactly the
    predictions ``compute_object_metrics`` would count toward its FP total.

    Parameters mirror :func:`match_instances`; ``raw_arr`` (if given) is
    used to compute each false positive's mean intensity, letting you check
    whether spurious detections tend to be dim/small (noise) or
    real-looking (a genuine over-segmentation problem).
    """
    gt_labels = label(gt_arr > 0, connectivity=connectivity)
    pr_labels = label(pr_arr > 0, connectivity=connectivity)

    num_pr = pr_labels.max()
    if num_pr == 0:
        return []

    _, matched_pr_ids = _one_to_one_match(gt_labels, pr_labels)

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
