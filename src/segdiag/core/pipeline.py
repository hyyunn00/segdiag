"""Single-pass dataset scan: the "Collect" stage of the collect/check/writer
pipeline.

:func:`collect` is now the *only* place that walks GT/prediction folders,
reads TIFFs, and runs :mod:`segdiag.core.matching`. It replaces the
per-step "scan folders -> read TIFFs -> match_instances ->
find_false_positives" logic that used to be duplicated six times (once per
diagnostic step), so the same GT/prediction pair is never re-matched twice
across a run and every check can share one cached table instead of
re-reading images.

Instance matching runs once per **3D volume** (all of a (sample, model)'s
z-slices stacked together via ``np.stack``), not once per 2D slice - a cell
spanning several z-slices must be counted once, not once per slice it
touches. See ``SEGDIAG_3D_INSTANCE_FIX.md`` for the bug this fixes and why.
Image-quality metrics remain a genuine per-slice measurement (SNR, focus,
etc. are properties of one 2D image) and are unaffected.

Image-quality metrics and FP root-cause classification are computed here via
pure functions imported from their respective checks
(:mod:`segdiag.checks.raw_image_quality`, :mod:`segdiag.checks.fp_root_cause`)
- those modules own the actual formulas; this module just calls them while
it already has the arrays in hand, so the ``raw-image-quality``/
``fp-root-cause`` checks themselves only need to aggregate and plot columns
that already exist in the collected tables.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import tifffile

from segdiag.core.io_utils import (
    extract_model_name,
    find_gt_folders,
    find_pred_folders,
    get_corresponding_file,
    list_tif_files,
    model_matches_exact,
    resolve_image_dir,
)
from segdiag.core.matching import (
    LOCATED_IOU_THRESHOLD,
    MARS_MAX_VOLUME,
    MARS_MIN_VOLUME,
    _build_false_positives,
    _build_instance_matches,
    _one_to_one_match,
    cc3d_to_skimage_connectivity,
    label_and_filter,
)
from segdiag.core.schema import ImageQualityRecord, InstanceRecord

logger = logging.getLogger(__name__)

_INSTANCE_COLUMNS = [f.name for f in fields(InstanceRecord)]
_QUALITY_COLUMNS = [f.name for f in fields(ImageQualityRecord)]


def _nearest_gt_distance(centroid: tuple, gt_matches: list) -> Optional[float]:
    if not gt_matches:
        return None
    return min(math.dist(centroid, m.centroid) for m in gt_matches)


def _volume_instance_rows(
    gt_vol: np.ndarray,
    pr_vol: np.ndarray,
    raw_vol: Optional[np.ndarray],
    *,
    dataset: str,
    sample: str,
    model: str,
    slice_names: List[str],
    connectivity: Optional[int] = None,
    min_volume: Optional[int] = MARS_MIN_VOLUME,
    max_volume: Optional[int] = MARS_MAX_VOLUME,
) -> List[dict]:
    """Build every ``InstanceRecord`` row for one (sample, model) pair,
    labeling the *entire* stacked 3D volume once - not once per 2D slice.
    This is the fix for the object-count inflation bug described in
    ``SEGDIAG_3D_INSTANCE_FIX.md``: a cell spanning several z-slices must be
    counted once, not once per slice it touches.

    ``gt_vol``/``pr_vol``/``raw_vol`` are ``(Z, Y, X)`` stacks built by
    :func:`collect` from the same ordered ``slice_names`` list, so index
    ``z`` in every array corresponds to ``slice_names[z]``. ``regionprops``
    on a 3D array returns 3-tuple centroids/6-tuple bboxes in ``(z, y, x)``
    order, which is why ``InstanceRecord``'s centroid/bbox fields split out
    a ``_z`` component alongside the existing ``_row``/``_col`` ones.

    Every instance's ``z_index``/``slice_name`` is derived from
    ``round(centroid_z)`` (clamped to the volume's bounds) - the single
    slice nearest that instance's 3D center, not "the slice it was found
    on" (a 3D instance has no such single slice).

    Labels/filters the volume once via
    :func:`segdiag.core.matching.label_and_filter`, then runs the shared
    one-to-one matcher *twice* on those same labels - once at the strict
    ``TP_IOU_THRESHOLD`` (drives ``classification``/``matched_instance_id``,
    unchanged from before) and once at ``LOCATED_IOU_THRESHOLD`` (drives
    ``located_matched``, see SEGDIAG_MARS_ALIGNMENT_COMPLETE.md Part 3) -
    instead of relabeling the whole 3D volume a second time just to change
    the IoU threshold.
    """
    from segdiag.checks.fp_root_cause import classify_fp_subtype, compute_local_background_stats

    rows: List[dict] = []
    num_z = gt_vol.shape[0]

    def _nearest_slice(z_coord: float) -> tuple:
        z_index = max(0, min(num_z - 1, int(round(z_coord))))
        return z_index, slice_names[z_index]

    gt_labels, pr_labels = label_and_filter(
        gt_vol,
        pr_vol,
        connectivity=cc3d_to_skimage_connectivity(connectivity),
        min_volume=min_volume,
        max_volume=max_volume,
    )

    matched_gt_ids, matched_pr_ids, pairs = _one_to_one_match(gt_labels, pr_labels)
    matches = (
        _build_instance_matches(gt_labels, pr_labels, matched_gt_ids, raw_vol)
        if gt_labels.max() > 0
        else []
    )
    fps = _build_false_positives(pr_labels, matched_pr_ids, raw_vol) if pr_labels.max() > 0 else []

    located_matched_gt_ids, located_matched_pr_ids, _ = _one_to_one_match(
        gt_labels, pr_labels, min_iou=LOCATED_IOU_THRESHOLD
    )

    for m in matches:
        z_index, slice_name = _nearest_slice(m.centroid[0])
        rows.append(
            asdict(
                InstanceRecord(
                    dataset=dataset,
                    sample=sample,
                    model=model,
                    slice_name=slice_name,
                    z_index=z_index,
                    role="gt",
                    instance_id=m.gt_id,
                    volume=m.volume,
                    mean_intensity=m.mean_intensity,
                    centroid_z=float(m.centroid[0]),
                    centroid_y=float(m.centroid[1]),
                    centroid_x=float(m.centroid[2]),
                    bbox_min_z=m.bbox[0],
                    bbox_min_row=m.bbox[1],
                    bbox_min_col=m.bbox[2],
                    bbox_max_z=m.bbox[3],
                    bbox_max_row=m.bbox[4],
                    bbox_max_col=m.bbox[5],
                    classification=m.classify(),
                    best_iou=m.best_iou,
                    matched_instance_id=pairs.get(m.gt_id),
                    located_matched=m.gt_id in located_matched_gt_ids,
                )
            )
        )

    for fp in fps:
        nearest_gt_distance = _nearest_gt_distance(fp.centroid, matches)
        background_mean, background_std = (
            compute_local_background_stats(raw_vol, gt_vol, pr_vol, fp.bbox)
            if raw_vol is not None
            else (None, None)
        )
        fp_subtype = classify_fp_subtype(
            fp_intensity=fp.mean_intensity,
            nearest_gt_distance=nearest_gt_distance,
            background_mean=background_mean,
            background_std=background_std,
        )
        z_index, slice_name = _nearest_slice(fp.centroid[0])
        rows.append(
            asdict(
                InstanceRecord(
                    dataset=dataset,
                    sample=sample,
                    model=model,
                    slice_name=slice_name,
                    z_index=z_index,
                    role="prediction",
                    instance_id=fp.pr_id,
                    volume=fp.volume,
                    mean_intensity=fp.mean_intensity,
                    centroid_z=float(fp.centroid[0]),
                    centroid_y=float(fp.centroid[1]),
                    centroid_x=float(fp.centroid[2]),
                    bbox_min_z=fp.bbox[0],
                    bbox_min_row=fp.bbox[1],
                    bbox_min_col=fp.bbox[2],
                    bbox_max_z=fp.bbox[3],
                    bbox_max_row=fp.bbox[4],
                    bbox_max_col=fp.bbox[5],
                    classification=fp.classification,
                    best_iou=None,
                    matched_instance_id=None,
                    fp_subtype=fp_subtype,
                    located_matched=fp.pr_id in located_matched_pr_ids,
                )
            )
        )

    return rows


def _slice_quality_row(
    gt_arr: np.ndarray,
    raw_arr: np.ndarray,
    *,
    dataset: str,
    sample: str,
    slice_name: str,
    z_index: int,
) -> dict:
    from segdiag.checks.raw_image_quality import compute_image_quality_metrics

    metrics = compute_image_quality_metrics(gt_arr, raw_arr)
    return asdict(
        ImageQualityRecord(
            dataset=dataset,
            sample=sample,
            slice_name=slice_name,
            z_index=z_index,
            source="raw",
            **metrics,
        )
    )


def collect(
    root: Path,
    *,
    mask_name: Optional[str] = None,
    raw_name: Optional[str] = None,
    sample_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    model_exact: Optional[str] = None,
    max_slices: Optional[int] = None,
    connectivity: Optional[int] = None,
    min_volume: Optional[int] = MARS_MIN_VOLUME,
    max_volume: Optional[int] = MARS_MAX_VOLUME,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single pass over the dataset: for every (sample, model) pair, stack
    all its z-slices into one 3D volume and run ``match_instances`` +
    ``find_false_positives`` on that volume *once* (not once per slice - see
    the module docstring), and return ``(instances_df, quality_df)``.

    If ``cache_path`` is given and both cache files already exist (and
    ``force_refresh`` is False), the cache is loaded instead of re-scanning.
    Every check downstream should read these DataFrames instead of touching
    TIFFs directly.

    ``mask_name``/``raw_name`` mirror ``find_gt_folders``/
    ``resolve_image_dir``'s own defaults: ``None`` uses the same
    ``analysis.py``-aligned auto-detection every step already relies on.

    ``model_filter`` is a case-insensitive *substring* match against the raw
    prediction-folder name (kept for backward compatibility). ``model_exact``
    is a stricter, comma-separated *exact* match against the already-
    extracted model name (see ``extract_model_name``) - use it when
    ``model_filter`` would also catch an unwanted sibling model whose name
    contains your term as a substring (e.g. ``model_exact="v12_unet_baseline"``
    won't also match a ``v12_unet_baseline_dark`` folder in another sample).
    Both filters apply together (AND) when both are given.

    ``max_slices`` caps the total number of (sample, slice) pairs read
    across the *entire* scan (unlimited by default). The pre-1.0 steps each
    had their own 100-150 slice cap applied independently per step/model;
    since collect() now reads every slice at most once for the whole run
    regardless of how many checks/models follow, one shared budget here
    replaces those six separate ones.

    ``connectivity``(cc3d/MARS 慣例：6/18/26，``None``=skimage 滿連通預設)決定
    連通元件分析時，兩個體素要多接近才算同一個物件——見
    ``core.matching.cc3d_to_skimage_connectivity`` 的轉換表。這會直接改變
    ``gt_count``/``pr_count``/``tp``/``fp``/``fn`` 等所有跟「物件數量」有關的數字，
    跟 ``SEGDIAG_3D_INSTANCE_FIX.md`` 描述的 2D/3D 計數問題是同一等級的影響範圍——
    改變這個值之後，務必用 ``--refresh-cache`` 重新掃描，不要沿用舊快取。

    ``min_volume``/``max_volume``(見 ``SEGDIAG_MARS_ALIGNMENT_COMPLETE.md``
    Part 2)在連通元件標記之後、one-to-one 配對之前，把體積不落在這個開區間
    內的連通元件濾掉——預設值(``40``/``10000``)是 MARS TH 標記的實務門檻，
    同樣會直接改變所有計數數字，改變後一樣要 ``--refresh-cache``。傳
    ``None`` 停用該側過濾。
    """
    inst_cache = cache_path.with_suffix(".instances.parquet") if cache_path else None
    qual_cache = cache_path.with_suffix(".quality.parquet") if cache_path else None

    if (
        inst_cache
        and qual_cache
        and not force_refresh
        and inst_cache.exists()
        and qual_cache.exists()
    ):
        return pd.read_parquet(inst_cache), pd.read_parquet(qual_cache)

    dataset_name = root.name
    gt_folders = find_gt_folders(root, mask_name, sample_filter=sample_filter)
    logger.info(
        "Collect: found %d GT folder(s) to scan under %s (this runs once; "
        "every check below reuses this pass instead of re-reading TIFFs)",
        len(gt_folders),
        root,
    )

    instance_rows: List[dict] = []
    quality_rows: List[dict] = []
    slices_processed = 0

    for gt_dir in gt_folders:
        if max_slices is not None and slices_processed >= max_slices:
            break

        parent = gt_dir.parent
        sample_name = parent.name
        img_dir = resolve_image_dir(gt_dir, raw_name=raw_name)
        pred_folders = find_pred_folders(parent, model_filter=model_filter)

        if not pred_folders:
            continue

        gt_files = list_tif_files(gt_dir)

        # GT/raw are read once per slice here (quality metrics don't depend
        # on which prediction folder/model is being evaluated) and cached
        # for reuse in the per-model loop below, instead of re-reading them
        # once per model.
        gt_arrays: Dict[str, np.ndarray] = {}
        raw_arrays: Dict[str, Optional[np.ndarray]] = {}

        for z_index, gtf in enumerate(gt_files):
            if max_slices is not None and slices_processed >= max_slices:
                break

            try:
                gt_arr = tifffile.imread(str(gtf))
            except Exception as exc:  # noqa: BLE001 - keep scanning on bad files
                logger.warning("Error reading %s: %s", gtf.name, exc)
                continue
            gt_arrays[gtf.name] = gt_arr

            raw_arr = None
            raw_f = get_corresponding_file(img_dir, gtf) if img_dir else None
            if raw_f is not None:
                try:
                    candidate = tifffile.imread(str(raw_f))
                    if candidate.shape == gt_arr.shape:
                        raw_arr = candidate
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error reading %s: %s", raw_f.name, exc)
            raw_arrays[gtf.name] = raw_arr

            if raw_arr is not None:
                quality_rows.append(
                    _slice_quality_row(
                        gt_arr,
                        raw_arr,
                        dataset=dataset_name,
                        sample=sample_name,
                        slice_name=gtf.name,
                        z_index=z_index,
                    )
                )

            slices_processed += 1
            if slices_processed % 50 == 0:
                logger.info("  ...%d slice(s) read so far", slices_processed)

        for p_dir in pred_folders:
            model_name = extract_model_name(img_dir.name if img_dir else "", p_dir.name)
            if not model_matches_exact(model_name, model_exact):
                continue
            # gt_dir.name/p_dir.name are just the shared subfolder basenames
            # (e.g. "Flatten_561_mask") - identical across every sample that
            # follows the same naming convention, so sample_name has to be
            # included here or this line is indistinguishable between
            # samples and looks like the same collect step repeating.
            logger.info(
                "Collecting: %s / %s vs %s (model=%s)",
                sample_name,
                gt_dir.name,
                p_dir.name,
                model_name,
            )

            # Read every prediction slice that has a same-shape GT
            # counterpart, then stack the whole (sample, model) run into one
            # 3D volume and match it *once* - matching per 2D slice here
            # would recount every cell that spans multiple z-slices once per
            # slice it touches (see SEGDIAG_3D_INSTANCE_FIX.md).
            pr_arrays: Dict[str, np.ndarray] = {}
            for gtf in gt_files:
                if gtf.name not in gt_arrays:
                    continue

                prf = get_corresponding_file(p_dir, gtf)
                if prf is None:
                    continue

                try:
                    pr_arr = tifffile.imread(str(prf))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error reading %s: %s", prf.name, exc)
                    continue

                if gt_arrays[gtf.name].shape != pr_arr.shape:
                    continue

                pr_arrays[gtf.name] = pr_arr

            common_files = [f for f in gt_files if f.name in gt_arrays and f.name in pr_arrays]
            if not common_files:
                continue
            if len(common_files) < len(gt_files):
                logger.warning(
                    "  %s / %s vs %s: only %d/%d slices had a matching "
                    "same-shape prediction - stacking those into the 3D "
                    "volume, skipping the rest",
                    sample_name,
                    gt_dir.name,
                    p_dir.name,
                    len(common_files),
                    len(gt_files),
                )

            gt_vol = np.stack([gt_arrays[f.name] for f in common_files], axis=0)
            pr_vol = np.stack([pr_arrays[f.name] for f in common_files], axis=0)
            raw_vol: Optional[np.ndarray] = None
            if all(raw_arrays.get(f.name) is not None for f in common_files):
                raw_vol = np.stack(
                    [cast(np.ndarray, raw_arrays[f.name]) for f in common_files], axis=0
                )

            instance_rows.extend(
                _volume_instance_rows(
                    gt_vol,
                    pr_vol,
                    raw_vol,
                    dataset=dataset_name,
                    sample=sample_name,
                    model=model_name,
                    slice_names=[f.name for f in common_files],
                    connectivity=connectivity,
                    min_volume=min_volume,
                    max_volume=max_volume,
                )
            )
            logger.info(
                "  ...matched %d slice(s) as one 3D volume for %s / %s vs %s",
                len(common_files),
                sample_name,
                gt_dir.name,
                p_dir.name,
            )

    instances_df = pd.DataFrame(instance_rows, columns=_INSTANCE_COLUMNS)
    quality_df = pd.DataFrame(quality_rows, columns=_QUALITY_COLUMNS)

    if inst_cache and qual_cache:
        inst_cache.parent.mkdir(parents=True, exist_ok=True)
        instances_df.to_parquet(inst_cache, index=False)
        quality_df.to_parquet(qual_cache, index=False)

    return instances_df, quality_df
