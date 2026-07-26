"""Single-pass dataset scan: the "Collect" stage of the collect/check/writer
pipeline.

:func:`collect` is now the *only* place that walks GT/prediction folders,
reads TIFFs, and runs :mod:`segdiag.core.matching`. It replaces the
per-step "scan folders -> read TIFFs -> match_instances ->
find_false_positives" logic that used to be duplicated six times (once per
diagnostic step), so the same GT/prediction pair is never re-matched twice
across a run and every check can share one cached table instead of
re-reading images.

Image-quality metrics and FP root-cause classification are computed here via
pure functions imported from their respective checks
(:mod:`segdiag.checks.raw_image_quality`, :mod:`segdiag.checks.fp_root_cause`)
- those modules own the actual formulas; this module just calls them once
per slice while it already has the arrays in hand, so the ``raw-image-
quality``/``fp-root-cause`` checks themselves only need to aggregate and
plot columns that already exist in the collected tables.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from segdiag.core.matching import match_and_find_false_positives
from segdiag.core.schema import ImageQualityRecord, InstanceRecord

logger = logging.getLogger(__name__)

_INSTANCE_COLUMNS = [f.name for f in fields(InstanceRecord)]
_QUALITY_COLUMNS = [f.name for f in fields(ImageQualityRecord)]


def _nearest_gt_distance(centroid: tuple, gt_matches: list) -> Optional[float]:
    if not gt_matches:
        return None
    return min(math.dist(centroid, m.centroid) for m in gt_matches)


def _background_stats(gt_arr: np.ndarray, raw_arr: Optional[np.ndarray]) -> tuple:
    if raw_arr is None:
        return None, None
    bg_mask = gt_arr == 0
    if not bg_mask.any():
        return None, None
    bg_values = raw_arr.astype(np.float64)[bg_mask]
    return float(bg_values.mean()), float(bg_values.std())


def _slice_instance_rows(
    gt_arr: np.ndarray,
    pr_arr: np.ndarray,
    raw_arr: Optional[np.ndarray],
    *,
    dataset: str,
    sample: str,
    model: str,
    slice_name: str,
    z_index: int,
) -> List[dict]:
    """Build every ``InstanceRecord`` row for one GT/prediction slice pair,
    using the shared :mod:`segdiag.core.matching` primitives.
    """
    from segdiag.checks.fp_root_cause import classify_fp_subtype

    rows: List[dict] = []

    gt_matches, fps, pairs = match_and_find_false_positives(gt_arr, pr_arr, raw_arr=raw_arr)

    for m in gt_matches:
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
                    centroid_y=float(m.centroid[0]),
                    centroid_x=float(m.centroid[1]),
                    bbox_min_row=m.bbox[0],
                    bbox_min_col=m.bbox[1],
                    bbox_max_row=m.bbox[2],
                    bbox_max_col=m.bbox[3],
                    classification=m.classify(),
                    best_iou=m.best_iou,
                    matched_instance_id=pairs.get(m.gt_id),
                )
            )
        )

    background_mean, background_std = _background_stats(gt_arr, raw_arr) if fps else (None, None)

    for fp in fps:
        nearest_gt_distance = _nearest_gt_distance(fp.centroid, gt_matches)
        fp_subtype = classify_fp_subtype(
            fp_volume=fp.volume,
            fp_intensity=fp.mean_intensity,
            nearest_gt_distance=nearest_gt_distance,
            background_mean=background_mean,
            background_std=background_std,
        )
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
                    centroid_y=float(fp.centroid[0]),
                    centroid_x=float(fp.centroid[1]),
                    bbox_min_row=fp.bbox[0],
                    bbox_min_col=fp.bbox[1],
                    bbox_max_row=fp.bbox[2],
                    bbox_max_col=fp.bbox[3],
                    classification=fp.classification,
                    best_iou=None,
                    matched_instance_id=None,
                    fp_subtype=fp_subtype,
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
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single pass over the dataset: for every GT/prediction slice pair,
    run ``match_instances`` + ``find_false_positives``, and return
    ``(instances_df, quality_df)``.

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

            for z_index, gtf in enumerate(gt_files):
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

                gt_arr = gt_arrays[gtf.name]
                if gt_arr.shape != pr_arr.shape:
                    continue

                instance_rows.extend(
                    _slice_instance_rows(
                        gt_arr,
                        pr_arr,
                        raw_arrays.get(gtf.name),
                        dataset=dataset_name,
                        sample=sample_name,
                        model=model_name,
                        slice_name=gtf.name,
                        z_index=z_index,
                    )
                )

                if (z_index + 1) % 50 == 0:
                    logger.info(
                        "  ...%d/%d slices matched for %s / %s vs %s",
                        z_index + 1,
                        len(gt_files),
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
