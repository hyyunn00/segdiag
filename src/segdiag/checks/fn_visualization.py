"""Step 2: False-Negative ("Ghost Cell") Visualization with 3D Z-Context.

Finds GT cells with best IoU < 0.05 (essentially undetected) and renders a
4x5 gallery per sample: rows are {Raw, Dark, GT, Prediction}, columns are the
center slice plus its two neighbouring slices on each side (Z-2 .. Z+2), so
that reviewers can visually inspect whether the model missed real signal.

Unlike every other check, this one does its own TIFF I/O instead of reading
``collect()``'s instances table: it needs dynamically-cropped 3D Z-context
around specific ghost cells, which is a sampled visualization task rather
than a statistical summary of the whole dataset - not something worth
forcing into the shared long-format table.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from segdiag.checks.base import Check
from segdiag.core.io_utils import (
    extract_model_name,
    find_gt_folders,
    find_pred_folders,
    get_corresponding_file,
    resolve_image_dir,
)
from segdiag.core.matching import find_fn_bboxes
from segdiag.core.report import ReportArtifact
from segdiag.core.reporting import log_scope

logger = logging.getLogger(__name__)


def _crop_with_padding(arr: np.ndarray, bbox, pad: int = 40) -> np.ndarray:
    """Crop ``arr`` to ``bbox`` expanded by ``pad`` pixels of context on each side."""
    min_row, min_col, max_row, max_col = bbox
    h, w = arr.shape
    r_start, r_end = max(0, min_row - pad), min(h, max_row + pad)
    c_start, c_end = max(0, min_col - pad), min(w, max_col + pad)
    return arr[r_start:r_end, c_start:c_end]


def _normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Rescale an intensity image to [0, 1] for matplotlib display."""
    arr_float = arr.astype(np.float32)
    min_v, max_v = arr_float.min(), arr_float.max()
    if max_v - min_v > 1e-6:
        return (arr_float - min_v) / (max_v - min_v)
    return arr_float


def _plot_context_sample(sample_data: dict, sample_id: int, z_name: str, model_name: str):
    """Render the 4x5 grid for a single FN sample and return the figure."""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    modalities = ["raw", "dark", "gt", "pr"]
    row_titles = ["Raw Image", "Dark Image", "Ground Truth", "Prediction"]
    col_titles = ["Z - 2", "Z - 1", "Center (Z)", "Z + 1", "Z + 2"]

    for r, mod in enumerate(modalities):
        for c in range(5):
            ax = axes[r, c]
            img = sample_data[mod][c]

            if mod in ("gt", "pr"):
                ax.imshow(img > 0, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(_normalize_for_display(img), cmap="gray")

            if mod == "gt" and c == 2:
                h, w = img.shape
                ax.plot([w // 2], [h // 2], "r+", markersize=25, markeredgewidth=3)

            if r == 0:
                ax.set_title(col_titles[c], fontsize=18, fontweight="bold")
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=18, fontweight="bold")

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.suptitle(
        f"FN Sample #{sample_id}  (Model: {model_name} | Center Slice: {z_name})",
        fontsize=22,
        y=1.02,
    )
    return fig


class FnVisualizationCheck(Check):
    name = "fn-visualize"
    description = "Step 2: Visualize ghost/false-negative cells with 3D Z-context"

    def run(
        self, instances: pd.DataFrame, quality: pd.DataFrame, args: argparse.Namespace
    ) -> List[ReportArtifact]:
        root: Path = args.root
        mask_name: Optional[str] = getattr(args, "mask_name", None) or "Flatten_561_mask"
        raw_name: Optional[str] = getattr(args, "raw_name", None) or "Flatten_561"
        dark_name: str = getattr(args, "dark_name", None) or "Flatten_561_dark"
        num_samples: int = getattr(args, "num_samples", None) or 20

        gt_folders = find_gt_folders(root, mask_name, sample_filter=args.sample)
        if not gt_folders:
            logger.error(
                "No folders named %s found (sample filter=%s).", mask_name, args.sample or "(none)"
            )
            return []

        log_scope(logger, gt_folders, args.sample, args.model)

        artifacts: List[ReportArtifact] = []
        sample_count = 0

        for gt_dir in gt_folders:
            if sample_count >= num_samples:
                break

            parent = gt_dir.parent
            raw_dir = resolve_image_dir(gt_dir, raw_name=raw_name)
            dark_dir = parent / dark_name

            pred_folders = find_pred_folders(parent, model_filter=args.model)
            if not pred_folders or raw_dir is None or not raw_dir.exists() or not dark_dir.exists():
                continue

            gt_files = sorted(
                f for f in gt_dir.iterdir() if f.is_file() and f.suffix in (".tif", ".tiff")
            )
            num_slices = len(gt_files)

            if num_slices < 5:
                continue

            for p_dir in pred_folders:
                if sample_count >= num_samples:
                    break

                model_name = extract_model_name(raw_dir.name, p_dir.name)
                logger.info("Searching FNs in: %s (model=%s)", parent.name, model_name)

                valid_indices = list(range(2, num_slices - 2))
                random.shuffle(valid_indices)

                for idx in valid_indices:
                    if sample_count >= num_samples:
                        break

                    all_exist = True
                    slices_paths = []

                    for dz in (-2, -1, 0, 1, 2):
                        z_idx = idx + dz
                        cur_gtf = gt_files[z_idx]
                        cur_raw = get_corresponding_file(raw_dir, cur_gtf)
                        cur_dark = get_corresponding_file(dark_dir, cur_gtf)
                        cur_pr = get_corresponding_file(p_dir, cur_gtf)

                        if not (cur_raw and cur_dark and cur_pr):
                            all_exist = False
                            break
                        slices_paths.append((cur_raw, cur_dark, cur_gtf, cur_pr))

                    if not all_exist:
                        continue

                    c_raw, c_dark, c_gt, c_pr = slices_paths[2]

                    try:
                        center_gt_arr = tifffile.imread(str(c_gt))
                        center_pr_arr = tifffile.imread(str(c_pr))
                        fn_bboxes = find_fn_bboxes(center_gt_arr, center_pr_arr, threshold=0.05)

                        if not fn_bboxes:
                            continue

                        target_bbox = random.choice(fn_bboxes)

                        sample_data: dict[str, list[np.ndarray]] = {
                            "raw": [],
                            "dark": [],
                            "gt": [],
                            "pr": [],
                        }
                        for s_raw, s_dark, s_gt, s_pr in slices_paths:
                            sample_data["raw"].append(
                                _crop_with_padding(tifffile.imread(str(s_raw)), target_bbox, pad=40)
                            )
                            sample_data["dark"].append(
                                _crop_with_padding(
                                    tifffile.imread(str(s_dark)), target_bbox, pad=40
                                )
                            )
                            sample_data["gt"].append(
                                _crop_with_padding(tifffile.imread(str(s_gt)), target_bbox, pad=40)
                            )
                            sample_data["pr"].append(
                                _crop_with_padding(tifffile.imread(str(s_pr)), target_bbox, pad=40)
                            )

                        sample_count += 1
                        fig = _plot_context_sample(sample_data, sample_count, c_gt.name, model_name)

                        table = pd.DataFrame(
                            [
                                {
                                    "sample_id": sample_count,
                                    "sample": parent.name,
                                    "model": model_name,
                                    "slice_name": c_gt.name,
                                    "bbox_min_row": target_bbox[0],
                                    "bbox_min_col": target_bbox[1],
                                    "bbox_max_row": target_bbox[2],
                                    "bbox_max_col": target_bbox[3],
                                }
                            ]
                        )
                        artifact_name = (
                            f"step2_fn_diagnosis_3d/fn_sample_{sample_count:02d}_{parent.name}_"
                            f"{model_name}_{c_gt.stem}"
                        )
                        artifacts.append(
                            ReportArtifact(name=artifact_name, table=table, figure=fig)
                        )
                        logger.info("Generated 3D context plot for sample %d", sample_count)

                    except Exception as exc:  # noqa: BLE001 - keep scanning on bad files
                        logger.warning("Error processing context around %s: %s", c_gt.name, exc)

        if sample_count == 0:
            logger.warning("No samples with IoU < 0.05 were found.")
        else:
            logger.info("Done! Generated %d diagnostic plots.", sample_count)

        return artifacts
