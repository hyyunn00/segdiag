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

import numpy as np
import pandas as pd
import tifffile

from segdiag.checks._visualization import crop_with_padding, plot_zcontext_sample
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


class FnVisualizationCheck(Check):
    name = "fn-visualize"
    description = "Step 2: Visualize ghost/false-negative cells with 3D Z-context"
    # Renders up to --num-samples (default 20) full 4x5 figures, each
    # requiring its own raw/dark/gt/pred TIFF reads on top of collect()'s
    # own pass - opt-in only, so `segdiag run all` stays fast by default.
    default_enabled = False

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
                                crop_with_padding(tifffile.imread(str(s_raw)), target_bbox, pad=40)
                            )
                            sample_data["dark"].append(
                                crop_with_padding(tifffile.imread(str(s_dark)), target_bbox, pad=40)
                            )
                            sample_data["gt"].append(
                                crop_with_padding(tifffile.imread(str(s_gt)), target_bbox, pad=40)
                            )
                            sample_data["pr"].append(
                                crop_with_padding(tifffile.imread(str(s_pr)), target_bbox, pad=40)
                            )

                        sample_count += 1
                        fig = plot_zcontext_sample(
                            sample_data,
                            title=(
                                f"FN Sample #{sample_count}  "
                                f"(Model: {model_name} | Center Slice: {c_gt.name})"
                            ),
                            highlight_row="gt",
                        )

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
