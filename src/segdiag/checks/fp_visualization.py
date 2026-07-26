"""FP visualization: 3D Z-context gallery for false-positive detections.

The mirror image of ``fn-visualize`` (Step 2, which renders missed GT
cells): this renders predictions that have no GT partner at all, so a
reviewer can visually judge whether a spurious detection is genuinely
concerning or not.

Unlike ``fn-visualize``, this check does *not* re-scan/re-match anything -
``collect()`` already ran :func:`segdiag.checks.fp_root_cause.classify_fp_subtype`
on every false positive and stored the result (plus its volume/intensity/
bbox/slice) in the instances table, so this check just reads that table to
pick which FPs to render, prioritizing ``hallucination_fp`` (looks like a
real cell but isn't - the most concerning bucket) over
``boundary_split_fp`` and ``noise_fp`` when ``--num-samples`` caps how many
galleries get generated. The only direct I/O left is reading the raw/dark/
gt/pred TIFFs needed to render the Z-context crop itself.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile

from segdiag.checks._visualization import crop_with_padding, plot_zcontext_sample
from segdiag.checks.base import Check
from segdiag.checks.fp_root_cause import FP_SUBTYPE_ORDER
from segdiag.core.io_utils import (
    extract_model_name,
    find_gt_folders,
    find_pred_folders,
    get_corresponding_file,
    list_tif_files,
    resolve_image_dir,
)
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)

#: Render the most concerning FP subtype first when --num-samples caps how
#: many galleries get generated - hallucination_fp is the one worth a
#: human's attention, noise_fp is the least (see fp_root_cause.py).
_SUBTYPE_PRIORITY = list(reversed(FP_SUBTYPE_ORDER))

_ResolvedFolders = Tuple[Path, Path, Path, Path, List[Path]]


class FpVisualizationCheck(Check):
    name = "fp-visualize"
    description = "Visualize false-positive detections (esp. hallucinations) with 3D Z-context"
    # Same reasoning as fn-visualize: renders up to --num-samples full 4x5
    # figures, each needing its own raw/dark/gt/pred TIFF reads - opt-in
    # only, so `segdiag run all` stays fast by default.
    default_enabled = False

    def run(
        self, instances: pd.DataFrame, quality: pd.DataFrame, args: argparse.Namespace
    ) -> List[ReportArtifact]:
        fp = instances[
            (instances["role"] == "prediction") & (instances["classification"] == "false_positive")
        ]
        if fp.empty:
            logger.warning("No false positives in the collected data - nothing to visualize.")
            return []

        root: Path = args.root
        mask_name: Optional[str] = getattr(args, "mask_name", None) or "Flatten_561_mask"
        raw_name: Optional[str] = getattr(args, "raw_name", None) or "Flatten_561"
        dark_name: str = getattr(args, "dark_name", None) or "Flatten_561_dark"
        num_samples: int = getattr(args, "num_samples", None) or 20

        artifacts: List[ReportArtifact] = []
        sample_count = 0
        # Candidates are grouped by subtype, so the same (sample, model)
        # pair repeats often - resolve its folders on disk at most once.
        folder_cache: Dict[Tuple[str, str], Optional[_ResolvedFolders]] = {}

        for row in self._ordered_candidates(fp):
            if sample_count >= num_samples:
                break

            key = (row["sample"], row["model"])
            if key not in folder_cache:
                folder_cache[key] = self._resolve_folders(
                    root, row["sample"], row["model"], mask_name, raw_name, dark_name
                )
            resolved = folder_cache[key]
            if resolved is None:
                continue
            gt_dir, raw_dir, dark_dir, pred_dir, gt_files = resolved

            z_idx = int(row["z_index"])
            if not (0 <= z_idx < len(gt_files)) or gt_files[z_idx].name != row["slice_name"]:
                continue
            if z_idx < 2 or z_idx > len(gt_files) - 3:
                continue  # too close to the volume's edge for a full Z-2..Z+2 window

            slices_paths = []
            all_exist = True
            for dz in (-2, -1, 0, 1, 2):
                cur_gtf = gt_files[z_idx + dz]
                cur_raw = get_corresponding_file(raw_dir, cur_gtf)
                cur_dark = get_corresponding_file(dark_dir, cur_gtf)
                cur_pr = get_corresponding_file(pred_dir, cur_gtf)
                if not (cur_raw and cur_dark and cur_pr):
                    all_exist = False
                    break
                slices_paths.append((cur_raw, cur_dark, cur_gtf, cur_pr))
            if not all_exist:
                continue

            target_bbox = (
                int(row["bbox_min_row"]),
                int(row["bbox_min_col"]),
                int(row["bbox_max_row"]),
                int(row["bbox_max_col"]),
            )

            try:
                sample_data: Dict[str, List[np.ndarray]] = {
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
            except Exception as exc:  # noqa: BLE001 - keep scanning on bad files
                logger.warning("Error reading Z-context around %s: %s", row["slice_name"], exc)
                continue

            sample_count += 1
            center_slice_name = gt_files[z_idx].name
            fig = plot_zcontext_sample(
                sample_data,
                title=(
                    f"FP Sample #{sample_count}  [{row['fp_subtype']}]  "
                    f"(Model: {row['model']} | Center Slice: {center_slice_name})"
                ),
                highlight_row="pr",
            )

            table = pd.DataFrame(
                [
                    {
                        "sample_id": sample_count,
                        "sample": row["sample"],
                        "model": row["model"],
                        "slice_name": center_slice_name,
                        "fp_subtype": row["fp_subtype"],
                        "volume": row["volume"],
                        "mean_intensity": row["mean_intensity"],
                        "bbox_min_row": target_bbox[0],
                        "bbox_min_col": target_bbox[1],
                        "bbox_max_row": target_bbox[2],
                        "bbox_max_col": target_bbox[3],
                    }
                ]
            )
            artifact_name = (
                f"fp_diagnosis_3d/fp_sample_{sample_count:02d}_{row['fp_subtype']}_"
                f"{row['sample']}_{row['model']}_{Path(center_slice_name).stem}"
            )
            artifacts.append(ReportArtifact(name=artifact_name, table=table, figure=fig))
            logger.info(
                "Generated 3D context plot for FP sample %d (%s)", sample_count, row["fp_subtype"]
            )

        if sample_count == 0:
            logger.warning("No FP samples could be rendered (missing TIFFs or edge slices).")
        else:
            logger.info("Done! Generated %d FP diagnostic plots.", sample_count)

        return artifacts

    @staticmethod
    def _ordered_candidates(fp: pd.DataFrame) -> List[dict]:
        """Return FP rows as plain dicts, grouped by subtype in
        ``_SUBTYPE_PRIORITY`` order (each group internally shuffled so
        repeated runs don't always render the same handful of samples from
        a large group), so the highest-priority subtype fills
        ``--num-samples`` slots first.
        """
        rows: List[dict] = []
        for subtype in _SUBTYPE_PRIORITY:
            group = fp[fp["fp_subtype"] == subtype].to_dict("records")
            random.shuffle(group)
            rows.extend(group)

        known = set(_SUBTYPE_PRIORITY)
        rows.extend(fp[~fp["fp_subtype"].isin(known)].to_dict("records"))
        return rows

    @staticmethod
    def _resolve_folders(
        root: Path,
        sample: str,
        model: str,
        mask_name: Optional[str],
        raw_name: Optional[str],
        dark_name: str,
    ) -> Optional[_ResolvedFolders]:
        """Locate the on-disk GT/raw/dark/prediction folders for one
        collected (sample, model) pair. ``sample_filter``/exact-model
        matching are used (not a bare ``root / sample`` join) since GT
        folders can live nested under ``root`` and several prediction
        folders in the same sample can share a naming substring.
        """
        gt_dirs = [
            d
            for d in find_gt_folders(root, mask_name, sample_filter=sample)
            if d.parent.name == sample
        ]
        if not gt_dirs:
            return None
        gt_dir = gt_dirs[0]
        parent = gt_dir.parent

        raw_dir = resolve_image_dir(gt_dir, raw_name=raw_name)
        dark_dir = parent / dark_name
        if raw_dir is None or not raw_dir.exists() or not dark_dir.exists():
            return None

        pred_dir = next(
            (
                p
                for p in find_pred_folders(parent, model_filter=model)
                if extract_model_name(raw_dir.name, p.name) == model
            ),
            None,
        )
        if pred_dir is None:
            return None

        gt_files = list_tif_files(gt_dir)
        if len(gt_files) < 5:
            return None

        return gt_dir, raw_dir, dark_dir, pred_dir, gt_files
