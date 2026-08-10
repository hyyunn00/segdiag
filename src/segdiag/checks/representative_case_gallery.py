"""Representative case gallery for report Figure 5.3.2: 3 FN patterns + 2 FP
patterns, each selected by an objective, reproducible rule (not eyeballed by
a reviewer), then fixed-seed sampled. See
``SEGDIAG_REPRESENTATIVE_CASE_GALLERY.md`` for the full spec this
implements (filter criteria per pattern, layout design, validation
checklist).

Unlike most checks, this one re-reads raw/dark/GT/prediction TIFFs to render
image panels - architecturally closer to ``fn_visualization.py``/
``fp_visualization.py`` (which also re-read TIFFs for dynamically-cropped
Z-context) than the purely tabular checks.

Two deliberate departures from the original spec, both because the actual
schema/table shape turned out to differ from what the spec assumed (per its
own instruction to verify current state before coding, not assume the
described fields already/don't already exist):

- ``z_min``/``z_max`` aren't separate fields - ``InstanceRecord`` already had
  ``bbox_min_z``/``bbox_max_z`` (half-open, like all its other bbox fields),
  reused here instead of adding a duplicate pair of fields.
- The spec's ``z_discontinuity`` filter joins a GT row against a *separate*
  "prediction" row via ``matched_instance_id`` to read the matched
  prediction's Z-span. That row doesn't exist: a prediction claimed by a GT
  match never gets its own "prediction"-role row in this table (only
  unmatched predictions/false positives do). Instead,
  ``core.pipeline._volume_instance_rows`` now stores the matched
  prediction's Z-span directly on the GT row as
  ``matched_pred_z_span`` (see ``core.schema.InstanceRecord``), so no join
  is needed at all.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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
    list_tif_files,
    resolve_image_dir,
)
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_N_PER_PATTERN = 3
DEFAULT_VOXEL_SIZE_UM = 1.82
SCALE_BAR_LENGTH_UM = 10.0

PATTERNS = [
    "contour_underestimate",
    "no_response",
    "z_discontinuity",
    "background_noise",
    "missing_gt_annotation",
]

#: (gt_dir, raw_dir, dark_dir, pred_dir, sorted gt tif files) for one
#: (sample, model) pair - mirrors ``FpVisualizationCheck._resolve_folders``.
_ResolvedFolders = Tuple[Path, Path, Path, Path, List[Path]]


def _select_candidates(instances: pd.DataFrame, pattern: str) -> pd.DataFrame:
    if pattern == "contour_underestimate":
        return instances[
            (instances["role"] == "gt")
            & (instances["classification"] == "true_positive")
            & (instances["best_iou"] >= 0.50)
            & (instances["best_iou"] < 0.60)
        ]
    if pattern == "no_response":
        return instances[(instances["role"] == "gt") & (instances["classification"] == "blind_fn")]
    if pattern == "background_noise":
        return instances[
            (instances["role"] == "prediction") & (instances["fp_subtype"] == "noise_fp")
        ]
    if pattern == "z_discontinuity":
        return _select_z_discontinuity_candidates(instances)
    if pattern == "missing_gt_annotation":
        return _select_missing_gt_candidates(instances)
    raise ValueError(f"unknown pattern: {pattern}")


def _select_z_discontinuity_candidates(instances: pd.DataFrame) -> pd.DataFrame:
    """GT cells the model *did* claim as a strict match, but only via a
    prediction spanning a single Z-slice, despite the GT cell itself
    spanning >= 3 - i.e. the model responded to just a thin cross-section of
    a tall cell rather than the whole thing.

    Bbox fields are half-open (``bbox_max_* `` is exclusive, matching
    ``skimage.measure.regionprops``), so a Z-span is ``max - min`` with no
    ``+1`` - a single-slice instance already has ``bbox_max_z == bbox_min_z + 1``.
    """
    gt = instances[
        (instances["role"] == "gt")
        & (instances["classification"] == "true_positive")
        & instances["matched_pred_z_span"].notna()
    ]
    gt_z_span = gt["bbox_max_z"] - gt["bbox_min_z"]
    return gt[(gt["matched_pred_z_span"] == 1) & (gt_z_span >= 3)]


def _select_missing_gt_candidates(instances: pd.DataFrame) -> pd.DataFrame:
    """``hallucination_fp`` predictions with local contrast strong enough
    (>= 2 sigma above the local background, stricter than the 1.5 sigma
    ``fp_root_cause`` uses for its looser ``noise_fp`` rule) that they look
    like they should have had a GT annotation.
    """
    fp = instances[
        (instances["role"] == "prediction") & (instances["fp_subtype"] == "hallucination_fp")
    ]
    return fp[fp["background_contrast_sigma"] >= 2.0]


def _sample_cases(candidates: pd.DataFrame, pattern: str, n: int, seed: int) -> pd.DataFrame:
    """Fixed-seed sample via ``pandas.DataFrame.sample(random_state=seed)`` -
    deliberately *not* the global ``random`` module ``fn_visualization.py``
    uses, so repeated calls (any pattern, any order) are independently
    reproducible and never interfere with each other's draws.
    """
    if candidates.empty:
        logger.warning("No candidates found for pattern=%s - skipping.", pattern)
        return candidates
    n = min(n, len(candidates))
    return candidates.sample(n=n, random_state=seed)


def _resolve_intensity_window(
    raw_arrays: List[np.ndarray], vmin: Optional[float], vmax: Optional[float]
) -> Tuple[float, float]:
    """One shared display window for every panel this run renders (Sec
    5.3): the 1st/99th percentile of every raw pixel actually sampled this
    run, unless the caller pinned one or both ends explicitly.
    """
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    if not raw_arrays:
        return (float(vmin) if vmin is not None else 0.0), (
            float(vmax) if vmax is not None else 1.0
        )

    pooled = np.concatenate([a.astype(np.float64).ravel() for a in raw_arrays])
    lo = float(vmin) if vmin is not None else float(np.percentile(pooled, 1))
    hi = float(vmax) if vmax is not None else float(np.percentile(pooled, 99))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _add_scale_bar(ax, voxel_size_um: float, length_um: float = SCALE_BAR_LENGTH_UM) -> None:
    """Draw a ``length_um``-long scale bar near the bottom-right corner.
    Length in pixels is computed from ``voxel_size_um`` at call time - never
    hardcode a pixel length, it silently goes wrong the next time the
    microscope/objective (and therefore voxel size) changes.
    """
    length_px = length_um / voxel_size_um
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    width = abs(xlim[1] - xlim[0])
    height = abs(ylim[1] - ylim[0])
    if length_px > width:
        length_px = width * 0.5  # degrade gracefully on tiny crops
    x_end = xlim[1] - 0.05 * width
    x_start = x_end - length_px
    y = max(ylim) - 0.08 * height
    ax.plot([x_start, x_end], [y, y], color="yellow", linewidth=2, solid_capstyle="butt")
    ax.text(
        (x_start + x_end) / 2,
        y - 0.03 * height,
        f"{length_um:.0f} µm",
        color="yellow",
        fontsize=7,
        ha="center",
        va="top",
    )


def _add_source_label(ax, case: pd.Series) -> None:
    label = f"{case['sample']} / z={int(case['z_index'])} / ({case['centroid_y']:.0f}, {case['centroid_x']:.0f})"
    ax.text(
        0.02,
        0.02,
        label,
        transform=ax.transAxes,
        fontsize=7,
        color="yellow",
        va="bottom",
        ha="left",
        bbox=dict(facecolor="black", alpha=0.5, pad=1),
    )


def _load_general_case_images(
    case: pd.Series, folders: _ResolvedFolders, pad: int = 40
) -> Optional[Dict[str, np.ndarray]]:
    """Read+crop the single flagged slice's raw/dark/gt/pred arrays for the
    2x4 general grid (Sec 5.1). Returns ``None`` if the case's recorded
    slice/bbox can no longer be resolved on disk (e.g. stale cache).
    """
    gt_dir, raw_dir, dark_dir, pred_dir, gt_files = folders
    z_idx = int(case["z_index"])
    if not (0 <= z_idx < len(gt_files)) or gt_files[z_idx].name != case["slice_name"]:
        return None

    gtf = gt_files[z_idx]
    raw_f = get_corresponding_file(raw_dir, gtf)
    dark_f = get_corresponding_file(dark_dir, gtf)
    pred_f = get_corresponding_file(pred_dir, gtf)
    if not (raw_f and dark_f and pred_f):
        return None

    bbox = (
        int(case["bbox_min_row"]),
        int(case["bbox_min_col"]),
        int(case["bbox_max_row"]),
        int(case["bbox_max_col"]),
    )
    try:
        return {
            "raw": crop_with_padding(tifffile.imread(str(raw_f)), bbox, pad=pad),
            "dark": crop_with_padding(tifffile.imread(str(dark_f)), bbox, pad=pad),
            "gt": crop_with_padding(tifffile.imread(str(gtf)), bbox, pad=pad),
            "pr": crop_with_padding(tifffile.imread(str(pred_f)), bbox, pad=pad),
        }
    except Exception as exc:  # noqa: BLE001 - keep scanning on bad files
        logger.warning("Error reading images around %s: %s", case["slice_name"], exc)
        return None


def _load_zcontext_images(
    case: pd.Series, folders: _ResolvedFolders, pad: int = 40
) -> Optional[Dict[str, List[np.ndarray]]]:
    """Read+crop the Z-2..Z+2 raw/dark/gt/pred arrays for the z_discontinuity
    layout (Sec 5.2) - same shape ``fn_visualization.py``/
    ``fp_visualization.py`` build, reused (not reimplemented) via
    ``_visualization.plot_zcontext_sample``.
    """
    gt_dir, raw_dir, dark_dir, pred_dir, gt_files = folders
    z_idx = int(case["z_index"])
    if not (0 <= z_idx < len(gt_files)) or gt_files[z_idx].name != case["slice_name"]:
        return None
    if z_idx < 2 or z_idx > len(gt_files) - 3:
        return None  # too close to the volume's edge for a full Z-2..Z+2 window

    bbox = (
        int(case["bbox_min_row"]),
        int(case["bbox_min_col"]),
        int(case["bbox_max_row"]),
        int(case["bbox_max_col"]),
    )
    sample_data: Dict[str, List[np.ndarray]] = {"raw": [], "dark": [], "gt": [], "pr": []}
    try:
        for dz in (-2, -1, 0, 1, 2):
            gtf = gt_files[z_idx + dz]
            raw_f = get_corresponding_file(raw_dir, gtf)
            dark_f = get_corresponding_file(dark_dir, gtf)
            pred_f = get_corresponding_file(pred_dir, gtf)
            if not (raw_f and dark_f and pred_f):
                return None
            sample_data["raw"].append(crop_with_padding(tifffile.imread(str(raw_f)), bbox, pad=pad))
            sample_data["dark"].append(
                crop_with_padding(tifffile.imread(str(dark_f)), bbox, pad=pad)
            )
            sample_data["gt"].append(crop_with_padding(tifffile.imread(str(gtf)), bbox, pad=pad))
            sample_data["pr"].append(crop_with_padding(tifffile.imread(str(pred_f)), bbox, pad=pad))
    except Exception as exc:  # noqa: BLE001 - keep scanning on bad files
        logger.warning("Error reading Z-context around %s: %s", case["slice_name"], exc)
        return None
    return sample_data


def _render_general_grid(
    loaded_cases: List[Tuple[pd.Series, Dict[str, np.ndarray]]],
    vmin: float,
    vmax: float,
    voxel_size_um: float,
) -> Optional[plt.Figure]:
    """Rows = cases, columns = [Raw | Dark Sectioning | GT Overlay |
    Prediction Overlay] (Sec 5.1). GT/Prediction "overlay" columns show the
    same raw crop as the Raw column, with the GT/prediction mask boundary
    contoured on top - not the bare binary mask fn-visualize/fp-visualize
    show, since this gallery is meant to let a reviewer judge the mask
    against the actual signal, not just see the mask shape.
    """
    if not loaded_cases:
        return None

    n = len(loaded_cases)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n), squeeze=False)
    col_titles = ["Raw", "Dark Sectioning", "GT Overlay", "Prediction Overlay"]

    for row_idx, (case, imgs) in enumerate(loaded_cases):
        panel_base = [imgs["raw"], imgs["dark"], imgs["raw"], imgs["raw"]]
        for col_idx, base_img in enumerate(panel_base):
            ax = axes[row_idx, col_idx]
            ax.imshow(base_img, cmap="gray", vmin=vmin, vmax=vmax)
            if col_idx == 2 and imgs["gt"].any():
                ax.contour(imgs["gt"] > 0, levels=[0.5], colors="lime", linewidths=1.2)
            elif col_idx == 3 and imgs["pr"].any():
                ax.contour(imgs["pr"] > 0, levels=[0.5], colors="red", linewidths=1.2)
            _add_scale_bar(ax, voxel_size_um)
            _add_source_label(ax, case)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx])
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    return fig


class RepresentativeCaseGalleryCheck(Check):
    name = "representative-case-gallery"
    description = (
        "Objectively-sampled example figures for 3 FN + 2 FP defect patterns (report Fig 5.3.2)"
    )
    # Same reasoning as fn-visualize/fp-visualize: re-reads raw/dark/gt/pred
    # TIFFs per sampled case on top of collect()'s own pass - opt-in only, so
    # `segdiag run all` stays fast by default.
    default_enabled = False

    def run(
        self, instances: pd.DataFrame, quality: pd.DataFrame, args: argparse.Namespace
    ) -> List[ReportArtifact]:
        if instances.empty:
            logger.warning("No collected instances - nothing to sample from.")
            return []

        root: Path = args.root
        mask_name: Optional[str] = getattr(args, "mask_name", None) or "Flatten_561_mask"
        raw_name: Optional[str] = getattr(args, "raw_name", None) or "Flatten_561"
        dark_name: str = getattr(args, "dark_name", None) or "Flatten_561_dark"
        seed: int = getattr(args, "gallery_seed", None) or DEFAULT_SEED
        n_per_pattern: int = getattr(args, "gallery_n_per_pattern", None) or DEFAULT_N_PER_PATTERN
        patterns_arg: Optional[str] = getattr(args, "gallery_patterns", None)
        voxel_size_um: float = getattr(args, "voxel_size_um", None) or DEFAULT_VOXEL_SIZE_UM
        cli_vmin: Optional[float] = getattr(args, "intensity_vmin", None)
        cli_vmax: Optional[float] = getattr(args, "intensity_vmax", None)

        if patterns_arg:
            patterns = [p.strip() for p in patterns_arg.split(",") if p.strip()]
            unknown = [p for p in patterns if p not in PATTERNS]
            if unknown:
                logger.warning("Ignoring unknown pattern(s) %s - valid: %s", unknown, PATTERNS)
                patterns = [p for p in patterns if p in PATTERNS]
        else:
            patterns = list(PATTERNS)

        candidate_counts: Dict[str, int] = {}
        cases_by_pattern: Dict[str, pd.DataFrame] = {}
        for pattern in patterns:
            candidates = _select_candidates(instances, pattern)
            candidate_counts[pattern] = len(candidates)
            logger.info("pattern=%s candidate_count=%d", pattern, len(candidates))
            cases_by_pattern[pattern] = _sample_cases(candidates, pattern, n_per_pattern, seed)

        folder_cache: Dict[Tuple[str, str], Optional[_ResolvedFolders]] = {}

        def _folders_for(sample: str, model: str) -> Optional[_ResolvedFolders]:
            key = (sample, model)
            if key not in folder_cache:
                folder_cache[key] = self._resolve_folders(
                    root, sample, model, mask_name, raw_name, dark_name
                )
            return folder_cache[key]

        # Phase 1: load every sampled case's raw pixel crop(s) up front, so
        # the shared intensity window (Sec 5.3) can be derived from what
        # actually got sampled this run, before any figure gets rendered.
        loaded_general: Dict[str, List[Tuple[pd.Series, Dict[str, np.ndarray]]]] = {}
        loaded_zcontext: Dict[str, List[Tuple[pd.Series, Dict[str, List[np.ndarray]]]]] = {}
        raw_samples_for_window: List[np.ndarray] = []

        for pattern in patterns:
            cases = cases_by_pattern.get(pattern)
            if cases is None or cases.empty:
                continue
            if pattern == "z_discontinuity":
                items: List[Tuple[pd.Series, Dict[str, List[np.ndarray]]]] = []
                for _, case in cases.iterrows():
                    folders = _folders_for(case["sample"], case["model"])
                    if folders is None:
                        continue
                    sample_data = _load_zcontext_images(case, folders)
                    if sample_data is None:
                        continue
                    items.append((case, sample_data))
                    raw_samples_for_window.extend(sample_data["raw"])
                loaded_zcontext[pattern] = items
            else:
                general_items: List[Tuple[pd.Series, Dict[str, np.ndarray]]] = []
                for _, case in cases.iterrows():
                    folders = _folders_for(case["sample"], case["model"])
                    if folders is None:
                        continue
                    imgs = _load_general_case_images(case, folders)
                    if imgs is None:
                        continue
                    general_items.append((case, imgs))
                    raw_samples_for_window.append(imgs["raw"])
                loaded_general[pattern] = general_items

        vmin, vmax = _resolve_intensity_window(raw_samples_for_window, cli_vmin, cli_vmax)
        logger.info(
            "representative-case-gallery: shared intensity window vmin=%.3f vmax=%.3f "
            "(voxel_size_um=%.3f, seed=%d)",
            vmin,
            vmax,
            voxel_size_um,
            seed,
        )

        artifacts: List[ReportArtifact] = []
        summary_rows: List[dict] = []

        for pattern in patterns:
            sampled_count = 0
            if pattern == "z_discontinuity":
                items = loaded_zcontext.get(pattern, [])
                for i, (case, sample_data) in enumerate(items):
                    fig = plot_zcontext_sample(
                        sample_data,
                        title=(
                            f"{pattern} #{i + 1}  (Model: {case['model']} | "
                            f"Sample: {case['sample']} | Center Slice: {case['slice_name']} | "
                            f"GT Z-span: {int(case['bbox_max_z'] - case['bbox_min_z'])} | "
                            f"Matched prediction Z-span: {int(case['matched_pred_z_span'])})"
                        ),
                        highlight_row="gt",
                        vmin=vmin,
                        vmax=vmax,
                    )
                    artifacts.append(
                        ReportArtifact(
                            name=f"case_gallery_{pattern}_{i + 1}",
                            table=case.to_frame().T,
                            figure=fig,
                            metadata={
                                "pattern": pattern,
                                "seed": seed,
                                "candidate_count": candidate_counts.get(pattern, 0),
                                "sampled_count": len(items),
                            },
                        )
                    )
                sampled_count = len(items)
            else:
                general_items = loaded_general.get(pattern, [])
                fig = _render_general_grid(general_items, vmin, vmax, voxel_size_um)
                if fig is not None:
                    artifacts.append(
                        ReportArtifact(
                            name=f"case_gallery_{pattern}",
                            table=pd.DataFrame([case for case, _ in general_items]),
                            figure=fig,
                            metadata={
                                "pattern": pattern,
                                "seed": seed,
                                "candidate_count": candidate_counts.get(pattern, 0),
                                "sampled_count": len(general_items),
                            },
                        )
                    )
                sampled_count = len(general_items)

            summary_rows.append(
                {
                    "pattern": pattern,
                    "seed": seed,
                    "candidate_count": candidate_counts.get(pattern, 0),
                    "sampled_count": sampled_count,
                }
            )

        # Total table: how large each pattern's candidate pool was, the seed
        # used, and how many cases actually got rendered - the numbers a
        # figure caption needs, so they don't have to be re-derived by hand.
        artifacts.append(
            ReportArtifact(name="case_gallery_sampling_summary", table=pd.DataFrame(summary_rows))
        )

        if not any(a.figure is not None for a in artifacts):
            logger.warning("No renderable cases found for any pattern.")

        return artifacts

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
        collected (sample, model) pair - same approach as
        ``FpVisualizationCheck._resolve_folders``.
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
        if not gt_files:
            return None

        return gt_dir, raw_dir, dark_dir, pred_dir, gt_files
