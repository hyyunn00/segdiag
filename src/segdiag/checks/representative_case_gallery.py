"""Representative case gallery for report Figure 5.3.2: 3 FN patterns + 2 FP
patterns, each selected by an objective, reproducible rule (not eyeballed by
a reviewer), then fixed-seed sampled. See
``SEGDIAG_REPRESENTATIVE_CASE_GALLERY.md`` for the filter-criteria spec this
implements; the rendering layout follows a later, more specific figure-legend
request that superseded the original doc's "one figure per pattern" design:

- **Panel A** (4 rows x 4 cols): the four patterns that read as a single
  flagged instance - ``contour_underestimate``/``no_response``/
  ``background_noise``/``missing_gt_annotation`` - one row each, columns
  [Raw | Dark Sectioning | GT Overlay | Prediction Overlay].
- **Panel B** (3 rows x 5 cols): ``z_discontinuity`` alone, which needs
  Z-context instead - rows [Raw | GT | Prediction], columns Z-2..Z+2.

Exactly **one** example per pattern (not a sampled gallery of several) - a
single well-chosen, objectively-selected case with a clear sampling
rationale is more persuasive in a 20-page report than several examples
crammed small enough to be illegible. ``--gallery-seed`` still matters: it's
what picks *which* one candidate out of the pool, reproducibly.

Unlike most checks, this one re-reads raw/dark/GT/prediction TIFFs to render
image panels - architecturally closer to ``fn_visualization.py``/
``fp_visualization.py`` than the purely tabular checks, though it does not
reuse their shared Z-context renderer: this figure's row/column layout,
overlay style (contour outlines, not the plain binary masks fn/fp-visualize
show), fixed crop window, and per-figure shared intensity window are all
different enough that forcing a shared renderer would cost more in
conditional branches than it saves in duplication.

Two departures from the original spec's data-layer plan, both because the
actual schema/table shape turned out to differ from what the spec assumed
(per its own instruction to verify current state before coding):

- ``z_min``/``z_max`` aren't separate fields - ``InstanceRecord`` already had
  ``bbox_min_z``/``bbox_max_z`` (half-open, like all its other bbox fields),
  reused here instead of adding a duplicate pair of fields.
- The spec's ``z_discontinuity`` filter joins a GT row against a *separate*
  "prediction" row via ``matched_instance_id`` to read the matched
  prediction's Z-span. That row doesn't exist: a prediction claimed by a GT
  match never gets its own "prediction"-role row in this table (only
  unmatched predictions/false positives do). Instead,
  ``core.pipeline._volume_instance_rows`` stores the matched prediction's
  Z-span directly on the GT row as ``matched_pred_z_span`` (see
  ``core.schema.InstanceRecord``), so no join is needed at all.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

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
    list_tif_files,
    resolve_image_dir,
)
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_VOXEL_SIZE_UM = 1.82
SCALE_BAR_LENGTH_UM = 10.0

#: Fixed crop window (voxels), centered on the flagged instance's centroid -
#: identical for all four Panel-A patterns and for Panel B, so panels are
#: directly visually comparable instead of each auto-sizing to its own bbox.
#: Matches the model's training patch size (also gives ~116 um field of view
#: at the default 1.82 um/voxel).
FIXED_CROP_SIZE = 64

PATTERNS = [
    "contour_underestimate",
    "no_response",
    "z_discontinuity",
    "background_noise",
    "missing_gt_annotation",
]

#: Panel A's row order - deliberately the same order as ``PATTERNS`` with
#: ``z_discontinuity`` removed (which gets its own Panel B).
_PANEL_A_PATTERNS = [p for p in PATTERNS if p != "z_discontinuity"]

#: English labels - deliberately not the Chinese text from the original
#: figure-legend request: matplotlib's default font (DejaVu Sans) has no
#: CJK glyph coverage, and a lab's analysis workstation/server can't be
#: relied on to have a CJK-capable font installed (confirmed missing-glyph
#: rendering on a Linux server, even though it renders fine on macOS, which
#: ships several). Bundling a font or requiring one to be installed would
#: make the check's output depend on the runtime environment; plain ASCII
#: labels sidestep the whole problem.
_PANEL_A_ROW_LABELS = {
    "contour_underestimate": "Contour Underestimate",
    "no_response": "No Response",
    "background_noise": "Background Noise",
    "missing_gt_annotation": "Missing GT Annotation",
}
_PANEL_A_COL_TITLES = ["Raw", "Dark Sectioning", "GT Overlay", "Prediction Overlay"]

_PANEL_B_ROW_TITLES = ["Raw", "GT", "Prediction"]
_PANEL_B_COL_TITLES = ["z-2", "z-1", "z", "z+1", "z+2"]
_PANEL_B_MODALITIES = ["raw", "gt", "pr"]

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

    Bbox fields are half-open (``bbox_max_*`` is exclusive, matching
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


_Loaded = TypeVar("_Loaded")


def _pick_and_load(
    candidates: pd.DataFrame,
    pattern: str,
    seed: int,
    folders_for: Callable[[str, str], Optional[_ResolvedFolders]],
    loader: Callable[[pd.Series, _ResolvedFolders], Optional[_Loaded]],
) -> Tuple[Optional[pd.Series], Optional[_Loaded]]:
    """Try every candidate for ``pattern``, in one fixed-seed shuffled order
    (:func:`_sample_cases` with ``n=len(candidates)`` - a full, seed-
    determined permutation of the pool, not a fresh random draw per
    attempt), and return the first ``(case, loaded)`` pair whose images
    actually load from disk.

    One candidate being unrenderable (stale parquet cache vs. current
    on-disk files, missing raw/dark/prediction TIFFs, or - for
    ``z_discontinuity`` - sitting too close to its sample's volume edge for
    a full Z-2..Z+2 window) no longer silently drops the whole pattern when
    a *different* candidate in the same pool would have worked; the caller
    used to have to notice the failure and manually retry with a different
    ``--gallery-seed``. The seed still fully determines the attempt order,
    so a given seed always produces the same outcome - this isn't a random
    retry loop, it's "try the whole pool, in a reproducible order."

    Returns ``(None, None)`` if the pool is empty or every candidate in it
    failed to load.
    """
    ordered = _sample_cases(candidates, pattern, n=len(candidates), seed=seed)
    attempts = 0
    for _, case in ordered.iterrows():
        attempts += 1
        folders = folders_for(case["sample"], case["model"])
        if folders is None:
            continue
        loaded = loader(case, folders)
        if loaded is not None:
            if attempts > 1:
                logger.info(
                    "pattern=%s: candidate #%d of %d (in seed=%d order) rendered - the "
                    "first %d failed (see preceding warnings) and were skipped.",
                    pattern,
                    attempts,
                    len(ordered),
                    seed,
                    attempts - 1,
                )
            return case, loaded
    if attempts:
        logger.warning(
            "pattern=%s: tried all %d candidate(s) in the pool (seed=%d), none could be "
            "rendered.",
            pattern,
            attempts,
            seed,
        )
    return None, None


def _resolve_intensity_window(
    raw_arrays: List[np.ndarray], vmin: Optional[float], vmax: Optional[float]
) -> Tuple[float, float]:
    """One shared display window for every panel in a figure: the
    0.1st/99.9th percentile of every raw pixel actually sampled for that
    figure, unless the caller pinned one or both ends explicitly. Panel A and
    Panel B each get their own window (computed separately, from their own
    sampled cases) since they're rendered as two independent figures.
    """
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    if not raw_arrays:
        return (float(vmin) if vmin is not None else 0.0), (
            float(vmax) if vmax is not None else 1.0
        )

    pooled = np.concatenate([a.astype(np.float64).ravel() for a in raw_arrays])
    lo = float(vmin) if vmin is not None else float(np.percentile(pooled, 0.1))
    hi = float(vmax) if vmax is not None else float(np.percentile(pooled, 99.9))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _add_scale_bar(ax, voxel_size_um: float, length_um: float = SCALE_BAR_LENGTH_UM) -> None:
    """Draw a ``length_um``-long scale bar near the bottom-right corner of
    ``ax``. Length in pixels is computed from ``voxel_size_um`` at call time
    - never hardcode a pixel length, it silently goes wrong the next time the
    microscope/objective (and therefore voxel size) changes. Called on
    exactly one panel per figure (the bottom-left cell) - repeating it on
    every panel would be redundant since every panel in a figure shares the
    same fixed crop size and voxel size.
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
        fontsize=8,
        ha="center",
        va="top",
    )


def _crop_fixed_window(
    arr: np.ndarray, center_row: float, center_col: float, size: int = FIXED_CROP_SIZE
) -> np.ndarray:
    """Crop a fixed ``size`` x ``size`` window centered on
    ``(center_row, center_col)`` - slides the window to stay in-bounds near
    an edge rather than shrinking it, so every panel is exactly ``size`` x
    ``size`` (unless the source image itself is smaller than ``size``).
    """
    h, w = arr.shape
    half = size // 2
    r_start = max(0, min(h - size, int(round(center_row)) - half)) if h >= size else 0
    c_start = max(0, min(w - size, int(round(center_col)) - half)) if w >= size else 0
    r_end = min(h, r_start + size)
    c_end = min(w, c_start + size)
    return arr[r_start:r_end, c_start:c_end]


def _load_panel_a_case_images(
    case: pd.Series, folders: _ResolvedFolders
) -> Optional[Dict[str, np.ndarray]]:
    """Read the flagged slice's raw/dark/gt/pred arrays and crop each to the
    fixed window centered on this case's centroid, for one Panel-A row.
    Returns ``None`` if the case's recorded slice can no longer be resolved
    on disk (e.g. stale cache).
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

    center_row, center_col = float(case["centroid_y"]), float(case["centroid_x"])
    try:
        return {
            "raw": _crop_fixed_window(tifffile.imread(str(raw_f)), center_row, center_col),
            "dark": _crop_fixed_window(tifffile.imread(str(dark_f)), center_row, center_col),
            "gt": _crop_fixed_window(tifffile.imread(str(gtf)), center_row, center_col),
            "pr": _crop_fixed_window(tifffile.imread(str(pred_f)), center_row, center_col),
        }
    except Exception as exc:  # noqa: BLE001 - keep scanning on bad files
        logger.warning("Error reading images around %s: %s", case["slice_name"], exc)
        return None


def _load_panel_b_case_images(
    case: pd.Series, folders: _ResolvedFolders
) -> Optional[Dict[str, List[np.ndarray]]]:
    """Read+crop the Z-2..Z+2 raw/gt/pred arrays (no dark - Panel B doesn't
    show it) for the z_discontinuity case, all cropped to the same fixed
    window centered on this GT cell's centroid.
    """
    gt_dir, raw_dir, dark_dir, pred_dir, gt_files = folders
    z_idx = int(case["z_index"])
    if not (0 <= z_idx < len(gt_files)) or gt_files[z_idx].name != case["slice_name"]:
        logger.warning(
            "Panel B: the z_discontinuity candidate's recorded slice (%s, z_index=%d) no "
            "longer matches this sample's on-disk GT files (%d slices found) - stale cache? "
            "Try --refresh-cache.",
            case["slice_name"],
            z_idx,
            len(gt_files),
        )
        return None
    if z_idx < 2 or z_idx > len(gt_files) - 3:
        logger.warning(
            "Panel B: this z_discontinuity candidate's center slice (z_index=%d of %d) is "
            "too close to its sample's volume edge for a full Z-2..Z+2 context window - "
            "trying the next candidate in the pool (if any; see _pick_and_load).",
            z_idx,
            len(gt_files),
        )
        return None

    center_row, center_col = float(case["centroid_y"]), float(case["centroid_x"])
    sample_data: Dict[str, List[np.ndarray]] = {"raw": [], "gt": [], "pr": []}
    try:
        for dz in (-2, -1, 0, 1, 2):
            gtf = gt_files[z_idx + dz]
            raw_f = get_corresponding_file(raw_dir, gtf)
            pred_f = get_corresponding_file(pred_dir, gtf)
            if not (raw_f and pred_f):
                logger.warning(
                    "Panel B: missing raw/prediction file for slice %s (z offset %+d from "
                    "center) - can't build the Z-context window.",
                    gtf.name,
                    dz,
                )
                return None
            sample_data["raw"].append(
                _crop_fixed_window(tifffile.imread(str(raw_f)), center_row, center_col)
            )
            sample_data["gt"].append(
                _crop_fixed_window(tifffile.imread(str(gtf)), center_row, center_col)
            )
            sample_data["pr"].append(
                _crop_fixed_window(tifffile.imread(str(pred_f)), center_row, center_col)
            )
    except Exception as exc:  # noqa: BLE001 - keep scanning on bad files
        logger.warning("Error reading Z-context around %s: %s", case["slice_name"], exc)
        return None
    return sample_data


def _render_panel_a(
    loaded_by_pattern: Dict[str, Dict[str, np.ndarray]],
    vmin: float,
    vmax: float,
    dark_vmin: float,
    dark_vmax: float,
    voxel_size_um: float,
) -> Optional[plt.Figure]:
    """4 rows (one per Panel-A pattern, in ``_PANEL_A_PATTERNS`` order) x 4
    cols [Raw | Dark Sectioning | GT Overlay | Prediction Overlay]. GT/
    Prediction overlay columns show the *same* raw crop as the Raw column,
    with the mask boundary contoured on top (outline only, not filled - a
    filled mask would hide the very signal a reviewer needs to judge the
    contour against, which matters most for ``contour_underestimate``).

    ``vmin``/``vmax`` apply to the Raw/GT-overlay/Prediction-overlay columns
    (all three display the raw crop); Dark Sectioning gets its own
    independently-computed ``dark_vmin``/``dark_vmax`` instead of reusing the
    raw window - it's a different imaging channel with its own intensity
    range, and stretching it through a window computed from Raw pixel values
    blows it out (or crushes it) rather than displaying it correctly.
    """
    patterns = [p for p in _PANEL_A_PATTERNS if p in loaded_by_pattern]
    if not patterns:
        return None

    n = len(patterns)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n), squeeze=False)
    col_windows = [(vmin, vmax), (dark_vmin, dark_vmax), (vmin, vmax), (vmin, vmax)]

    for row_idx, pattern in enumerate(patterns):
        imgs = loaded_by_pattern[pattern]
        panel_base = [imgs["raw"], imgs["dark"], imgs["raw"], imgs["raw"]]
        for col_idx, base_img in enumerate(panel_base):
            ax = axes[row_idx, col_idx]
            col_vmin, col_vmax = col_windows[col_idx]
            ax.imshow(base_img, cmap="gray", vmin=col_vmin, vmax=col_vmax)
            if col_idx == 2 and imgs["gt"].any():
                ax.contour(imgs["gt"] > 0, levels=[0.5], colors="lime", linewidths=1)
            elif col_idx == 3 and imgs["pr"].any():
                ax.contour(imgs["pr"] > 0, levels=[0.5], colors="magenta", linewidths=1)
            if row_idx == 0:
                ax.set_title(_PANEL_A_COL_TITLES[col_idx], fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(_PANEL_A_ROW_LABELS[pattern], fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    _add_scale_bar(axes[n - 1, 0], voxel_size_um)
    plt.tight_layout()
    return fig


def _render_panel_b(
    sample_data: Dict[str, List[np.ndarray]], vmin: float, vmax: float, voxel_size_um: float
) -> plt.Figure:
    """3 rows [Raw | GT | Prediction] x 5 cols [z-2 .. z+2], for the single
    z_discontinuity example. GT/Prediction rows show the plain binary mask
    (not an overlay) - there's no separate "base" image to overlay onto in
    this layout the way Panel A's Raw column serves that role.
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12), squeeze=False)

    for row_idx, mod in enumerate(_PANEL_B_MODALITIES):
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            img = sample_data[mod][col_idx]
            if mod == "raw":
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img > 0, cmap="gray", vmin=0, vmax=1)
            if row_idx == 0:
                ax.set_title(_PANEL_B_COL_TITLES[col_idx], fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(_PANEL_B_ROW_TITLES[row_idx], fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    _add_scale_bar(axes[2, 0], voxel_size_um)
    plt.tight_layout()
    return fig


class RepresentativeCaseGalleryCheck(Check):
    name = "representative-case-gallery"
    description = (
        "Two report figures (Fig 5.3.2 A/B): one objectively-selected example per defect pattern"
    )
    # Same reasoning as fn-visualize/fp-visualize: re-reads raw/dark/gt/pred
    # TIFFs per selected case on top of collect()'s own pass - opt-in only,
    # so `segdiag run all` stays fast by default.
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
        voxel_size_um: float = getattr(args, "voxel_size_um", None) or DEFAULT_VOXEL_SIZE_UM
        cli_vmin: Optional[float] = getattr(args, "intensity_vmin", None)
        cli_vmax: Optional[float] = getattr(args, "intensity_vmax", None)

        folder_cache: Dict[Tuple[str, str], Optional[_ResolvedFolders]] = {}

        def _folders_for(sample: str, model: str) -> Optional[_ResolvedFolders]:
            key = (sample, model)
            if key not in folder_cache:
                resolved = self._resolve_folders(
                    root, sample, model, mask_name, raw_name, dark_name
                )
                if resolved is None:
                    logger.warning(
                        "Couldn't resolve GT/raw/dark/prediction folders on disk for "
                        "(sample=%s, model=%s) - check --mask-name/--raw-name/--dark-name "
                        "match this sample's actual folder names, and that a dark folder "
                        "exists (folder resolution requires one even for patterns/panels "
                        "that don't render it). Logged once per (sample, model) pair.",
                        sample,
                        model,
                    )
                folder_cache[key] = resolved
            return folder_cache[key]

        # Exactly one case rendered per pattern - the whole point of this
        # figure design is a single, defensible example per defect type, not
        # a sampled gallery of several. Candidate pools themselves are kept
        # in full (not pre-trimmed to one row) so `_pick_and_load` below can
        # fall back through the rest of a pool, in fixed-seed order, if its
        # first choice turns out to be unrenderable.
        candidate_counts: Dict[str, int] = {}
        candidates_by_pattern: Dict[str, pd.DataFrame] = {}
        for pattern in PATTERNS:
            candidates = _select_candidates(instances, pattern)
            candidate_counts[pattern] = len(candidates)
            logger.info("pattern=%s candidate_count=%d", pattern, len(candidates))
            candidates_by_pattern[pattern] = candidates

        artifacts: List[ReportArtifact] = []

        # --- Panel A: the four single-slice patterns -----------------------
        panel_a_loaded: Dict[str, Dict[str, np.ndarray]] = {}
        panel_a_cases: List[pd.Series] = []
        panel_a_raw: List[np.ndarray] = []
        panel_a_dark: List[np.ndarray] = []
        for pattern in _PANEL_A_PATTERNS:
            case, imgs = _pick_and_load(
                candidates_by_pattern[pattern],
                pattern,
                seed,
                _folders_for,
                _load_panel_a_case_images,
            )
            if case is None or imgs is None:
                continue
            panel_a_loaded[pattern] = imgs
            panel_a_cases.append(case)
            panel_a_raw.append(imgs["raw"])
            panel_a_dark.append(imgs["dark"])

        panel_a_sampled_patterns = set(panel_a_loaded)
        if panel_a_loaded:
            vmin_a, vmax_a = _resolve_intensity_window(panel_a_raw, cli_vmin, cli_vmax)
            # Dark Sectioning is a different imaging channel from Raw, with
            # its own intensity range - it gets its own auto-computed window
            # (not the CLI --intensity-vmin/vmax override, which is
            # documented as pinning the Raw-derived window) rather than
            # inheriting Raw's, which would blow it out/crush it.
            dark_vmin_a, dark_vmax_a = _resolve_intensity_window(panel_a_dark, None, None)
            logger.info(
                "representative-case-gallery Panel A: shared intensity window "
                "vmin=%.3f vmax=%.3f (dark vmin=%.3f vmax=%.3f, voxel_size_um=%.3f, seed=%d)",
                vmin_a,
                vmax_a,
                dark_vmin_a,
                dark_vmax_a,
                voxel_size_um,
                seed,
            )
            fig_a = _render_panel_a(
                panel_a_loaded, vmin_a, vmax_a, dark_vmin_a, dark_vmax_a, voxel_size_um
            )
            if fig_a is not None:
                artifacts.append(
                    ReportArtifact(
                        name="case_gallery_panel_a_general",
                        table=pd.DataFrame(panel_a_cases),
                        figure=fig_a,
                        metadata={
                            "patterns": sorted(panel_a_sampled_patterns),
                            "seed": seed,
                            "vmin": vmin_a,
                            "vmax": vmax_a,
                            "dark_vmin": dark_vmin_a,
                            "dark_vmax": dark_vmax_a,
                            "crop_size": FIXED_CROP_SIZE,
                        },
                    )
                )
        else:
            logger.warning("Panel A: no renderable case found for any of %s", _PANEL_A_PATTERNS)

        # --- Panel B: z_discontinuity's Z-context ---------------------------
        z_case, sample_data = _pick_and_load(
            candidates_by_pattern["z_discontinuity"],
            "z_discontinuity",
            seed,
            _folders_for,
            _load_panel_b_case_images,
        )
        panel_b_rendered = False
        if z_case is not None and sample_data is not None:
            vmin_b, vmax_b = _resolve_intensity_window(sample_data["raw"], cli_vmin, cli_vmax)
            logger.info(
                "representative-case-gallery Panel B: shared intensity window "
                "vmin=%.3f vmax=%.3f (voxel_size_um=%.3f, seed=%d)",
                vmin_b,
                vmax_b,
                voxel_size_um,
                seed,
            )
            fig_b = _render_panel_b(sample_data, vmin_b, vmax_b, voxel_size_um)
            artifacts.append(
                ReportArtifact(
                    name="case_gallery_panel_b_z_discontinuity",
                    table=z_case.to_frame().T,
                    figure=fig_b,
                    metadata={
                        "pattern": "z_discontinuity",
                        "seed": seed,
                        "vmin": vmin_b,
                        "vmax": vmax_b,
                        "crop_size": FIXED_CROP_SIZE,
                    },
                )
            )
            panel_b_rendered = True
        # `_pick_and_load` already logged the specific reason if this stayed
        # False (empty pool, or every candidate in it failed to load).

        # Bookkeeping table: candidate-pool size, seed, and whether a case
        # actually got rendered for every pattern - the numbers a figure
        # caption needs, so they don't have to be re-derived by hand.
        summary_rows = [
            {
                "pattern": pattern,
                "seed": seed,
                "candidate_count": candidate_counts.get(pattern, 0),
                "sampled_count": int(
                    pattern in panel_a_sampled_patterns
                    or (pattern == "z_discontinuity" and panel_b_rendered)
                ),
            }
            for pattern in PATTERNS
        ]
        artifacts.append(
            ReportArtifact(name="case_gallery_sampling_summary", table=pd.DataFrame(summary_rows))
        )

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
