"""Shared 4x5 "Z-context gallery" rendering for fn-visualize/fp-visualize.

Both checks render the same grid - rows {Raw, Dark, GT, Prediction}, columns
the flagged slice plus its two neighbours on each side (Z-2 .. Z+2) - around
one flagged instance. The only difference between the two checks is *which*
row gets the center-slice marker (GT for a missed cell, Prediction for a
spurious one) and the title text, so that part is factored out here instead
of being duplicated across both check modules.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

#: Row order/labels and column labels are fixed - every gallery this module
#: renders shows the same four modalities across the same five Z-offsets.
_MODALITIES = ["raw", "dark", "gt", "pr"]
_ROW_TITLES = ["Raw Image", "Dark Image", "Ground Truth", "Prediction"]
_COL_TITLES = ["Z - 2", "Z - 1", "Center (Z)", "Z + 1", "Z + 2"]


def crop_with_padding(arr: np.ndarray, bbox, pad: int = 40) -> np.ndarray:
    """Crop ``arr`` to ``bbox`` expanded by ``pad`` pixels of context on each side."""
    min_row, min_col, max_row, max_col = bbox
    h, w = arr.shape
    r_start, r_end = max(0, min_row - pad), min(h, max_row + pad)
    c_start, c_end = max(0, min_col - pad), min(w, max_col + pad)
    return arr[r_start:r_end, c_start:c_end]


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Rescale an intensity image to [0, 1] for matplotlib display."""
    arr_float = arr.astype(np.float32)
    min_v, max_v = arr_float.min(), arr_float.max()
    if max_v - min_v > 1e-6:
        return (arr_float - min_v) / (max_v - min_v)
    return arr_float


def plot_zcontext_sample(
    sample_data: Dict[str, List[np.ndarray]],
    *,
    title: str,
    highlight_row: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Render the 4x5 Z-context grid and return the figure.

    ``sample_data`` maps each of ``raw``/``dark``/``gt``/``pr`` to a list of
    5 cropped arrays (Z-2 .. Z+2, center at index 2). ``highlight_row`` picks
    which row gets a red "+" marker on the center column - ``"gt"`` for a
    missed ground-truth cell (fn-visualize), ``"pr"`` for a spurious
    prediction (fp-visualize).

    ``vmin``/``vmax`` (both required together) display the raw/dark rows
    against one shared intensity window instead of each panel's own
    independent ``normalize_for_display`` min-max stretch - used by
    ``representative_case_gallery`` (SEGDIAG_REPRESENTATIVE_CASE_GALLERY.md
    section 5.3), which needs every panel in a run to share one window so
    panels are visually comparable. ``fn-visualize``/``fp-visualize`` don't
    pass these, so their existing per-panel auto-stretch is unchanged.
    """
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    shared_window = vmin is not None and vmax is not None

    for r, mod in enumerate(_MODALITIES):
        for c in range(5):
            ax = axes[r, c]
            img = sample_data[mod][c]

            if mod in ("gt", "pr"):
                ax.imshow(img > 0, cmap="gray", vmin=0, vmax=1)
            elif shared_window:
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            else:
                ax.imshow(normalize_for_display(img), cmap="gray")

            if mod == highlight_row and c == 2:
                h, w = img.shape
                ax.plot([w // 2], [h // 2], "r+", markersize=25, markeredgewidth=3)

            if r == 0:
                ax.set_title(_COL_TITLES[c], fontsize=18, fontweight="bold")
            if c == 0:
                ax.set_ylabel(_ROW_TITLES[r], fontsize=18, fontweight="bold")

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.suptitle(title, fontsize=22, y=1.02)
    return fig
