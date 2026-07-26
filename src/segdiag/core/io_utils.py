"""Dataset discovery and file-pairing helpers.

Folder discovery here is deliberately aligned with the lab's production
``analysis.py`` evaluation script rather than a fixed naming convention,
so that segdiag's diagnostics and ``analysis.py``'s official evaluation
report always agree on *which* GT/image/prediction folders belong together
for a given sample:

- GT folders: any directory matching ``*_mask`` (or ``--mask-name`` exactly)
  that isn't itself a prediction folder.
- Image (raw) folder: same convention as ``analysis.py`` - the GT folder's
  name with the ``_mask`` suffix removed (``Flatten_561_mask`` ->
  ``Flatten_561``), falling back to any other non-mask folder in the same
  parent if that doesn't exist. An explicit ``--raw-name``/``--dark-name``
  is tried first when the caller supplies one, for backward compatibility
  with datasets that don't follow the auto-detected convention.
- Prediction folders: any directory ending in ``.scroll-tif`` *or*
  ``.scroll-tiff`` (``analysis.py`` accepts both).
- Model/version name: extracted from a prediction folder name via the same
  ``{image_folder}_{model}_mask`` regex ``analysis.py`` uses, with the same
  fallback chain.

Two optional filters are supported everywhere a folder is discovered:

- ``sample_filter``: restrict to sample folders (the folder that directly
  contains ``Flatten_561_mask`` etc.) whose name matches.
- ``model_filter``: restrict to prediction folders whose name matches.

Both filters accept a comma-separated list of case-insensitive substrings;
a folder is kept if *any* term matches. Passing ``None`` (the default)
keeps everything, preserving the original "evaluate everything" behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

TIF_SUFFIXES = (".tif", ".tiff")

#: Both extensions analysis.py's discovery accepts for prediction folders.
PRED_FOLDER_SUFFIXES = (".scroll-tif", ".scroll-tiff")


def _matches_filter(name: str, filter_str: Optional[str]) -> bool:
    """Return True if ``name`` matches ``filter_str``.

    ``filter_str`` may contain several comma-separated terms; a match on
    any one term (case-insensitive substring) is enough. ``None`` or an
    empty string always matches (i.e. "no filter applied").
    """
    if not filter_str:
        return True
    terms = [t.strip().lower() for t in filter_str.split(",") if t.strip()]
    if not terms:
        return True
    name_lower = name.lower()
    return any(term in name_lower for term in terms)


def _is_pred_folder(p: Path) -> bool:
    name = p.name.strip()
    return any(name.endswith(suffix) for suffix in PRED_FOLDER_SUFFIXES)


def find_gt_folders(
    root: Path,
    mask_name: Optional[str] = None,
    sample_filter: Optional[str] = None,
) -> List[Path]:
    """Recursively find ground-truth mask folders under ``root``.

    If ``mask_name`` is given, only folders with that exact name are
    returned. Otherwise, any folder ending in ``_mask`` is considered a
    candidate - this matches ``root.rglob("*_mask")`` in ``analysis.py``.

    ``sample_filter`` optionally restricts results to samples (the GT
    folder's parent directory) whose name contains one of the given
    comma-separated substrings, e.g. ``"case01"`` or ``"case01,case02"``.
    """
    pattern = mask_name if mask_name else "*_mask"
    candidates: Iterable[Path] = root.rglob(pattern)
    folders = [
        p
        for p in candidates
        if p.is_dir()
        and not _is_pred_folder(p)
        and _matches_filter(p.parent.name, sample_filter)
    ]
    return sorted(folders)


def resolve_image_dir(gt_dir: Path, raw_name: Optional[str] = None) -> Optional[Path]:
    """Find the raw-image folder that goes with ``gt_dir``.

    Tries, in order:

    1. ``gt_dir.parent / raw_name`` if ``raw_name`` was given and exists
       (lets an explicit ``--raw-name`` keep working as before).
    2. ``gt_dir``'s own name with the ``_mask`` suffix stripped, e.g.
       ``Flatten_561_mask`` -> ``Flatten_561`` (``analysis.py``'s primary
       convention).
    3. The first other non-mask folder found in the same parent, sorted by
       name for determinism (``analysis.py``'s fallback, which under its
       ``{image}_{model}_mask.scroll-tif`` prediction-folder naming also
       happens to exclude prediction folders here, since their names
       contain ``_mask`` too).

    Returns ``None`` if nothing matches.
    """
    parent = gt_dir.parent

    if raw_name:
        candidate = parent / raw_name
        if candidate.is_dir():
            return candidate

    img_folder_name = gt_dir.name.replace("_mask", "")
    candidate = parent / img_folder_name
    if candidate.is_dir():
        return candidate

    other_dirs = sorted(
        d for d in parent.iterdir() if d.is_dir() and d != gt_dir and "_mask" not in d.name
    )
    return other_dirs[0] if other_dirs else None


def find_pred_folders(parent: Path, model_filter: Optional[str] = None) -> List[Path]:
    """Find model-prediction folders (``*.scroll-tif`` / ``*.scroll-tiff``)
    directly under ``parent``.

    ``model_filter`` optionally restricts results to model/version folders
    whose name contains one of the given comma-separated substrings, e.g.
    ``"v9"`` or ``"unet_v9,swin_v9"``.
    """
    folders = [
        p
        for p in parent.iterdir()
        if p.is_dir() and _is_pred_folder(p) and _matches_filter(p.name, model_filter)
    ]
    return sorted(folders)


def extract_model_name(img_dir_name: str, pred_dir_name: str) -> str:
    """Extract a model/version name from a prediction folder name, using
    the same ``{image_folder}_{model}_mask`` regex and fallback chain as
    ``analysis.py``.

    >>> extract_model_name("Flatten_561", "Flatten_561_unet_v9_mask.scroll-tif")
    'unet_v9'
    """
    pred_name = pred_dir_name.strip()

    match = re.search(f"{re.escape(img_dir_name)}_(.*)_mask", pred_name)
    if match:
        return match.group(1)

    # analysis.py's own fallback: strip the image-folder prefix and the
    # "_mask.scroll-tif" suffix.
    fallback = pred_name.replace(img_dir_name, "").replace("_mask.scroll-tif", "").strip("_")

    # Extra safety net (segdiag addition) for folders that don't follow the
    # {image}_{model}_mask convention at all - e.g. a bare
    # "<model>.scroll-tiff" folder - so a leftover suffix never ends up
    # baked into the reported "model name".
    for suffix in PRED_FOLDER_SUFFIXES:
        if fallback.endswith(suffix):
            fallback = fallback[: -len(suffix)]
            break

    return fallback.strip("_") or pred_name


def model_name_from_pred_dir(pred_dir: Path, img_dir_name: Optional[str] = None) -> str:
    """Convenience wrapper around :func:`extract_model_name` that accepts a
    ``Path`` directly. When ``img_dir_name`` is omitted, only the
    suffix-independent fallback chain applies (no ``{image}_`` prefix to
    strip), which still degrades gracefully for the older
    ``<model>.scroll-tiff`` convention.
    """
    return extract_model_name(img_dir_name or "", pred_dir.name)


def get_corresponding_file(dir_path: Path, base_f: Path) -> Optional[Path]:
    """Return the file in ``dir_path`` matching ``base_f``'s stem, tolerating
    a ``.tif``/``.tiff`` extension mismatch. Returns ``None`` if neither
    variant exists.
    """
    f1 = dir_path / base_f.name
    if f1.exists():
        return f1
    other_suffix = ".tiff" if base_f.suffix == ".tif" else ".tif"
    f2 = dir_path / (base_f.stem + other_suffix)
    if f2.exists():
        return f2
    return None


def list_tif_files(directory: Path) -> List[Path]:
    """Return all ``.tif``/``.tiff`` files directly under ``directory``,
    sorted by filename (which is assumed to encode Z-slice order).
    """
    return sorted(f for f in directory.iterdir() if f.is_file() and f.suffix in TIF_SUFFIXES)
