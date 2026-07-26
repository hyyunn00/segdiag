"""Shared CLI options and output-path helpers used by every diagnostic step.

This module gives all six steps a consistent way to:

1. Scope a run to a single sample folder, a single model/version, or
   everything (``--sample`` / ``--model``).
2. Redirect every figure into one consolidated output folder instead of
   scattering files across the dataset tree (``--output-dir``).
3. Stamp every output filename with *where it came from* (dataset root,
   sample filter, model filter) so that re-running with different filters
   never silently overwrites a previous result.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``--sample``, ``--model``, and ``--output-dir`` options
    shared by every step.
    """
    parser.add_argument(
        "--sample",
        default=None,
        help=(
            "只評估「樣本」資料夾名稱包含此字串的資料（例如 case01）。"
            "可用逗號分隔多個樣本，例如 case01,case02。"
            "預設不篩選，會評估 --base-dir 底下找到的全部樣本。"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "只評估「模型/版本」預測資料夾（*.scroll-tif / *.scroll-tiff）名稱包含此字串的資料"
            "（例如 v9，會比對到 Flatten_561_unet_v9_mask.scroll-tif 等）。"
            "可用逗號分隔多個版本。預設不篩選，會評估找到的全部模型。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "將這個步驟產生的圖表集中存到此資料夾（會自動建立），並在檔名前面"
            "加上資料集/樣本/模型來源標籤，避免不同次分析互相覆蓋。"
            "預設維持原本行為：直接存在 --base-dir 底下。"
        ),
    )


def resolve_output_dir(args: argparse.Namespace, root: Path) -> Path:
    """Return the directory a step should save its figures into.

    If ``--output-dir`` was given, that directory is created (if needed)
    and returned. Otherwise, falls back to ``root`` (the original
    per-dataset behaviour).
    """
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out
    return root


def _slugify(text: str) -> str:
    """Turn free-form filter text into a filesystem-safe tag fragment."""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "-" for c in text.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "all"


def source_tag(root: Path, sample_filter: Optional[str], model_filter: Optional[str]) -> str:
    """Build a short, filesystem-safe tag describing the data a result was
    computed from: ``<dataset>__<sample>__<model>``.
    """
    dataset = _slugify(root.name)
    sample = _slugify(sample_filter.replace(",", "+")) if sample_filter else "all-samples"
    model = _slugify(model_filter.replace(",", "+")) if model_filter else "all-models"
    return f"{dataset}__{sample}__{model}"


def annotated_path(
    output_dir: Path,
    root: Path,
    sample_filter: Optional[str],
    model_filter: Optional[str],
    filename: str,
) -> Path:
    """Return ``output_dir / "<source-tag>__<filename>"``.

    This is the standard way every step should build its final save path,
    so that consolidating results under ``--output-dir`` never collides
    across different ``--sample``/``--model`` runs.
    """
    tag = source_tag(root, sample_filter, model_filter)
    return output_dir / f"{tag}__{filename}"


def annotated_subdir(
    output_dir: Path,
    root: Path,
    sample_filter: Optional[str],
    model_filter: Optional[str],
    base_name: str,
) -> Path:
    """Like :func:`annotated_path`, but for steps (e.g. ``fn-visualize``)
    that write a whole gallery of files into a subfolder rather than a
    single figure. Returns a created subdirectory.
    """
    tag = source_tag(root, sample_filter, model_filter)
    sub = output_dir / f"{base_name}__{tag}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def log_scope(
    logger_: logging.Logger,
    gt_folders: list,
    sample_filter: Optional[str],
    model_filter: Optional[str],
) -> None:
    """Log a short, consistent summary of what a run will actually cover,
    so users scoping to a single folder/version can confirm it matched
    what they expected before the (potentially slow) analysis runs.
    """
    sample_names = sorted({p.parent.name for p in gt_folders})
    logger_.info(
        "評估範圍: %d 個樣本 %s | sample filter=%s | model filter=%s",
        len(sample_names),
        sample_names,
        sample_filter or "(全部)",
        model_filter or "(全部)",
    )
