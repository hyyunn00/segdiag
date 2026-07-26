"""Shared synthetic dataset fixture for tests/test_checks/.

Unlike the small 1-2 instance arrays used in test_pipeline.py, checks like
``detection-probability`` (decile bins) and ``story-closure`` (quartile
bins) need enough distinct volume/intensity values for
``pandas.qcut(..., q=10)`` / ``q=4`` to find non-degenerate bin edges - so
this fixture builds a single richer dataset (one sample, one model, four
slices, ~2 dozen GT cells with varied size/intensity) that every check test
in this package shares, and deliberately plants one of each interesting
case: a volume outlier, a border-truncated cell, a density-anomaly slice,
and one false positive of each fp_root_cause subtype.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import tifffile

from segdiag.core.pipeline import collect

SHAPE = (300, 300)
BACKGROUND_MEAN = 20.0


def _place_square(
    gt: np.ndarray, raw: np.ndarray, row: int, col: int, size: int, intensity: float
) -> None:
    gt[row : row + size, col : col + size] = 1
    raw[row : row + size, col : col + size] = intensity


def _grid_positions(start=10, stop=280, pitch=30):
    for r in range(start, stop, pitch):
        for c in range(start, stop, pitch):
            yield r, c


def _build_dataset(root: Path) -> None:
    sample_dir = root / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    raw_dir = sample_dir / "Flatten_561"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    for d in (gt_dir, raw_dir, pred_dir):
        d.mkdir(parents=True)

    rng = np.random.default_rng(0)
    positions = _grid_positions()

    for z_index in range(4):
        gt = np.zeros(SHAPE, dtype=np.uint8)
        pr = np.zeros(SHAPE, dtype=np.uint8)
        raw = rng.normal(BACKGROUND_MEAN, 1.0, SHAPE).astype(np.float32)

        # A spread of true positives (matched exactly) with varied volume
        # and intensity, so decile/quartile binning has enough variety.
        # Slice 2 gets many more cells than the others - the density
        # anomaly this dataset plants for gt_annotation_quality.
        num_tp = 20 if z_index == 2 else 6
        for i in range(num_tp):
            row, col = next(positions)
            size = 8 + (i % 6) * 3  # 8..23 -> volumes 64..529
            intensity = 40 + (i % 5) * 8  # 40..72
            _place_square(gt, raw, row, col, size, intensity)
            pr[row : row + size, col : col + size] = 1

        if z_index == 0:
            # Blind FN: a GT cell with nothing overlapping it at all.
            row, col = next(positions)
            _place_square(gt, raw, row, col, 14, 50.0)

            # Merged FN + its boundary_split_fp: prediction is shifted just
            # enough to fail one-to-one matching, but sits close to the GT.
            row, col = next(positions)
            _place_square(gt, raw, row, col, 14, 55.0)
            pr[row + 6 : row + 20, col + 6 : col + 20] = 1

            # Volume outlier: a GT cell far larger than the rest, matched
            # exactly (so it isn't also flagged as a miss).
            row, col = next(positions)
            _place_square(gt, raw, row, col, 30, 50.0)  # volume 900
            pr[row : row + 30, col : col + 30] = 1

            # Border-truncated GT cell: touches the top-left image edge.
            gt[0:10, 0:10] = 1
            raw[0:10, 0:10] = 50.0
            pr[0:10, 0:10] = 1

            # noise_fp: tiny, isolated, background-level intensity.
            row, col = next(positions)
            pr[row : row + 5, col : col + 5] = 1
            raw[row : row + 5, col : col + 5] = BACKGROUND_MEAN

            # hallucination_fp: large, isolated, clearly-not-background intensity.
            row, col = next(positions)
            pr[row : row + 20, col : col + 20] = 1
            raw[row : row + 20, col : col + 20] = 90.0

        tifffile.imwrite(gt_dir / f"slice_{z_index:04d}.tif", gt)
        tifffile.imwrite(pred_dir / f"slice_{z_index:04d}.tif", pr)
        tifffile.imwrite(raw_dir / f"slice_{z_index:04d}.tif", raw)


@pytest.fixture
def collected_dataset(tmp_path):
    """Build the shared synthetic dataset and return
    ``(instances_df, quality_df, root, args)`` - ``args`` is a ready-to-use
    ``argparse.Namespace`` every ``Check.run()`` expects.
    """
    _build_dataset(tmp_path)
    instances_df, quality_df = collect(tmp_path)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    args = argparse.Namespace(
        root=tmp_path,
        sample=None,
        model=None,
        output_dir=output_dir,
        mask_name=None,
        raw_name=None,
        dark_name="Flatten_561_dark",
        num_samples=5,
    )
    return instances_df, quality_df, tmp_path, args
