"""Tests for the fn-visualize check, which (uniquely) still does its own
TIFF I/O instead of reading collect()'s tables - it needs a dedicated small
dataset with a ghost cell placed at a slice index that has two full
neighbours on each side (indices 0/1 and the last two can never be a
"center" slice, by construction of the Z-context window).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import tifffile

from segdiag.checks.fn_visualization import FnVisualizationCheck

SHAPE = (60, 90)


def _write_dataset(root) -> None:
    sample_dir = root / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    raw_dir = sample_dir / "Flatten_561"
    dark_dir = sample_dir / "Flatten_561_dark"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    for d in (gt_dir, raw_dir, dark_dir, pred_dir):
        d.mkdir(parents=True)

    num_slices = 7
    ghost_z = 3  # eligible center: range(2, num_slices - 2) == [2, 3, 4]

    for z in range(num_slices):
        gt = np.zeros(SHAPE, dtype=np.uint8)
        pr = np.zeros(SHAPE, dtype=np.uint8)
        raw = np.full(SHAPE, 30.0, dtype=np.float32)
        dark = np.full(SHAPE, 5.0, dtype=np.float32)

        if z == ghost_z:
            # A ghost cell: GT present, nothing overlapping in the prediction.
            gt[20:30, 20:30] = 1
        else:
            # Every other slice: a normal matched cell, so folders aren't
            # otherwise empty/degenerate.
            gt[5:10, 5:10] = 1
            pr[5:10, 5:10] = 1

        tifffile.imwrite(gt_dir / f"slice_{z:04d}.tif", gt)
        tifffile.imwrite(pred_dir / f"slice_{z:04d}.tif", pr)
        tifffile.imwrite(raw_dir / f"slice_{z:04d}.tif", raw)
        tifffile.imwrite(dark_dir / f"slice_{z:04d}.tif", dark)


def test_fn_visualization_finds_the_ghost_cell(tmp_path):
    _write_dataset(tmp_path)
    args = argparse.Namespace(
        root=tmp_path,
        sample=None,
        model=None,
        output_dir=tmp_path / "out",
        mask_name=None,
        raw_name=None,
        dark_name="Flatten_561_dark",
        num_samples=5,
    )

    artifacts = FnVisualizationCheck().run(pd.DataFrame(), pd.DataFrame(), args)

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.name.startswith("step2_fn_diagnosis_3d/")
    assert artifact.figure is not None
    assert artifact.table.iloc[0]["sample"] == "case01"
    assert artifact.table.iloc[0]["model"] == "unet_v9"
    assert artifact.table.iloc[0]["volume"] == 100  # the ghost cell is a 10x10 square


def test_fn_visualization_respects_fn_volume_filter(tmp_path):
    _write_dataset(tmp_path)  # its one ghost cell is a 10x10 = 100 voxel square
    base_kwargs = dict(
        root=tmp_path,
        sample=None,
        model=None,
        output_dir=tmp_path / "out",
        mask_name=None,
        raw_name=None,
        dark_name="Flatten_561_dark",
        num_samples=5,
    )

    # A range that excludes the only ghost cell in the dataset -> no samples.
    excluding_args = argparse.Namespace(**base_kwargs, fn_min_volume=200, fn_max_volume=None)
    assert FnVisualizationCheck().run(pd.DataFrame(), pd.DataFrame(), excluding_args) == []

    # A range that includes it -> the sample still comes through.
    including_args = argparse.Namespace(**base_kwargs, fn_min_volume=50, fn_max_volume=150)
    artifacts = FnVisualizationCheck().run(pd.DataFrame(), pd.DataFrame(), including_args)
    assert len(artifacts) == 1
    assert artifacts[0].table.iloc[0]["volume"] == 100


def test_fn_visualization_returns_empty_when_no_ghost_cells(tmp_path):
    sample_dir = tmp_path / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    raw_dir = sample_dir / "Flatten_561"
    dark_dir = sample_dir / "Flatten_561_dark"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    for d in (gt_dir, raw_dir, dark_dir, pred_dir):
        d.mkdir(parents=True)

    for z in range(7):
        gt = np.zeros(SHAPE, dtype=np.uint8)
        gt[5:10, 5:10] = 1
        pr = gt.copy()  # every GT cell matched -> no ghosts anywhere
        raw = np.full(SHAPE, 30.0, dtype=np.float32)
        dark = np.full(SHAPE, 5.0, dtype=np.float32)
        tifffile.imwrite(gt_dir / f"slice_{z:04d}.tif", gt)
        tifffile.imwrite(pred_dir / f"slice_{z:04d}.tif", pr)
        tifffile.imwrite(raw_dir / f"slice_{z:04d}.tif", raw)
        tifffile.imwrite(dark_dir / f"slice_{z:04d}.tif", dark)

    args = argparse.Namespace(
        root=tmp_path,
        sample=None,
        model=None,
        output_dir=tmp_path / "out",
        mask_name=None,
        raw_name=None,
        dark_name="Flatten_561_dark",
        num_samples=5,
    )

    assert FnVisualizationCheck().run(pd.DataFrame(), pd.DataFrame(), args) == []
