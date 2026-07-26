"""Tests for the fp-visualize check - the mirror of fn-visualize, but reading
its target instances from collect()'s table instead of re-matching, and only
falling back to direct TIFF I/O to render the Z-context crop itself.
"""

from __future__ import annotations

import argparse

import numpy as np
import tifffile

from segdiag.checks.fp_visualization import FpVisualizationCheck
from segdiag.core.pipeline import collect

SHAPE = (150, 200)
BACKGROUND_MEAN = 20.0


def _write_dataset(root, *, with_hallucination_fp: bool) -> None:
    sample_dir = root / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    raw_dir = sample_dir / "Flatten_561"
    dark_dir = sample_dir / "Flatten_561_dark"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    for d in (gt_dir, raw_dir, dark_dir, pred_dir):
        d.mkdir(parents=True)

    num_slices = 7
    fp_z = 3  # eligible center: range(2, num_slices - 2) == [2, 3, 4]
    rng = np.random.default_rng(0)

    for z in range(num_slices):
        gt = np.zeros(SHAPE, dtype=np.uint8)
        pr = np.zeros(SHAPE, dtype=np.uint8)
        raw = rng.normal(BACKGROUND_MEAN, 1.0, SHAPE).astype(np.float32)
        dark = np.full(SHAPE, 5.0, dtype=np.float32)

        # A normal matched cell on every slice, so folders/matching aren't
        # otherwise degenerate.
        gt[10:20, 10:20] = 1
        pr[10:20, 10:20] = 1
        raw[10:20, 10:20] = 45.0

        if z == fp_z and with_hallucination_fp:
            # hallucination_fp: sizable, isolated, clearly non-background
            # intensity, far (>20px) from the matched GT cell above.
            pr[100:130, 100:130] = 1
            raw[100:130, 100:130] = 95.0

        tifffile.imwrite(gt_dir / f"slice_{z:04d}.tif", gt)
        tifffile.imwrite(pred_dir / f"slice_{z:04d}.tif", pr)
        tifffile.imwrite(raw_dir / f"slice_{z:04d}.tif", raw)
        tifffile.imwrite(dark_dir / f"slice_{z:04d}.tif", dark)


def _args(tmp_path):
    return argparse.Namespace(
        root=tmp_path,
        sample=None,
        model=None,
        output_dir=tmp_path / "out",
        mask_name=None,
        raw_name=None,
        dark_name="Flatten_561_dark",
        num_samples=5,
    )


def test_fp_visualization_renders_the_hallucination(tmp_path):
    _write_dataset(tmp_path, with_hallucination_fp=True)
    instances_df, quality_df = collect(tmp_path)

    artifacts = FpVisualizationCheck().run(instances_df, quality_df, _args(tmp_path))

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.name.startswith("fp_diagnosis_3d/")
    assert artifact.figure is not None
    assert artifact.table.iloc[0]["sample"] == "case01"
    assert artifact.table.iloc[0]["model"] == "unet_v9"
    assert artifact.table.iloc[0]["fp_subtype"] == "hallucination_fp"


def test_fp_visualization_returns_empty_when_no_false_positives(tmp_path):
    _write_dataset(tmp_path, with_hallucination_fp=False)
    instances_df, quality_df = collect(tmp_path)

    assert FpVisualizationCheck().run(instances_df, quality_df, _args(tmp_path)) == []


def test_fp_visualization_returns_empty_when_dark_folder_missing(collected_dataset):
    # The shared tests/test_checks fixture has false positives of every
    # fp_root_cause subtype but no Flatten_561_dark folder - fp-visualize
    # should degrade to an empty result, not raise.
    instances_df, quality_df, _root, args = collected_dataset

    assert FpVisualizationCheck().run(instances_df, quality_df, args) == []
