"""End-to-end smoke tests for the typer-based CLI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
from typer.testing import CliRunner

from segdiag.cli import app

runner = CliRunner()


def _write_dataset(root) -> None:
    shape = (40, 60)
    sample_dir = root / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    raw_dir = sample_dir / "Flatten_561"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    for d in (gt_dir, raw_dir, pred_dir):
        d.mkdir(parents=True)

    gt = np.zeros(shape, dtype=np.uint8)
    gt[5:15, 5:15] = 1
    pr = gt.copy()
    raw = np.full(shape, 30.0, dtype=np.float32)

    tifffile.imwrite(gt_dir / "slice_0000.tif", gt)
    tifffile.imwrite(pred_dir / "slice_0000.tif", pr)
    tifffile.imwrite(raw_dir / "slice_0000.tif", raw)


def test_cli_help_lists_run_and_list_checks_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "list-checks" in result.stdout


def test_cli_list_checks_includes_every_registered_check():
    result = runner.invoke(app, ["list-checks"])
    assert result.exit_code == 0
    assert "iou-distribution" in result.stdout
    assert "fp-root-cause" in result.stdout


def test_cli_run_unknown_check_exits_nonzero(tmp_path):
    _write_dataset(tmp_path)
    result = runner.invoke(app, ["run", "does-not-exist", "--base-dir", str(tmp_path)])
    assert result.exit_code == 1


def test_cli_run_iou_distribution_writes_csv_output(tmp_path):
    _write_dataset(tmp_path)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            "iou-distribution",
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
            "--format",
            "csv",
        ],
    )

    assert result.exit_code == 0, result.output
    csv_files = list(out_dir.glob("*step1_iou_distribution.csv"))
    assert len(csv_files) == 1


def test_cli_run_all_produces_a_consolidated_html_report(tmp_path):
    _write_dataset(tmp_path)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            "all",
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
            "--format",
            "html",
        ],
    )

    assert result.exit_code == 0, result.output
    index_files = list(out_dir.glob("*__index.html"))
    assert len(index_files) == 1


def test_cli_run_all_skips_opt_in_fn_visualize_by_default(tmp_path):
    _write_dataset(tmp_path)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            "all",
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
            "--format",
            "csv",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Running fn-visualize" not in result.output
    assert list(out_dir.glob("*fn_sample_*")) == []


def test_cli_run_fn_visualize_by_name_still_works(tmp_path):
    """Opt-in checks stay fully runnable - they just don't ride along with
    `all` - by naming them explicitly."""
    _write_dataset(tmp_path)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            "fn-visualize",
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
            "--format",
            "csv",
        ],
    )

    assert result.exit_code == 0, result.output


def test_cli_list_checks_marks_opt_in_checks(tmp_path):
    result = runner.invoke(app, ["list-checks"])
    assert result.exit_code == 0
    assert "fn-visualize" in result.stdout
    assert "opt-in" in result.stdout


def test_cli_list_checks_includes_cell_count_agreement():
    result = runner.invoke(app, ["list-checks"])
    assert result.exit_code == 0
    assert "cell-count-agreement" in result.stdout


def _write_dataset_with_small_fp(root) -> None:
    """A 900-voxel TP cell plus a 25-voxel hallucination FP - below
    MARS_MIN_VOLUME(40), so it's dropped under the default volume filter but
    kept when --min-volume/--max-volume are set to 0 (disabled).
    """
    shape = (60, 60)
    sample_dir = root / "case01"
    gt_dir = sample_dir / "Flatten_561_mask"
    pred_dir = sample_dir / "Flatten_561_unet_v9_mask.scroll-tif"
    for d in (gt_dir, pred_dir):
        d.mkdir(parents=True)

    gt = np.zeros(shape, dtype=np.uint8)
    pr = np.zeros(shape, dtype=np.uint8)
    gt[5:35, 5:35] = 1  # area 900, matched -> true_positive
    pr[5:35, 5:35] = 1
    pr[40:45, 40:45] = 1  # area 25 -> hallucination FP, below MARS_MIN_VOLUME

    tifffile.imwrite(gt_dir / "slice_0000.tif", gt)
    tifffile.imwrite(pred_dir / "slice_0000.tif", pr)


def test_cli_run_default_min_max_volume_filters_small_fp(tmp_path):
    _write_dataset_with_small_fp(tmp_path)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        ["run", "iou-distribution", "--base-dir", str(tmp_path), "--output-dir", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    summary_csv = next(out_dir.glob("*step1_iou_distribution_summary.csv"))
    summary = pd.read_csv(summary_csv)
    assert summary.iloc[0]["fp"] == 0


def test_cli_run_min_volume_zero_disables_the_filter(tmp_path):
    _write_dataset_with_small_fp(tmp_path)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            "iou-distribution",
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
            "--min-volume",
            "0",
            "--max-volume",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    summary_csv = next(out_dir.glob("*step1_iou_distribution_summary.csv"))
    summary = pd.read_csv(summary_csv)
    assert summary.iloc[0]["fp"] == 1
