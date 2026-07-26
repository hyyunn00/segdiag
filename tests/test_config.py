"""Tests for segdiag.core.config."""

from __future__ import annotations

from segdiag.core.config import (
    DatasetConfig,
    OutputConfig,
    SegdiagConfig,
    ThresholdsConfig,
    load_config,
)


def test_load_config_returns_defaults_when_file_missing(tmp_path):
    config = load_config(tmp_path / "does-not-exist.toml")

    assert config == SegdiagConfig()
    assert config.dataset.mask_name is None
    assert config.thresholds.tp_iou == 0.5
    assert config.output.format == ["csv"]


def test_load_config_parses_all_sections(tmp_path):
    toml_path = tmp_path / "segdiag.toml"
    toml_path.write_text(
        """
        [dataset]
        raw_name = "Flatten_561"
        dark_name = "Flatten_561_dark"
        mask_name = "Flatten_561_mask"

        [thresholds]
        tp_iou = 0.6
        blind_fn_iou = 0.1
        fp_boundary_split_distance = 15.0

        [output]
        format = ["csv", "html"]
        output_dir = "results/"
        """,
        encoding="utf-8",
    )

    config = load_config(toml_path)

    assert config.dataset == DatasetConfig(
        raw_name="Flatten_561", dark_name="Flatten_561_dark", mask_name="Flatten_561_mask"
    )
    assert config.thresholds == ThresholdsConfig(
        tp_iou=0.6, blind_fn_iou=0.1, fp_boundary_split_distance=15.0
    )
    assert config.output == OutputConfig(format=["csv", "html"], output_dir="results/")


def test_load_config_partial_file_fills_in_defaults(tmp_path):
    toml_path = tmp_path / "segdiag.toml"
    toml_path.write_text('[dataset]\nmask_name = "custom_mask"\n', encoding="utf-8")

    config = load_config(toml_path)

    assert config.dataset.mask_name == "custom_mask"
    assert config.dataset.raw_name is None
    assert config.thresholds == ThresholdsConfig()
