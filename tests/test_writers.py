"""Tests for segdiag.core.writers."""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segdiag.core.report import ReportArtifact
from segdiag.core.writers import WRITER_REGISTRY, CsvWriter, HtmlWriter, ParquetWriter


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _make_artifact(with_figure=True) -> ReportArtifact:
    table = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    figure = None
    if with_figure:
        figure = plt.figure()
        plt.plot([0, 1], [0, 1])
    return ReportArtifact(name="demo_artifact", table=table, figure=figure)


def test_writer_registry_has_csv_parquet_html():
    assert set(WRITER_REGISTRY) == {"csv", "parquet", "html"}


def test_csv_writer_writes_table_and_companion_png(tmp_path):
    artifact = _make_artifact(with_figure=True)
    path = CsvWriter().write(artifact, tmp_path)

    assert path == tmp_path / "demo_artifact.csv"
    assert path.exists()
    assert path.with_suffix(".png").exists()

    loaded = pd.read_csv(path)
    assert list(loaded["a"]) == [1, 2]


def test_csv_writer_skips_png_when_no_figure(tmp_path):
    artifact = _make_artifact(with_figure=False)
    path = CsvWriter().write(artifact, tmp_path)

    assert path.exists()
    assert not path.with_suffix(".png").exists()


def test_csv_writer_applies_source_tag_prefix(tmp_path):
    artifact = _make_artifact(with_figure=False)
    path = CsvWriter().write(artifact, tmp_path, source_tag="mydataset__case01__v9")

    assert path.name == "mydataset__case01__v9__demo_artifact.csv"


def test_parquet_writer_writes_table_without_figure(tmp_path):
    artifact = _make_artifact(with_figure=True)
    path = ParquetWriter().write(artifact, tmp_path)

    assert path == tmp_path / "demo_artifact.parquet"
    assert not path.with_suffix(".png").exists()

    loaded = pd.read_parquet(path)
    assert list(loaded["a"]) == [1, 2]


def test_html_writer_writes_self_contained_page(tmp_path):
    artifact = _make_artifact(with_figure=True)
    path = HtmlWriter().write(artifact, tmp_path)

    assert path == tmp_path / "demo_artifact.html"
    content = path.read_text(encoding="utf-8")
    assert "demo_artifact" in content
    assert "data:image/png;base64," in content
    assert "<table" in content


def test_html_writer_write_index_combines_multiple_artifacts(tmp_path):
    artifacts = [_make_artifact(with_figure=True), _make_artifact(with_figure=False)]
    for i, a in enumerate(artifacts):
        a.name = f"artifact_{i}"

    path = HtmlWriter().write_index(artifacts, tmp_path, source_tag="tag")

    assert path == tmp_path / "tag__index.html"
    content = path.read_text(encoding="utf-8")
    assert "artifact_0" in content
    assert "artifact_1" in content
