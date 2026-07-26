"""Serializing ``ReportArtifact``s to disk.

This is the only place that decides *how* a check's output gets saved
(csv / parquet / html) and *where* (the ``--output-dir``/source-tag
filename convention previously duplicated across every step). A check never
opens a file itself - it returns ``ReportArtifact`` objects and a
:class:`ReportWriter` takes it from there.

The ``--sample``/``--model``/``--output-dir`` filename-tagging convention
itself still lives in :mod:`segdiag.core.reporting` (``source_tag`` /
``resolve_output_dir``) unchanged - this module is just its new caller.
"""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

from segdiag.core.report import ReportArtifact


class ReportWriter(ABC):
    """Serializes one :class:`ReportArtifact` at a time."""

    extension: str

    @abstractmethod
    def write(
        self, artifact: ReportArtifact, output_dir: Path, *, source_tag: Optional[str] = None
    ) -> Path: ...

    def _target_path(self, artifact: ReportArtifact, output_dir: Path) -> Path:
        filename = f"{artifact.name}{self.extension}"
        return output_dir / filename

    def _tagged_target_path(
        self, artifact: ReportArtifact, output_dir: Path, source_tag: Optional[str]
    ) -> Path:
        path = self._target_path(artifact, output_dir)
        if source_tag:
            path = path.with_name(f"{source_tag}__{path.name}")
        # artifact.name may itself contain "/" (e.g. fn_visualization groups
        # its per-sample galleries under "step2_fn_diagnosis_3d/...") to ask
        # for a subfolder under output_dir - output_dir.mkdir() alone doesn't
        # create that, so every writer needs the *resolved* parent created,
        # not just the top-level output_dir.
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class CsvWriter(ReportWriter):
    extension = ".csv"

    def write(self, artifact, output_dir, *, source_tag=None):
        path = self._tagged_target_path(artifact, output_dir, source_tag)
        artifact.table.to_csv(path, index=False)
        if artifact.figure is not None:
            artifact.figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
        return path


class ParquetWriter(ReportWriter):
    extension = ".parquet"

    def write(self, artifact, output_dir, *, source_tag=None):
        path = self._tagged_target_path(artifact, output_dir, source_tag)
        artifact.table.to_parquet(path, index=False)
        return path


def _figure_to_data_uri(figure) -> str:
    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class HtmlWriter(ReportWriter):
    """Writes one self-contained HTML page per artifact (table + inline
    figure). Use :meth:`write_index` once per run to additionally build a
    single consolidated overview page linking every artifact together.
    """

    extension = ".html"

    def write(self, artifact, output_dir, *, source_tag=None):
        path = self._tagged_target_path(artifact, output_dir, source_tag)
        path.write_text(self._render_page(artifact.name, [artifact]), encoding="utf-8")
        return path

    def write_index(
        self,
        artifacts: Iterable[ReportArtifact],
        output_dir: Path,
        *,
        source_tag: Optional[str] = None,
    ) -> Path:
        artifacts = list(artifacts)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "index.html"
        if source_tag:
            filename = f"{source_tag}__{filename}"
        path = output_dir / filename
        path.write_text(self._render_page("segdiag report", artifacts), encoding="utf-8")
        return path

    def _render_page(self, title: str, artifacts: List[ReportArtifact]) -> str:
        sections = []
        for artifact in artifacts:
            table_html = artifact.table.to_html(index=False, max_rows=200)
            figure_html = ""
            if artifact.figure is not None:
                figure_html = (
                    f'<img alt="{artifact.name}" src="{_figure_to_data_uri(artifact.figure)}">'
                )
            sections.append(f"<section><h2>{artifact.name}</h2>{figure_html}{table_html}</section>")
        body = "\n".join(sections)
        return (
            f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            "<style>body{font-family:sans-serif;margin:2rem;}"
            "table{border-collapse:collapse;margin-bottom:1rem;}"
            "th,td{border:1px solid #ccc;padding:4px 8px;font-size:0.85rem;}"
            "img{max-width:100%;margin-bottom:1rem;}</style></head>"
            f"<body><h1>{title}</h1>{body}</body></html>"
        )


WRITER_REGISTRY = {
    "csv": CsvWriter,
    "parquet": ParquetWriter,
    "html": HtmlWriter,
}
