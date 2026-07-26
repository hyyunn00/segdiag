"""Unified command-line entry point for the segdiag toolkit.

Install the package (``pip install -e .``) and run::

    segdiag --help
    segdiag run iou-distribution --base-dir /path/to/data
    segdiag run all --base-dir /path/to/data

Every check reads from a single ``collect()`` pass over the dataset (see
``segdiag.core.pipeline``) instead of re-scanning folders/TIFFs itself, and
returns standardized ``ReportArtifact``s that get serialized by whichever
``--format`` writer(s) were requested (see ``segdiag.core.writers``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.logging import RichHandler

from segdiag.checks import CHECKS
from segdiag.core.config import load_config
from segdiag.core.pipeline import collect
from segdiag.core.reporting import resolve_output_dir, source_tag
from segdiag.core.writers import WRITER_REGISTRY, HtmlWriter

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help=(
        "Diagnostic toolkit for false-negative analysis of 3D instance "
        "segmentation models on microscopy data."
    ),
)


def _check_names() -> str:
    return ", ".join(sorted(CHECKS)) + ", all"


@app.command()
def run(
    check: str = typer.Argument(..., help=f"Check to run - one of: {_check_names()}"),
    base_dir: Path = typer.Option(
        ..., "--base-dir", help="Root directory to search for GT/prediction folders"
    ),
    sample: Optional[str] = typer.Option(
        None, "--sample", help="只評估樣本資料夾名稱包含此字串的資料，可用逗號分隔多個樣本"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="只評估模型/版本預測資料夾名稱包含此字串的資料，可用逗號分隔多個版本"
    ),
    mask_name: Optional[str] = typer.Option(
        None, "--mask-name", help="GT mask 資料夾名稱（預設自動偵測 *_mask）"
    ),
    raw_name: Optional[str] = typer.Option(
        None, "--raw-name", help="Raw image 資料夾名稱（預設自動偵測，去掉 _mask 後綴）"
    ),
    dark_name: Optional[str] = typer.Option(
        None,
        "--dark-name",
        help="Dark image 資料夾名稱（僅 fn-visualize 需要，預設 Flatten_561_dark）",
    ),
    num_samples: int = typer.Option(20, "--num-samples", help="fn-visualize 產生的 FN 樣本數量"),
    max_slices: Optional[int] = typer.Option(
        None, "--max-slices", help="整個 collect 掃描階段最多處理的 slice 數量（預設不限制）"
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="輸出集中存放的資料夾"),
    format: Optional[str] = typer.Option(
        None, "--format", help="輸出格式，逗號分隔：csv,parquet,html（預設 csv）"
    ),
    refresh_cache: bool = typer.Option(
        False, "--refresh-cache", help="強制重新掃描資料集，忽略既有的 parquet 快取"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging"),
) -> None:
    """Run one diagnostic check (or ``all``) against a dataset."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_path=False)],
    )

    if check != "all" and check not in CHECKS:
        typer.echo(f"Unknown check '{check}'. Available: {_check_names()}", err=True)
        raise typer.Exit(code=1)

    # segdiag.toml only fills in a default for a flag that wasn't passed at
    # all - an explicit CLI flag always wins.
    config = load_config()
    effective_mask_name = mask_name or config.dataset.mask_name
    effective_raw_name = raw_name or config.dataset.raw_name
    effective_dark_name = dark_name or config.dataset.dark_name or "Flatten_561_dark"
    effective_output_dir = output_dir or (
        Path(config.output.output_dir) if config.output.output_dir else None
    )
    effective_format = format or ",".join(config.output.format) or "csv"

    root = base_dir.resolve()
    out_dir = resolve_output_dir(
        argparse.Namespace(output_dir=str(effective_output_dir) if effective_output_dir else None),
        root,
    )
    tag = source_tag(root, sample, model)

    logger.info(
        "Scope for this run: sample filter=%s | model filter=%s | output-dir=%s",
        sample or "(全部)",
        model or "(全部)",
        out_dir,
    )

    instances, quality = collect(
        root,
        mask_name=effective_mask_name,
        raw_name=effective_raw_name,
        sample_filter=sample,
        model_filter=model,
        max_slices=max_slices,
        cache_path=out_dir / f"{tag}__collect_cache",
        force_refresh=refresh_cache,
    )

    run_args = argparse.Namespace(
        root=root,
        sample=sample,
        model=model,
        output_dir=out_dir,
        mask_name=effective_mask_name,
        raw_name=effective_raw_name,
        dark_name=effective_dark_name,
        num_samples=num_samples,
    )

    checks_to_run = list(CHECKS.values()) if check == "all" else [CHECKS[check]]
    formats = [f.strip() for f in effective_format.split(",") if f.strip()]

    all_artifacts = []
    for c in checks_to_run:
        logger.info("=== Running %s ===", c.name)
        try:
            artifacts = c.run(instances, quality, run_args)
        except Exception as exc:  # noqa: BLE001 - one failing check shouldn't abort `run all`
            logger.error("Check '%s' failed: %s", c.name, exc)
            continue

        all_artifacts.extend(artifacts)
        for artifact in artifacts:
            for fmt in formats:
                writer_cls = WRITER_REGISTRY.get(fmt)
                if writer_cls is None:
                    logger.warning("Unknown format '%s' - skipping.", fmt)
                    continue
                path = writer_cls().write(artifact, out_dir, source_tag=tag)  # type: ignore[abstract]
                logger.info("Wrote %s", path)

    if "html" in formats and all_artifacts:
        index_path = HtmlWriter().write_index(all_artifacts, out_dir, source_tag=tag)
        logger.info("Wrote consolidated report: %s", index_path)


@app.command(name="list-checks")
def list_checks() -> None:
    """List every available check name and description."""
    for name, c in sorted(CHECKS.items()):
        typer.echo(f"{name}: {c.description}")


def main(argv: Optional[List[str]] = None) -> int:
    try:
        app(args=argv, standalone_mode=False)
    except typer.Exit as exc:
        return exc.exit_code
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
