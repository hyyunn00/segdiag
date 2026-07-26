# segdiag

**A model-evaluation framework for false-negative analysis of 3D instance segmentation models on microscopy data.**

`segdiag` answers one recurring research question: *"Why is my segmentation
model missing cells (low instance recall), and which cells are most
responsible for it?"* — and, since the 1.0 rearchitecture, two more that
previously had no tooling at all: *"Is the raw input data itself the
problem?"* and *"When the model hallucinates a detection, what kind of
mistake is it (noise, over-segmentation, or a genuine hallucination)?"*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

---

## Architecture: collect -> checks -> writers

```
Collect  --->  Checks (analysis + report)  --->  Writers (output)
```

- **Collect** (`segdiag.core.pipeline.collect`) scans `--base-dir` once,
  runs the shared GT<->prediction instance-matching algorithm
  (`segdiag.core.matching`), and returns two long-format tables
  (`instances`, `quality`) that every check reads from - no check re-reads
  TIFFs or re-runs matching itself.
- **Checks** (`segdiag.checks.*`) are pluggable: each one takes those two
  tables and returns standardized `ReportArtifact`s (a table + optional
  figure). Adding a new check is one new file + one registry line.
- **Writers** (`segdiag.core.writers`) serialize whatever `ReportArtifact`s a
  run produced as `csv`/`parquet`/`html`, decided by `--format` - not
  hardcoded per check.

See [`docs/architecture.md`](docs/architecture.md) for the full write-up.

## Checks

| Check | Question it answers | Runs in `run all`? |
|---|---|---|
| `iou-distribution` | Are missed cells "never detected" or "detected but poorly matched"? | Yes |
| `fn-visualize` | What do the missed cells actually look like, in 3D context? | **Opt-in** |
| `fn-characteristics` | Are missed/spurious cells systematically smaller, dimmer, or deeper (Z)? | Yes |
| `detection-probability` | What's the empirical P(detected) curve vs. volume/intensity? | Yes |
| `fn-contribution` | Which cell-size bucket drives the missed cells - and the false positives? | Yes |
| `story-closure` | Combined volume x intensity interaction heatmap + final summary | Yes |
| `raw-image-quality` *(new in 1.0)* | Is low recall a modeling problem, or is the raw image itself noisy/blurry/saturated? | Yes |
| `gt-annotation-quality` *(new in 1.0)* | Does the GT itself have volume outliers, border-truncated cells, or density anomalies? | Yes |
| `fp-root-cause` *(new in 1.0)* | Is a false positive noise, an over-segmentation split, or a genuine hallucination? | Yes |
| `fp-visualize` *(new in 1.0)* | What do the spurious detections actually look like, in 3D context - especially hallucinations? | **Opt-in** |

Run `segdiag list-checks` to see this list (with descriptions, and an
`[opt-in, not run by 'all']` marker) at any time. Full details in
[`docs/checks.md`](docs/checks.md).

**Opt-in checks.** `fn-visualize` and `fp-visualize` each render up to
`--num-samples` (default 20) full 4x5 Z-context figures, with their own
raw/dark/gt/pred TIFF reads on top of `collect()`'s own pass - slow enough
relative to the other (purely tabular) checks that `segdiag run all` skips
them by default. Run them explicitly to opt in:

```bash
segdiag run fn-visualize --base-dir /path/to/dataset --output-dir /path/to/results
segdiag run fp-visualize --base-dir /path/to/dataset --output-dir /path/to/results
```

`fp-visualize` prioritizes `hallucination_fp` samples (the most concerning
`fp-root-cause` subtype - looks like a real cell, but isn't) over
`boundary_split_fp`/`noise_fp` whenever `--num-samples` can't fit everything.

## Installation

```bash
git clone https://github.com/your-org/segdiag.git
cd segdiag
pip install -e .
```

Requires Python 3.9+.

## Expected data layout

`collect()` scans `--base-dir` recursively for GT/image/prediction folder
triplets, using the **same dynamic discovery rules as the lab's production
`analysis.py` evaluation script**:

```
<base_dir>/
  .../<sample_name>/
    Flatten_561/                                  # raw microscopy images (*.tif / *.tiff)
    Flatten_561_dark/                              # optional dark-sectioning images (fn-visualize/fp-visualize only)
    Flatten_561_mask/                              # ground-truth instance masks
    Flatten_561_<model_name>_mask.scroll-tif(f)/    # model prediction masks (one folder per model)
```

- **GT folders**: any directory matching `*_mask` (or `--mask-name` exactly).
- **Image folder**: an explicit `--raw-name`/`--dark-name` is tried first if
  given; otherwise it's auto-detected as the GT folder's name with the
  `_mask` suffix stripped (`Flatten_561_mask` -> `Flatten_561`), falling
  back to any other non-mask folder in the same sample directory.
- **Prediction folders**: any directory ending in `.scroll-tif` *or*
  `.scroll-tiff`.
- **Model/version name**: extracted from the prediction folder name via the
  `{image_folder}_{model}_mask` pattern (e.g.
  `Flatten_561_unet_v9_mask.scroll-tif` -> `unet_v9`), with a graceful
  fallback for prediction folders that don't follow that convention.

Slice files must share the same filename stem across folders (`.tif` vs.
`.tiff` extension mismatches are handled automatically).

## Quick start

```bash
# Run every check
segdiag run all --base-dir /path/to/dataset --output-dir /path/to/results

# Or just one
segdiag run iou-distribution --base-dir /path/to/dataset

# Scope to one sample/model, and pick output format(s)
segdiag run all --base-dir /path/to/dataset --sample case01 --model v9 --format csv,html

# --model is a substring match, so it can over-match a sibling model whose
# name contains your term (e.g. "v12_unet_baseline" also matches a
# "v12_unet_baseline_dark" folder). --model-exact matches the *extracted*
# model name exactly instead:
segdiag run all --base-dir /path/to/dataset --model-exact v12_unet_baseline
```

`segdiag run --help` lists every option; `segdiag list-checks` lists every
registered check. `--sample`/`--model`/`--output-dir` work exactly as
before: filter to one or more sample/model-version folders (comma-separated
substrings), and consolidate every run's output into one folder with a
`<dataset>__<sample-filter>__<model-filter>__` filename prefix so different
scopes never collide. `--model` and `--model-exact` can be combined - both
must match, so a conflicting pair (e.g. a `--model` substring that excludes
everything `--model-exact` was trying to select) silently yields zero
matches rather than an error, so keep them consistent.

> Coming from the pre-1.0 `segdiag <step-name> --base-dir ...` CLI? See
> [`docs/migration.md`](docs/migration.md) for the full command mapping and
> behavior changes.

### Configuration file

Pin your folder-naming convention and default output settings in
`segdiag.toml` instead of repeating CLI flags every time - copy
[`segdiag.toml.example`](segdiag.toml.example) to get started. An explicit CLI
flag always overrides the config file. See
[`docs/configuration.md`](docs/configuration.md).

### Caching

`collect()` caches its scan as parquet under `--output-dir`, so re-running
different checks (or `run all` again) against the same dataset doesn't
re-read every TIFF. Pass `--refresh-cache` to force a re-scan, and
`--max-slices N` to cap how many slices a single run reads in total.

## Alignment with `analysis.py`

segdiag's instance-matching algorithm and folder discovery are deliberately
kept in lockstep with the lab's production evaluation script,
`analysis.py`, so the two tools never quietly disagree about what a
"detection" is, which files belong to which sample/model, or how many
false positives a model produced:

- **One-to-one matching, not just a shared threshold.**
  `segdiag.core.matching.match_instances` determines TP/FN membership with
  the *same one-to-one greedy matching algorithm and iteration order* as
  `compute_object_metrics` in the lab's own `metrics.py`. Verified against
  `compute_object_metrics` on 30 random synthetic cases with zero
  mismatches in tp/fp/fn counts (see `tests/test_matching.py`).
- **False positives, not just false negatives.**
  `segdiag.core.matching.find_false_positives` surfaces every predicted
  instance with no real GT partner - exactly what `compute_object_metrics`
  counts toward its FP total - and `fp-root-cause` further splits those into
  noise / over-segmentation / hallucination.
- **Folder/model discovery** (`segdiag.core.io_utils`) mirrors
  `analysis.py`'s dynamic detection: the image folder is auto-resolved from
  the GT folder's name, prediction folders can end in `.scroll-tif` *or*
  `.scroll-tiff`, and model names are extracted with the same
  `{image}_{model}_mask` regex. An explicit `--raw-name`/`--mask-name`
  still takes priority when given.

segdiag doesn't vendor or call `analysis.py`/`metrics.py` directly - it's a
standalone toolkit whose matching rules and thresholds were built to agree
with them, checked by the test suite above.

## Library usage

The shared matching logic and the collect/check tables are also usable
directly as a Python library, independent of the CLI:

```python
import tifffile
from segdiag.core.matching import find_false_positives, match_instances

gt = tifffile.imread("case01/Flatten_561_mask/slice_000.tif")
pred = tifffile.imread("case01/Flatten_561_unet_v9_mask.scroll-tif/slice_000.tif")

for m in match_instances(gt, pred):
    print(m.gt_id, m.volume, m.best_iou, m.classify())  # "blind_fn" / "merged_fn" / "true_positive"

for fp in find_false_positives(gt, pred):
    print(fp.pr_id, fp.volume)  # predictions with no real GT partner
```

```python
from pathlib import Path
from segdiag.core.pipeline import collect

instances, quality = collect(Path("/path/to/dataset"))
instances[instances["role"] == "gt"].groupby("classification").size()
```

## Project layout

```
segdiag/
  pyproject.toml
  segdiag.toml.example
  mkdocs.yml
  docs/                         # mkdocs site (architecture/checks/config/migration)
  src/segdiag/
    cli.py                      # `segdiag` CLI (typer), reads the CHECKS registry
    core/
      matching.py                # shared GT<->prediction instance-matching logic
      io_utils.py                  # analysis.py-aligned dataset discovery / file-pairing
      reporting.py                  # --sample/--model/--output-dir + filename tagging
      schema.py                     # InstanceRecord / ImageQualityRecord
      pipeline.py                   # collect(): the single dataset-scanning pass
      report.py                     # ReportArtifact
      writers.py                    # Csv/Parquet/HtmlWriter
      config.py                     # segdiag.toml loading
    checks/                      # pluggable checks (registry in __init__.py)
      base.py                       # Check ABC + default_enabled (opt-in vs. `run all`)
      _visualization.py             # shared 4x5 Z-context gallery rendering
      iou_distribution.py           # Step 1
      fn_visualization.py           # Step 2 (opt-in; still does its own TIFF I/O - see docs)
      fn_characteristics.py         # Step 3
      detection_probability.py      # Step 4
      fn_contribution.py            # Step 5
      story_closure.py              # Step 6
      raw_image_quality.py          # new in 1.0
      gt_annotation_quality.py      # new in 1.0
      fp_root_cause.py              # new in 1.0
      fp_visualization.py           # new in 1.0 (opt-in; mirrors fn_visualization.py for FPs)
  tests/
    test_matching.py
    test_io_utils_and_reporting.py
    test_schema.py
    test_pipeline.py
    test_writers.py
    test_config.py
    test_cli.py
    test_checks/                 # one test file per check
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install                                    # optional: ruff/black/mypy on commit
pytest --cov=segdiag --cov-report=term-missing          # tests + coverage (CI gate: 80%)
ruff check src tests
black --check src tests
mypy src/segdiag
```

See [`docs/contributing.md`](docs/contributing.md) for how to add a new check.

## License

[MIT](LICENSE)
