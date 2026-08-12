# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`representative-case-gallery` check** (`SEGDIAG_REPRESENTATIVE_CASE_GALLERY.md`,
  rendering layout per a later figure-legend request that superseded the
  original doc's "one figure per pattern" design): produces report Figure
  5.3.2 as **two figures**, each with exactly **one** objectively-selected,
  fixed-seed-picked example - not a multi-case gallery, and not crammed
  into a single figure (`z_discontinuity` needs an entirely different
  layout than the other four). Opt-in (like `fn-visualize`/`fp-visualize`),
  since it re-reads raw/dark/GT/prediction TIFFs per selected case.
  - **Panel A** (`case_gallery_panel_a_general`, 4 rows x 4 cols): one row
    each for `contour_underestimate`/`no_response`/`background_noise`/
    `missing_gt_annotation`, columns [Raw | Dark Sectioning | GT Overlay |
    Prediction Overlay] - overlay columns show the raw crop with the mask
    boundary contoured in outline only (green GT / magenta prediction,
    1px), never filled, so the underlying signal stays visible. Labels are
    plain English, not the original figure-legend request's Chinese text -
    matplotlib's default font has no CJK glyph coverage and a lab analysis
    server can't be relied on to have a CJK-capable font installed
    (confirmed missing-glyph/tofu rendering on one), so English labels
    sidestep the whole problem instead of depending on the runtime
    environment's fonts.
  - **Panel B** (`case_gallery_panel_b_z_discontinuity`, 3 rows x 5 cols):
    `z_discontinuity` alone, rows [Raw | GT | Prediction], columns Z-2..Z+2.
    Deliberately does *not* reuse `fn-visualize`/`fp-visualize`'s shared
    `plot_zcontext_sample` renderer (row count, overlay style, crop
    strategy, and intensity-window handling all diverge enough that a
    bespoke renderer was cheaper than branching a shared one).
  - Every panel in both figures uses a **fixed 64x64 voxel crop** centered
    on the flagged instance's centroid (`FIXED_CROP_SIZE` - matches the
    model's training patch size), not a bbox-derived one, so panels within
    a figure are directly comparable. Each figure gets its own single
    shared intensity window (0.1st/99.9th percentile of that figure's own
    sampled raw pixels, overridable via `--intensity-vmin`/
    `--intensity-vmax`) and exactly one scale bar (bottom-left panel only).
    Panel A's Dark Sectioning column gets its own separately-computed
    window (same method, from that column's own sampled Dark pixels, not
    CLI-overridable) instead of reusing Raw's - Dark is a different imaging
    channel with its own intensity range, and the original implementation
    blew it out by stretching it through Raw's window.
  - `segdiag.core.schema.InstanceRecord` gained two fields, both populated
    by `core.pipeline._volume_instance_rows()` for every run regardless of
    whether this check ever runs: `matched_pred_z_span` (a matched GT row's
    claiming-prediction's own Z-span, stored directly on the GT row since a
    claimed prediction never gets its own "prediction"-role row in this
    table - a simpler design than the spec's proposed join against a
    non-existent row) and `background_contrast_sigma` (the continuous
    `|mean_intensity - background_mean| / background_std` ratio
    `fp_root_cause` already computes internally but previously only ever
    collapsed into the `fp_subtype` threshold decision).
  - `z_min`/`z_max` from the original spec were **not** added as new
    fields - `InstanceRecord` already had equivalent `bbox_min_z`/
    `bbox_max_z` (half-open, like its other bbox fields), reused directly.
  - New CLI flags (this check only): `--gallery-seed` (default `42`, fixes
    the attempt order for each pattern's candidate pool),
    `--intensity-vmin`/`--intensity-vmax` (per-figure override),
    `--voxel-size-um` (default `1.82`, drives the scale bar).
  - `_pick_and_load()`: rendering tries every candidate in a pattern's pool,
    in fixed-seed order, and uses the first one that actually loads,
    instead of committing to a single `n=1` sample and silently dropping
    the whole pattern/panel if that one candidate turns out unrenderable
    (stale parquet cache vs. current on-disk files, missing raw/dark/
    prediction TIFFs, or - `z_discontinuity` only - sitting too close to
    its sample's volume edge for a full Z-2..Z+2 window). Logs the specific
    reason for each skipped candidate, and again if every candidate in a
    pool fails. The seed still fully determines the outcome (same seed ->
    same result every time); this is a reproducible fallback through the
    pool, not a random retry.

### Fixed
- **`fn-visualize` ignored `--model-exact`**: unlike every other check, this
  one does its own folder scan instead of reading `collect()`'s
  already-filtered instances table, and that scan only ever applied the
  substring `--model` filter - `--model-exact` was silently dropped, so a
  sibling model sharing a naming substring (e.g. `unet_v9_dark` when you
  asked for `unet_v9`) still got logged as "Searching FNs in..." and had
  ghost cells rendered from it. It now calls the same
  `io_utils.model_matches_exact()` gate `collect()` uses, before logging or
  processing each prediction folder.

### Added
- **`fn-visualize` size filter**: `--fn-min-volume`/`--fn-max-volume` (voxels,
  inclusive, unset = no limit) let a reviewer narrow the ghost-cell gallery
  down to a specific voxel-size range, independent of the global
  `--min-volume`/`--max-volume` MARS filter (which decides what counts as
  an instance at all, upstream of this). Each rendered sample's voxel
  count is now shown in its plot title and in the artifact table's new
  `volume` column.
  - `segdiag.core.matching.find_fn_bboxes()` gained `min_volume`/
    `max_volume` parameters and now returns `(bbox, volume)` pairs instead
    of bare bboxes.
- **MARS volume-filter alignment** (`SEGDIAG_MARS_ALIGNMENT_COMPLETE.md`
  Part 2): `--min-volume`/`--max-volume` CLI flags (and
  `[thresholds] min_volume`/`max_volume` in `segdiag.toml`), defaulting to
  `40`/`10000` - MARS `3Dfilter/filter_annotation.py`'s
  `AnnotationAnalyzer.get_points()` open-interval volume gate that decides
  whether a connected component counts as "one cell." **The `10000` upper
  bound is copied verbatim from that code's `if 25 < MarkerS < 10000:`
  check; the `40` lower bound is *not* - it's the TH (tyrosine
  hydroxylase)-marker operational threshold from lab experience, not
  `25` as literally written in MARS's source. Don't "fix" this back to `25`
  if you diff against MARS's code - see the module docstring in
  `core/matching.py` and the dedicated note below.** Unlike
  `--connectivity` (which defaults to unchanged behavior), these two
  **default to the MARS-aligned values**, so `gt_count`/`pr_count`/`tp`/
  `fp`/`fn` numbers change out of the box - pass `--min-volume 0
  --max-volume 0` to see unfiltered counts.
  - `segdiag.core.matching.filter_labels_by_volume()`: drops connected
    components whose voxel count isn't in `(min_volume, max_volume)` and
    relabels survivors to a contiguous `1..N`.
  - `segdiag.core.matching.label_and_filter()`: the new shared
    label-then-filter step behind `match_instances`/`find_false_positives`/
    `match_and_find_false_positives`, all three of which gained
    `min_volume`/`max_volume` parameters (defaulting to the MARS values).
  - `segdiag.core.pipeline.collect()`/`_volume_instance_rows()` thread
    `min_volume`/`max_volume` through to the same place `connectivity`
    already threads through - **re-run with `--refresh-cache` after
    changing either value; do not compare cached results across different
    volume-filter settings.**
- **Position-lenient count agreement** (`SEGDIAG_MARS_ALIGNMENT_COMPLETE.md`
  Part 3): a new `cell-count-agreement` check answering "if this model's
  predictions were run through MARS, would the reported cell count agree
  with GT?" - deliberately *not* just `min(gt_count, pr_count)`, since a
  model that hallucinates predictions in empty background could still get
  the raw totals to line up. Reuses the existing one-to-one greedy matcher
  at a position-lenient `min_iou=0.05` (`LOCATED_IOU_THRESHOLD`, the same
  bar as a "blind" FN) instead of `obj_f1`'s strict `0.5` shape-fit
  requirement.
  - `segdiag.core.matching`: `match_instances`/`find_false_positives`/
    `match_and_find_false_positives` gained a `min_iou` parameter
    (`None` default preserves the existing `TP_IOU_THRESHOLD` behavior
    unchanged).
  - `segdiag.core.schema.InstanceRecord.located_matched`: whether an
    instance is claimed under the lenient match, independent of
    `classification`/`matched_instance_id` (the strict match) - populated
    by `core.pipeline._volume_instance_rows()`, which now runs the shared
    one-to-one matcher twice on the same labeled/filtered arrays (once
    strict, once lenient) instead of relabeling the 3D volume twice.
  - `segdiag.checks.cell_count_agreement.CellCountAgreementCheck`
    (`cell-count-agreement`): `located_count_f1` per `(dataset, sample,
    model)`, macro-averaged across samples as the primary ranking metric
    (a pooled/summed-first version is reported alongside as a supplement,
    not the headline number - it lets cross-sample over/under-count errors
    cancel out and can flatter a model with one badly-off sample). Also
    emits a `gt_count` vs. `pr_count` scatter (+ `located_pr_count`, the
    "not pure noise" prediction count) and a Bland-Altman agreement plot.
- `--connectivity {6,18,26}` CLI flag (and `[thresholds] connectivity` in
  `segdiag.toml`) to align 3D connected-component labeling with MARS's
  `cc3d`-based `3Dfilter` (`connectivity=18`), instead of the previously
  hardcoded skimage full-connectivity default (equivalent to `26`). See
  `SEGDIAG_MARS_CONNECTIVITY.md` for the cc3d<->skimage mapping and the
  reasoning. Default remains `26`, so behavior is unchanged unless you pass
  the new flag/config value.
  - `segdiag.core.matching.cc3d_to_skimage_connectivity()`: the single
    conversion point from cc3d's neighbor-count convention (6/18/26) to
    `skimage.measure.label`'s `1..3`.
  - `segdiag.core.pipeline.collect()` / `_volume_instance_rows()` now accept
    a `connectivity` parameter and thread it through to
    `match_and_find_false_positives()`.
  - `segdiag.core.config.ThresholdsConfig.connectivity` (default `26`) is
    the first `thresholds` field actually wired through to
    `core.matching`/`core.pipeline`.
  - `checks/raw_image_quality.py`'s per-2D-slice GT instance count
    deliberately keeps its own hardcoded `connectivity=None` - MARS's
    18-connectivity is a 3D-only concept and doesn't apply to a single
    slice.

**This changes the absolute `gt_count`/`pr_count`/`tp`/`fp`/`fn` numbers**
whenever `--connectivity` differs from the default `26` - the same class of
impact as the fix in `SEGDIAG_3D_INSTANCE_FIX.md`. Re-run with
`--refresh-cache` after changing this value; do not compare cached results
produced under different `connectivity` settings.

## [1.0.0] - 2026-07-26

**Breaking change.** `segdiag` moved from six independent, argparse-driven
scripts to a modular collect/check/writer evaluation framework. See
[`docs/migration.md`](docs/migration.md) for the full old-CLI-to-new-CLI
command mapping and behavior-change list; the short version:

- `segdiag <step-name> --base-dir D` -> `segdiag run <check-name> --base-dir D`
- `segdiag run-all --base-dir D` -> `segdiag run all --base-dir D`
- Three checks that didn't exist before: `raw-image-quality`,
  `gt-annotation-quality`, `fp-root-cause`.
- `--max-slices` is now one global budget for the whole run instead of a
  per-step one, and defaults to unbounded instead of always-capped.

### Added (Phase 1 of the evaluation-framework upgrade: data-layer foundation)
- `segdiag.core.schema`: `InstanceRecord` (one row per GT/predicted instance)
  and `ImageQualityRecord` (one row per slice's image-quality metrics) -
  the standardized long-format tables every future check will read instead
  of re-reading TIFFs. `ImageQualityRecord` population lands with the
  `raw_image_quality` check in a later phase; for now `collect()` returns
  an empty, correctly-shaped quality table.
- `segdiag.core.pipeline.collect()`: a single-pass dataset scan that unifies
  the "find GT/prediction folders -> read TIFFs -> match_instances ->
  find_false_positives" logic previously duplicated across all six
  diagnostic steps. Supports optional parquet caching (`cache_path`,
  `force_refresh`) so repeated analysis runs don't re-scan the dataset.
  The six existing steps are unchanged in this phase and do not use
  `collect()` yet (that's Phase 2); this only adds the shared foundation
  and cross-validates it against the existing steps' own per-slice
  computation.
- `tests/test_schema.py`, `tests/test_pipeline.py`: including direct
  cross-validation of `pipeline._slice_instance_rows()` against the
  existing `fn_characteristics`/`detection_probability`/`fn_contribution`
  steps' own (unmodified) per-slice collection helpers on identical
  synthetic arrays, confirming identical volume/mean_intensity/best_iou/
  classification values.
- `pyarrow` added as a dependency (parquet read/write for the collect()
  cache).

### Added (Phase 2: output standardization + Check abstraction layer)
- `segdiag.core.report.ReportArtifact`: the standard "one table + optional
  one figure" container every check now returns, instead of each step
  deciding for itself what to `print()`/`savefig()`.
- `segdiag.core.writers`: `CsvWriter`/`ParquetWriter`/`HtmlWriter` (registered
  in `WRITER_REGISTRY`), selected by the new `--format csv,parquet,html`
  CLI flag (comma-separated, multiple allowed). `HtmlWriter.write_index()`
  additionally builds one consolidated `index.html` per run combining every
  artifact produced. The `--sample`/`--model`/`--output-dir` filename-
  tagging convention itself is unchanged (`segdiag.core.reporting.
  source_tag`/`resolve_output_dir`) - writers just call it now.
- `segdiag.checks.base.Check`: the abstract base every check implements
  (`name`, `description`, `run(instances, quality, args) ->
  list[ReportArtifact]`). All six original steps were rewritten as
  `Check` subclasses under `segdiag/checks/` (replacing `segdiag/steps/`,
  which is removed) that read from `collect()`'s tables instead of
  scanning folders/TIFFs themselves - `fn-visualize` is the one exception,
  since it needs dynamically-cropped 3D Z-context around sampled ghost
  cells rather than a statistical summary. `fn-characteristics` and
  `fn-contribution` also now emit an extra `_confusion`/`_fp_summary`
  artifact (previously console-only output).
- `segdiag.checks.CHECKS`: the registry `segdiag run <name>` / `run all`
  reads from. Adding a check needs one new file + one registry line - no
  CLI or I/O changes.
- `cli.py` rewritten with `typer` (`segdiag run <check|all> --base-dir ...`,
  `segdiag list-checks`), replacing the hand-rolled argparse subparsers.
- Cross-validated with `tests/test_checks/test_migrated_checks.py`: every
  migrated check's tp/fp/fn/precision/recall/f1 output is asserted against
  numbers computed directly from `match_instances()`/`find_false_positives()`
  on the same synthetic dataset.

### Added (Phase 3: input-data-quality and FP-root-cause checks)
- `segdiag.checks.raw_image_quality` (`raw-image-quality`): per-slice SNR
  estimate, Laplacian-variance blur proxy, saturation %, and a per-sample
  recall/FP-rate-vs-quality scatter - answers "is low recall a modeling
  problem, or is the raw data itself the problem?" Its pure
  `compute_image_quality_metrics()` is called directly by `collect()` to
  populate `ImageQualityRecord` (previously an empty placeholder table from
  Phase 1).
- `segdiag.checks.gt_annotation_quality` (`gt-annotation-quality`): IQR-based
  GT volume-outlier detection, border-touching-instance detection, and
  density-anomaly detection (GT count per slice vs. its own sample's
  median, flagging annotator-drift-style cliffs).
- `segdiag.checks.fp_root_cause` (`fp-root-cause`): classifies every false
  positive into `noise_fp` / `boundary_split_fp` / `hallucination_fp` via
  the pure, rule-based `classify_fp_subtype()`, called directly by
  `collect()` so `InstanceRecord.fp_subtype` is populated for every FP row
  regardless of whether this check ever runs.
- `tests/test_checks/`: one test file per check (migrated and new), plus a
  shared `conftest.py` fixture with a richer synthetic dataset (varied
  volume/intensity, planted outliers/border cells/density anomalies/FP
  subtypes) so decile/quartile-based checks have enough variety to bin on.

### Added (Phase 4: configuration + engineering conventions)
- `segdiag.core.config.load_config()` + `segdiag.toml.example`: optional
  `segdiag.toml` project config for `[dataset]`/`[output]` defaults (CLI
  flags always take precedence). `[thresholds]` is parsed/validated but not
  yet wired through to matching/collect internals - see
  `docs/configuration.md`.
- `mypy` (`[tool.mypy]` in `pyproject.toml`) and `ruff` (`[tool.ruff.lint]`,
  scoped to pyflakes/pycodestyle-core/isort - pyupgrade rules deliberately
  excluded since this codebase intentionally uses `typing.Optional`/`List`
  for Python 3.9 compatibility) now enforced in CI, alongside `black`.
- `pytest-cov` with an 80% coverage gate (`[tool.coverage.report]`).
- `.pre-commit-config.yaml` (ruff + black + mypy) and
  `.github/workflows/ci.yml` (pytest matrix across 3.9-3.12, lint, mypy).
- `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`.
- Logging switched from `logging.basicConfig` plain text to
  `rich.logging.RichHandler`.
- `mkdocs.yml` + `docs/` (architecture, checks, configuration, migration,
  contributing) - `pip install -e ".[docs]"` + `mkdocs serve`.
- `setuptools-scm` (git-tag-derived versioning) was considered but not
  adopted yet: this repository isn't currently under git version control,
  so there are no tags for it to derive a version from. The version is
  still bumped manually in `pyproject.toml`/`segdiag/__init__.py` for now.

### Added (Phase 5: documentation)
- `README.md` rewritten around the collect/check/writer architecture.
- This changelog entry / migration guide.

### Added (core alignment: matching algorithm, not just discovery)
- **`segdiag.core.matching`'s TP/FN membership is now computed by the same
  one-to-one greedy matching algorithm, iteration order, and IoU threshold
  as the lab's own `compute_object_metrics` (`metrics.py`)** - previously
  segdiag only used a per-GT "best IoU" rule with no one-to-one constraint,
  which could disagree with the lab's metrics in cases where two GT
  instances both overlap the same prediction. Verified against
  `compute_object_metrics` on 30 random synthetic cases with zero
  mismatches in tp/fp/fn counts.
- **`segdiag.core.matching.find_false_positives`**: segdiag previously never
  computed false positives at all (only GT-side "was this cell found?"
  diagnostics). This returns every predicted instance with no real GT
  partner - exactly what `compute_object_metrics` counts toward its FP
  total - with the same volume/intensity/centroid/bbox info already
  available for GT instances, so spurious detections can be characterized
  (are they small noise, or large plausible-looking hallucinations?) the
  same way missed cells already were.
- **Step 3 (`fn-characteristics`)** now plots false positives as a fourth
  category alongside blind FN / merged FN / true positive on all three
  panels (volume, intensity, Z-depth), and prints a per-model
  `tp/fp/fn/precision/recall/f1` summary computed with the exact same
  definitions as `compute_object_metrics`.
- **Step 5 (`fn-contribution`)** gained a fourth panel: false-positive
  count by volume bin, answering "are the model's hallucinated detections
  concentrated in the small/noisy end, or is it inventing large,
  plausible-looking cells?"
- **Step 1 (`iou-distribution`)** now splits its TP/FN histogram bars using
  the aligned `matched` flag (not a raw `best_iou >= 0.5` cutoff), and logs
  an aligned `tp/fp/fn/precision/recall/f1` summary alongside the plot.

### Added (CLI scoping, output consolidation, analysis.py-aligned discovery)
- `--sample TEXT` on every step and `run-all`: scope a run to one or more
  sample folders (comma-separated substrings), instead of always scanning
  the whole `--base-dir`.
- `--model TEXT` on every step and `run-all`: scope a run to one or more
  model/version prediction folders (comma-separated substrings, e.g. `v9`).
- `--output-dir PATH` on every step and `run-all`: consolidate all figures
  from a run into one folder instead of scattering them across the dataset
  tree. Every saved figure (and step 2's gallery subfolder) is
  automatically prefixed with a source tag
  (`<dataset>__<sample-filter>__<model-filter>`), so repeated runs with
  different scopes never overwrite each other.
- `segdiag.core.reporting` module: shared implementation of the above.
- Steps now log a one-line "evaluation scope" summary before doing any
  (potentially slow) work, so a mis-typed `--sample`/`--model` filter is
  caught immediately.
- `segdiag.core.io_utils.resolve_image_dir` / `extract_model_name`:
  analysis.py-aligned dynamic image-folder resolution and model-name
  extraction (`{image_folder}_{model}_mask` regex), with an explicit
  `--raw-name` still tried first for backward compatibility.
- Prediction-folder discovery now accepts `.scroll-tif` *and* `.scroll-tiff`.

### Changed (naming standardized to snake_case / consistent file prefixes)
- **Breaking**: `InstanceMatch.classify()` now returns short snake_case
  codes (`"blind_fn"`, `"merged_fn"`, `"true_positive"`) instead of
  descriptive strings like `"Blind FN (IoU < 0.05)"`. Look up
  `segdiag.core.matching.CLASSIFICATION_LABELS[code]` for the old
  human-readable text (used for chart labels).
- **Breaking**: every step's internal DataFrame columns are now lowercase
  snake_case (`volume`, `mean_intensity`, `z_depth`, `best_iou`,
  `classification`, `sample`, `model`, `volume_bin`, `intensity_bin`,
  `is_tp`, `is_fn`, `total_cells`, `fn_count`, `tp_count`,
  `pct_of_total_gt`, `recall_pct`, `pct_contribution_to_total_fn`, ...) -
  previously a mix of `"Title Case With Spaces"`, `"Mixed_Case"`, and
  `%`-suffixed names. This matches the lab's own `metrics.py` convention
  (`tp`, `fp`, `accuracy`, `obj_f1`, ...). Chart axis labels/titles are set
  explicitly and remain human-readable; only the underlying column names
  changed.
- **Breaking**: output filenames are now consistently `stepN_<name>.<ext>`
  for every step (previously only steps 4-6 had a step-number prefix):
  `step1_iou_distribution.png`, `step2_fn_diagnosis_3d/` (gallery folder),
  `step3_fn_characteristics.png`, `step4_detection_probability.png`,
  `step5_fn_contribution.png` (was `step5_fn_roi_analysis.png`),
  `step6_story_closure.png` (was `step6_final_story_closure.png`).
- `segdiag.core.matching.TP_IOU_THRESHOLD`/`BLIND_FN_IOU_THRESHOLD` are
  unchanged (still 0.5 / 0.05), but `InstanceMatch` gained a `matched: bool`
  field that now drives `is_tp`/`is_fn` (previously those properties were
  computed directly from `best_iou >= 0.5`).
- Default connectivity for connected-component labeling changed from `1`
  (face-connectivity) to `None` (skimage's full-connectivity default),
  matching `compute_object_metrics`'s default. Pass `connectivity=1`
  explicitly to reproduce the old behaviour.
- README's documented data layout now describes `analysis.py`'s actual
  prediction-folder naming convention
  (`{image}_{model}_mask.scroll-tif(f)`) as the primary/aligned convention.

### Fixed
- Steps 2-6 previously only ever evaluated the *first* prediction folder
  found per sample (`pred_folders[0]`), silently ignoring any additional
  models/versions present. They now evaluate every matching prediction
  folder. Use `--model` to isolate a single version instead.
- Collected per-cell records in steps 3-6 now carry `sample`/`model`
  columns, and each step prints a per-model breakdown when a run covers
  more than one model.

### Removed
- The `full-metrics` step (previously "Step 7") and the vendored copy of
  the lab's `utils/metrics.py` it depended on have been removed, along with
  the `torch`/`scikit-learn`/`scipy` dependencies and `xlsx` extra that
  only existed to support it. segdiag no longer reproduces `analysis.py`'s
  report directly; it stays a standalone, numpy/skimage-only toolkit whose
  matching algorithm and thresholds are built to *agree* with
  `analysis.py`/`metrics.py` (see the "Added (core alignment...)" entries
  above and `tests/test_matching.py`) without depending on or vendoring it.

## [0.1.0] - 2026-07-17

### Added
- Initial packaged release, consolidating six standalone diagnostic scripts
  into a single installable toolkit with a unified CLI (`segdiag`).
- Shared `segdiag.core.matching` module: a single, tested implementation of
  instance-level GT vs. prediction IoU matching (previously duplicated six
  times across the original scripts).
- Shared `segdiag.core.io_utils` module for dataset folder discovery and
  `.tif` / `.tiff` file pairing.
- `segdiag run-all` command to execute the full 6-step diagnostic pipeline
  in one call.
- Unit tests for the core matching logic.
