# Architecture

segdiag splits what used to be six independent "read images -> compute ->
plot -> save" scripts into three stages:

```
Collect  --->  Checks (analysis + report)  --->  Writers (output)
```

## 1. Collect (`segdiag.core.pipeline.collect`)

A single pass over `--base-dir`: for every GT/prediction slice pair it finds
(using the same `analysis.py`-aligned folder discovery as always,
`segdiag.core.io_utils`), it runs `segdiag.core.matching.match_instances` +
`find_false_positives` once, plus per-slice image-quality metrics when a raw
image is available, and returns two long-format tables:

- **`instances`** (`segdiag.core.schema.InstanceRecord`) - one row per GT or
  predicted instance: volume, mean intensity, centroid, bbox,
  classification (`true_positive` / `blind_fn` / `merged_fn` /
  `false_positive`), best IoU, and (for false positives) an `fp_subtype`.
- **`quality`** (`segdiag.core.schema.ImageQualityRecord`) - one row per raw
  slice: SNR estimate, Laplacian variance (blur), saturation %, GT instance
  count and border-touching %.

Every check reads from these two DataFrames instead of re-reading TIFFs, so
a `segdiag run all` invocation only scans the dataset once regardless of how
many checks run. Pass `--refresh-cache` to force a re-scan when the
underlying images changed; otherwise a parquet cache under `--output-dir`
is reused.

## 2. Checks (`segdiag.checks.*`)

A `Check` is: given `(instances, quality, args)`, return zero or more
`segdiag.core.report.ReportArtifact`s. Nine checks ship today - the original
six diagnostic steps (rewritten to read from `collect()`'s tables instead of
scanning folders themselves) plus three new ones added in 1.0:

- `raw-image-quality`, `gt-annotation-quality`, `fp-root-cause` - see
  [Checks](checks.md) for what each answers.

`fn-visualize` is the one exception: it still does its own TIFF I/O, because
it needs dynamically-cropped 3D Z-context around specific sampled ghost
cells rather than a statistical summary of the whole dataset.

Adding a new check needs exactly two changes: a new module implementing
`segdiag.checks.base.Check`, and one line registering an instance of it in
`segdiag/checks/__init__.py`'s `CHECKS` dict - no CLI or I/O wiring required.

## 3. Writers (`segdiag.core.writers`)

A `ReportArtifact` is one table (always) plus an optional one figure.
`--format csv,parquet,html` (comma-separated, repeatable) decides how every
artifact from every check that ran gets serialized - a check never opens a
file itself. `html` additionally produces one consolidated `index.html`
combining every artifact from the run.

Output filenames are still tagged with a `<dataset>__<sample-filter>__
<model-filter>__` prefix (`segdiag.core.reporting.source_tag`, unchanged
since before the 1.0 rearchitecture), so re-running with different
`--sample`/`--model` scopes into the same `--output-dir` never collides.
