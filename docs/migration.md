# Migrating from 0.x

segdiag 1.0 is a breaking change: the CLI moved from argparse subcommands
(one per step) to a single `typer`-based `run` command reading a
`CHECKS` registry, and every step now reads from one shared `collect()`
pass instead of scanning folders/TIFFs itself.

## Command mapping

| 0.x | 1.0 |
|---|---|
| `segdiag iou-distribution --base-dir D` | `segdiag run iou-distribution --base-dir D` |
| `segdiag fn-visualize --base-dir D --num-samples 20` | `segdiag run fn-visualize --base-dir D --num-samples 20` |
| `segdiag fn-characteristics --base-dir D --max-slices 100` | `segdiag run fn-characteristics --base-dir D --max-slices 100` |
| `segdiag detection-probability --base-dir D` | `segdiag run detection-probability --base-dir D` |
| `segdiag fn-contribution --base-dir D` | `segdiag run fn-contribution --base-dir D` |
| `segdiag story-closure --base-dir D` | `segdiag run story-closure --base-dir D` |
| `segdiag run-all --base-dir D` | `segdiag run all --base-dir D` |
| *(none)* | `segdiag run raw-image-quality --base-dir D` |
| *(none)* | `segdiag run gt-annotation-quality --base-dir D` |
| *(none)* | `segdiag run fp-root-cause --base-dir D` |
| *(none)* | `segdiag list-checks` |

`--sample`, `--model`, and `--output-dir` all work exactly as before.

## Behavior changes to be aware of

- **`--max-slices` is now a single global budget for the whole run, not a
  per-step one.** Previously each of steps 3-6 independently capped itself
  at 100-150 slices; since `collect()` now scans the dataset once for every
  check to share, one `--max-slices` value caps the total number of
  (sample, slice) pairs read across the entire `collect()` pass. **The
  default changed from "always capped" to unbounded** (`--max-slices` omitted
  means the whole dataset is scanned) - pass an explicit value if you want
  the old fast/partial-scan behavior back.
- **`--mask-name`/`--raw-name` now default to auto-detection** (the same
  `*_mask` glob / `_mask`-suffix-stripping convention `iou-distribution`
  already used) **instead of the literal `Flatten_561_mask`/`Flatten_561`
  defaults** steps 3-6 used to hardcode. This is a superset, not a
  narrowing: any dataset that used to work with the old hardcoded defaults
  auto-detects to the same folders today.
- **Output filenames/columns are unchanged** for the six original checks -
  same `stepN_<name>.<ext>` convention, same snake_case columns. Every
  check's output is now wrapped in a `ReportArtifact` and can additionally
  be written as `--format parquet` or `--format html` (previously PNG-only).
- **`fn-characteristics` and `fn-contribution` also emit an extra
  `_confusion`/`_fp_summary` artifact** (the aligned tp/fp/fn table and the
  FP-by-volume-bin table, respectively) - previously only printed to the
  console, now also saved as a table alongside the figure.

## New capabilities

Three checks that didn't exist in 0.x: `raw-image-quality`,
`gt-annotation-quality`, `fp-root-cause` - see [Checks](checks.md).
Also new: `segdiag.toml` project configuration (see
[Configuration](configuration.md)) and a parquet cache so repeated runs
against the same dataset don't re-read every TIFF (`--refresh-cache` to
force a re-scan).
