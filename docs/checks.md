# Checks

Run any of these with `segdiag run <name> --base-dir ...`, or all of them
with `segdiag run all --base-dir ...`. List them (with descriptions, and an
`[opt-in, not run by 'all']` marker) at any time with `segdiag list-checks`.

## The original six (model output vs. ground truth)

| Check | Question it answers | Runs in `run all`? |
|---|---|---|
| `iou-distribution` | Are missed cells "never detected" or "detected but poorly matched"? | Yes |
| `fn-visualize` | What do the missed cells actually look like, in 3D context? | Opt-in |
| `fn-characteristics` | Are missed/spurious cells systematically smaller, dimmer, or deeper (Z)? | Yes |
| `detection-probability` | What's the empirical P(detected) curve vs. volume/intensity? | Yes |
| `fn-contribution` | Which cell-size bucket drives the missed cells - and the false positives? | Yes |
| `story-closure` | Combined volume x intensity interaction heatmap + final summary | Yes |

## New in 1.0 (input data quality + FP root-cause)

| Check | Question it answers | Runs in `run all`? |
|---|---|---|
| `raw-image-quality` | Is low recall a modeling problem, or is the raw image itself noisy/blurry/saturated? | Yes |
| `gt-annotation-quality` | Does the GT itself have volume outliers, border-truncated cells, or density anomalies (annotator drift)? | Yes |
| `fp-root-cause` | Is a false positive noise, a same-cell over-segmentation split, or a genuine hallucination? | Yes |
| `fp-visualize` | What do the spurious detections actually look like, in 3D context - especially hallucinations? | Opt-in |

`fp-root-cause`'s three buckets - `noise_fp`, `boundary_split_fp`,
`hallucination_fp` - are computed once during `collect()`
(`segdiag.checks.fp_root_cause.classify_fp_subtype`) and stored on every
false-positive row's `fp_subtype` column, so any other check can also group
by it. `fp-visualize` reads that same column to decide which FPs to render,
prioritizing `hallucination_fp` (the most concerning bucket) whenever
`--num-samples` can't fit everything.

## Opt-in checks

`fn-visualize` and `fp-visualize` each render up to `--num-samples` (default
20) full 4x5 Z-context figures (rows: Raw/Dark/GT/Prediction, columns: the
flagged slice +/- 2 Z-neighbours), each requiring its own raw/dark/gt/pred
TIFF reads on top of `collect()`'s own pass. That's slow enough relative to
the other (purely tabular) checks that `segdiag run all` skips them by
default (`Check.default_enabled = False` in `segdiag/checks/base.py`) -
`run all` logs which opt-in checks it skipped. Run them explicitly to opt
in:

```bash
segdiag run fn-visualize --base-dir /path/to/dataset --output-dir /path/to/results
segdiag run fp-visualize --base-dir /path/to/dataset --output-dir /path/to/results
```

Both share their plotting/cropping code
(`segdiag.checks._visualization.plot_zcontext_sample`); the only difference
is which row gets the center-slice marker - `gt` for a missed cell,
`pr` for a spurious one - and the title.
