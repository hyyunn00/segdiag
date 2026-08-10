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
| `cell-count-agreement` | If this model's predictions were run through MARS, would the reported cell count agree with GT? | Yes |
| `representative-case-gallery` | What do objectively-selected, reproducibly-sampled example cases of each defect pattern actually look like? | Opt-in |

`cell-count-agreement` answers a different question than `obj_f1`: it uses a
position-*lenient* one-to-one match (`min_iou=0.05`, the same bar as a
"blind" FN) instead of the strict `min_iou=0.5` shape-fit `obj_f1` requires,
because the thing being validated here is *counting*, not localization
accuracy. Comparing `gt_count` to `pr_count` alone isn't enough - a model
that hallucinates predictions in empty background could still get the total
right - so `located_count_f1` requires those predictions to actually have
touched a real cell. Reports both a macro-average (primary ranking metric)
and a pooled supplement, plus a scatter plot and a Bland-Altman agreement
plot. See `SEGDIAG_MARS_ALIGNMENT_COMPLETE.md` Part 3 for the full
reasoning.

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

`fn-visualize` also takes `--fn-min-volume`/`--fn-max-volume` (voxels,
inclusive, unset = no limit) to narrow which ghost cells it samples down to
a specific size range - e.g. `--fn-min-volume 40 --fn-max-volume 100` to
only look at small misses. This is a second, ad-hoc filter applied on top
of the ghost cells that already survive the global `--min-volume`/
`--max-volume` MARS filter (see above) - it doesn't affect any counts/
metrics, only which samples get rendered. Each rendered sample's voxel
count is included in its title and in the artifact table's `volume`
column.

### `representative-case-gallery`

Produces the report's Figure 5.3.2: objectively-selected, reproducibly
fixed-seed-sampled example cases of five defect patterns (three FN, two
FP) - not eyeballed by a reviewer. Each pattern is a rule over columns
`collect()` already computes:

| Pattern | What it captures | Selection rule |
|---|---|---|
| `contour_underestimate` | Matched, but the mask undershoots the real cell's contour | GT row, `classification == "true_positive"`, `0.50 <= best_iou < 0.60` |
| `no_response` | Model didn't react to the cell at all | GT row, `classification == "blind_fn"` |
| `z_discontinuity` | Model only caught a thin cross-section of a tall cell | GT row, `classification == "true_positive"`, GT Z-span (`bbox_max_z - bbox_min_z`) >= 3, matched prediction's Z-span == 1 |
| `background_noise` | Spurious detection indistinguishable from local background | Prediction row, `fp_subtype == "noise_fp"` |
| `missing_gt_annotation` | Spurious detection that looks like a real, un-annotated cell | Prediction row, `fp_subtype == "hallucination_fp"`, `background_contrast_sigma >= 2.0` |

`--gallery-seed`/`--gallery-n-per-pattern` control the fixed-seed sample
(`pandas.DataFrame.sample(random_state=seed)` - deliberately not the global
`random` module `fn-visualize` uses, so repeated runs/patterns never
interfere with each other's draws); `--gallery-patterns` restricts to a
comma-separated subset instead of all five. `--intensity-vmin`/
`--intensity-vmax` pin the shared display window used across every panel
in the run (default: the 1st/99th percentile of every raw pixel actually
sampled that run - never each panel auto-stretching independently).
`--voxel-size-um` (default `1.82`) drives the scale bar's pixel length.

Four patterns render as a grid (rows = sampled cases, columns = Raw / Dark
Sectioning / GT Overlay / Prediction Overlay); `z_discontinuity` reuses
`fn-visualize`/`fp-visualize`'s shared Z-context renderer
(`checks._visualization.plot_zcontext_sample`) instead, since judging a
Z-discontinuity needs the neighbouring slices, not just the flagged one.

Two fields feed this check that don't exist for any other purpose:
`InstanceRecord.matched_pred_z_span` (a matched GT row's claiming
prediction's own Z-span - stored directly on the GT row, since a claimed
prediction never gets its own "prediction"-role row in this table) and
`InstanceRecord.background_contrast_sigma` (the continuous
`|mean_intensity - background_mean| / background_std` ratio
`fp_root_cause` computes but only ever thresholds into `fp_subtype`, kept
here as a number so this check can re-threshold it at a stricter 2 sigma).
Both are populated by `core.pipeline._volume_instance_rows()` for every
run, regardless of whether this check is ever invoked.
