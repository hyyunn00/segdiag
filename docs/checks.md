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

Produces the report's Figure 5.3.2: exactly **one** objectively-selected,
reproducibly fixed-seed-sampled example case per defect pattern (three FN,
two FP) - not eyeballed by a reviewer, and not a multi-case gallery (a
single well-chosen example with a clear sampling rationale reads better in
a 20-page report than several examples crammed small enough to be
illegible). Each pattern is a rule over columns `collect()` already
computes:

| Pattern | What it captures | Selection rule |
|---|---|---|
| `contour_underestimate` | Matched, but the mask undershoots the real cell's contour | GT row, `classification == "true_positive"`, `0.50 <= best_iou < 0.60` |
| `no_response` | Model didn't react to the cell at all | GT row, `classification == "blind_fn"` |
| `z_discontinuity` | Model only caught a thin cross-section of a tall cell | GT row, `classification == "true_positive"`, GT Z-span (`bbox_max_z - bbox_min_z`) >= 3, matched prediction's Z-span == 1 |
| `background_noise` | Spurious detection indistinguishable from local background | Prediction row, `fp_subtype == "noise_fp"` |
| `missing_gt_annotation` | Spurious detection that looks like a real, un-annotated cell | Prediction row, `fp_subtype == "hallucination_fp"`, `background_contrast_sigma >= 2.0` |

Renders as **two figures**, not one figure per pattern - `z_discontinuity`
needs an entirely different layout (Z-context) than the other four (a
single flagged slice), so cramming all five into one figure would be
unreadable:

- **Panel A** (`case_gallery_panel_a_general`, 4 rows x 4 cols): one row
  each for `contour_underestimate` / `no_response` / `background_noise` /
  `missing_gt_annotation`, columns `[Raw | Dark Sectioning | GT Overlay |
  Prediction Overlay]`. The GT/Prediction overlay columns show the *same*
  raw crop as the Raw column with the mask boundary contoured on top in
  outline only (green for GT, magenta for prediction, 1px) - never filled,
  since a filled mask would hide the very signal a reviewer needs to judge
  the contour against (the whole point of `contour_underestimate`).
- **Panel B** (`case_gallery_panel_b_z_discontinuity`, 3 rows x 5 cols):
  `z_discontinuity` alone, rows `[Raw | GT | Prediction]`, columns Z-2..Z+2
  around the flagged GT cell's center slice.

Panel/row/column labels are plain English, not the original figure-legend
request's Chinese text - matplotlib's default font has no CJK glyph
coverage, and a lab analysis server can't be relied on to have a
CJK-capable font installed (confirmed missing-glyph rendering on one),
so ASCII labels sidestep the whole problem instead of depending on the
runtime environment's fonts.

Every panel in each figure is cropped to the same **fixed 64x64 voxel
window** centered on the flagged instance's centroid (`FIXED_CROP_SIZE` -
matches the model's training patch size, ~116 um field of view at the
default voxel size) - a fixed size, not a bbox-derived one, so every panel
in a figure is directly visually comparable. Each figure also gets its own
single shared intensity window (0.1st/99.9th percentile of that figure's
own sampled raw pixels, unless `--intensity-vmin`/`--intensity-vmax` pin
it) - never each panel auto-stretching independently - and exactly one
scale bar, drawn only on the bottom-left panel.

`--gallery-seed` (default `42`) picks which single candidate gets sampled
out of each pattern's pool (`pandas.DataFrame.sample(random_state=seed)` -
deliberately not the global `random` module `fn-visualize` uses, so
repeated calls never interfere with each other's draws). `--voxel-size-um`
(default `1.82`) drives the scale bar's pixel length.

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

A `case_gallery_sampling_summary` table is always emitted alongside the
figures, listing every pattern's candidate-pool size, seed, and whether a
case actually got rendered (`sampled_count` 0 or 1) - the numbers a figure
caption needs, so a missing pattern (empty candidate pool, or its case's
TIFFs no longer resolve on disk) is visible in the table even though the
other panel still renders.
