# Checks

Run any of these with `segdiag run <name> --base-dir ...`, or all of them
with `segdiag run all --base-dir ...`. List them (with descriptions) at any
time with `segdiag list-checks`.

## The original six (model output vs. ground truth)

| Check | Question it answers |
|---|---|
| `iou-distribution` | Are missed cells "never detected" or "detected but poorly matched"? |
| `fn-visualize` | What do the missed cells actually look like, in 3D context? |
| `fn-characteristics` | Are missed/spurious cells systematically smaller, dimmer, or deeper (Z)? |
| `detection-probability` | What's the empirical P(detected) curve vs. volume/intensity? |
| `fn-contribution` | Which cell-size bucket drives the missed cells - and the false positives? |
| `story-closure` | Combined volume x intensity interaction heatmap + final summary |

## New in 1.0 (input data quality + FP root-cause)

| Check | Question it answers |
|---|---|
| `raw-image-quality` | Is low recall a modeling problem, or is the raw image itself noisy/blurry/saturated? |
| `gt-annotation-quality` | Does the GT itself have volume outliers, border-truncated cells, or density anomalies (annotator drift)? |
| `fp-root-cause` | Is a false positive noise, a same-cell over-segmentation split, or a genuine hallucination? |

`fp-root-cause`'s three buckets - `noise_fp`, `boundary_split_fp`,
`hallucination_fp` - are computed once during `collect()`
(`segdiag.checks.fp_root_cause.classify_fp_subtype`) and stored on every
false-positive row's `fp_subtype` column, so any other check can also group
by it.
