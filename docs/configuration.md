# Configuration (`segdiag.toml`)

Drop an `segdiag.toml` in the directory you run `segdiag` from to pin your
folder-naming convention and default output settings instead of repeating
the same flags every time. Copy `segdiag.toml.example` to get started:

```toml
[dataset]
raw_name = "Flatten_561"
dark_name = "Flatten_561_dark"
mask_name = "Flatten_561_mask"

[thresholds]
tp_iou = 0.5
blind_fn_iou = 0.05
fp_boundary_split_distance = 20.0
connectivity = 26      # cc3d/MARS convention (6/18/26); MARS alignment uses 18
min_volume = 40         # MARS TH-marker volume filter lower bound (voxels, open interval)
max_volume = 10000      # MARS TH-marker volume filter upper bound (voxels, open interval)

[output]
format = ["csv", "html"]
output_dir = "results/"
```

**An explicit CLI flag always wins.** A config value only fills in a
default for a flag you didn't pass at all - so `segdiag run all --base-dir
data/ --format parquet` still writes parquet even if `segdiag.toml` says
`format = ["csv"]`.

`connectivity`/`min_volume`/`max_volume` are threaded all the way through to
`segdiag.core.matching`/`segdiag.core.pipeline` (via the `--connectivity`/
`--min-volume`/`--max-volume` CLI flags) - see `SEGDIAG_MARS_CONNECTIVITY.md`
and `SEGDIAG_MARS_ALIGNMENT_COMPLETE.md`. `min_volume`/`max_volume` default
to `40`/`10000` (MARS's TH-marker volume filter, an open interval applied
after connected-component labeling and before instance matching) - pass `0`
on the CLI, or `min_volume = 0`/`max_volume = 0` in `segdiag.toml`, to
disable one side.

!!! note
    The rest of `[thresholds]` (`tp_iou`, `blind_fn_iou`,
    `fp_boundary_split_distance`) is parsed and validated (a typo fails
    fast), but is not yet threaded through to the matching/collect/
    fp-root-cause internals - those still use their own module-level
    constants (`segdiag.core.matching.TP_IOU_THRESHOLD`,
    `segdiag.checks.fp_root_cause.BOUNDARY_SPLIT_DISTANCE_THRESHOLD`, etc).
    Wiring the remaining configurable thresholds all the way through is a
    natural next step, not yet done.

Python 3.11+ uses the standard library's `tomllib`; 3.9/3.10 fall back to
the `tomli` backport (installed automatically as a dependency).
