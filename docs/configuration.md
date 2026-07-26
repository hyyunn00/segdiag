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

[output]
format = ["csv", "html"]
output_dir = "results/"
```

**An explicit CLI flag always wins.** A config value only fills in a
default for a flag you didn't pass at all - so `segdiag run all --base-dir
data/ --format parquet` still writes parquet even if `segdiag.toml` says
`format = ["csv"]`.

!!! note
    `[thresholds]` is parsed and validated (a typo fails fast), but is not
    yet threaded through to the matching/collect/fp-root-cause internals -
    those still use their own module-level constants
    (`segdiag.core.matching.TP_IOU_THRESHOLD`,
    `segdiag.checks.fp_root_cause.BOUNDARY_SPLIT_DISTANCE_THRESHOLD`, etc).
    Wiring configurable thresholds all the way through is a natural next
    step, not yet done.

Python 3.11+ uses the standard library's `tomllib`; 3.9/3.10 fall back to
the `tomli` backport (installed automatically as a dependency).
