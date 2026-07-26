# segdiag

**A model-evaluation framework for false-negative analysis of 3D instance
segmentation models on microscopy data.**

segdiag answers one recurring research question: *"Why is my segmentation
model missing cells (low instance recall), and which cells are most
responsible for it?"* — and, since the 1.0 rearchitecture, two more that
used to have no tooling at all: *"Is the raw input data itself the
problem?"* and *"When the model hallucinates a detection, what kind of
mistake is it?"*

## Quick start

```bash
pip install -e ".[dev]"
segdiag run all --base-dir /path/to/dataset --output-dir /path/to/results
```

Run a single check instead of everything:

```bash
segdiag run iou-distribution --base-dir /path/to/dataset
segdiag list-checks   # see every registered check name + description
```

See [Architecture](architecture.md) for how a run actually flows, [Checks](checks.md)
for what each one answers, [Configuration](configuration.md) for `segdiag.toml`,
and [Migrating from 0.x](migration.md) if you're coming from the old
`segdiag <step-name>` CLI.
