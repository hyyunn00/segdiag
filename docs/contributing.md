# Contributing

```bash
git clone <repo>
cd segdiag
pip install -e ".[dev]"
pre-commit install   # optional but recommended - runs ruff/black/mypy on commit
```

## Running checks locally

```bash
pytest --cov=segdiag --cov-report=term-missing   # tests + coverage (gate: 80%)
ruff check src tests
black --check src tests
mypy src/segdiag
```

All four run in CI (`.github/workflows/ci.yml`) on Python 3.9-3.12.

## Adding a new check

1. Create `src/segdiag/checks/<name>.py` implementing
   `segdiag.checks.base.Check` (`name`, `description`, `run(instances,
   quality, args) -> list[ReportArtifact]`).
2. Register an instance of it in `src/segdiag/checks/__init__.py`'s `CHECKS`
   dict.
3. Add `tests/test_checks/test_<name>.py`. If you need synthetic data, the
   shared `collected_dataset` fixture in `tests/test_checks/conftest.py`
   already has enough variety (volume/intensity spread, planted outliers,
   planted FP subtypes) for decile/quartile-based checks.

No CLI or I/O wiring is needed - `segdiag run <name> --base-dir ...` and
`segdiag run all` pick it up automatically once it's registered.

## Docs

```bash
pip install -e ".[docs]"
mkdocs serve
```
