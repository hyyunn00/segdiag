## Summary

<!-- What does this PR change, and why? -->

## Checklist

- [ ] `pytest` passes locally (`pytest --cov=segdiag --cov-report=term-missing`)
- [ ] `ruff check src tests` and `black --check src tests` pass
- [ ] `mypy src/segdiag` passes
- [ ] If this adds a new check: it's registered in `segdiag/checks/__init__.py`
      and has a test under `tests/test_checks/`
- [ ] If this changes existing output values/columns/filenames: noted in
      `CHANGELOG.md` under an `### Changed` (or `### Breaking`) heading

## Test plan

<!-- How did you verify this works? -->
