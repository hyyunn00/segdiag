"""``segdiag.toml`` configuration loading.

Lets a project pin its folder-naming convention and default output settings
once instead of repeating the same CLI flags on every invocation. CLI flags
always win: :mod:`segdiag.cli` only falls back to a config value when the
corresponding flag wasn't passed at all.

``thresholds`` is parsed and validated here (so a typo in ``segdiag.toml``
fails fast) but is not yet threaded through to
:mod:`segdiag.core.matching`/:mod:`segdiag.core.pipeline` or
:mod:`segdiag.checks.fp_root_cause` - those still use their own module-level
constants. Wiring configurable thresholds all the way through is future
work; this module is the config-loading foundation for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.9/3.10
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_CONFIG_FILENAME = "segdiag.toml"


@dataclass
class DatasetConfig:
    raw_name: Optional[str] = None
    dark_name: Optional[str] = None
    mask_name: Optional[str] = None


@dataclass
class ThresholdsConfig:
    tp_iou: float = 0.5
    blind_fn_iou: float = 0.05
    fp_boundary_split_distance: float = 20.0


@dataclass
class OutputConfig:
    format: List[str] = field(default_factory=lambda: ["csv"])
    output_dir: Optional[str] = None


@dataclass
class SegdiagConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: Optional[Path] = None) -> SegdiagConfig:
    """Load ``segdiag.toml`` (or an explicit ``path``).

    Returns all-defaults (equivalent to no config at all) if the file
    doesn't exist - segdiag works with zero configuration.
    """
    if path is None:
        path = Path.cwd() / DEFAULT_CONFIG_FILENAME
    if not path.exists():
        return SegdiagConfig()

    with path.open("rb") as fh:
        data = tomllib.load(fh)

    return SegdiagConfig(
        dataset=DatasetConfig(**data.get("dataset", {})),
        thresholds=ThresholdsConfig(**data.get("thresholds", {})),
        output=OutputConfig(**data.get("output", {})),
    )
