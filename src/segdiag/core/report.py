"""Standardized output container every check returns.

Every :class:`segdiag.checks.base.Check` produces a list of
:class:`ReportArtifact` instead of deciding for itself what file format to
save or what path to save to - that decision belongs to
:mod:`segdiag.core.writers`, driven by the CLI's ``--format`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class ReportArtifact:
    """One table (always) plus an optional one figure.

    ``name`` becomes the output filename stem (e.g.
    ``"step3_fn_characteristics"``); ``metadata`` carries context that isn't
    itself tabular data - the filter/threshold values a check was run with,
    for instance - so a writer or downstream reader can tell how a result
    was produced without re-deriving it.
    """

    name: str
    table: pd.DataFrame
    figure: Optional["Figure"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
