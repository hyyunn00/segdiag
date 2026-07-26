"""Check abstract base class.

Every check is: given the tables :func:`segdiag.core.pipeline.collect`
already produced, compute zero or more standardized
:class:`~segdiag.core.report.ReportArtifact` objects. A check never scans
folders, reads TIFFs (with the deliberate exception of ``fn_visualize``,
which needs dynamic 3D-context crops - see its module docstring), or decides
how its own output gets saved; that's the collect stage and the writers'
job, respectively.

``args`` is a plain ``argparse.Namespace`` carrying the settings every check
might need (``root``, ``sample``, ``model``, ``output_dir``, ``raw_name``,
``dark_name``, ``mask_name``, ``num_samples``) - built once by the CLI from
its (typer-parsed) options and passed identically to every check, rather
than each check registering its own CLI flags. This keeps "add a new check"
to "one new file + one registry line" (see ``checks/__init__.py``): no
argparse/typer wiring required.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from segdiag.core.report import ReportArtifact


class Check(ABC):
    name: str
    description: str

    #: Whether ``segdiag run all`` includes this check automatically.
    #: ``False`` means the check is opt-in only - it still runs (and still
    #: shows up in ``list-checks``) when named explicitly, e.g.
    #: ``segdiag run fn-visualize --base-dir ...``, it just doesn't run as
    #: part of ``all``. Use this for checks that are expensive/slow relative
    #: to the tabular checks (e.g. ones that render and save a gallery of
    #: figures) rather than something every "just run everything" call
    #: should pay for by default.
    default_enabled: bool = True

    @abstractmethod
    def run(
        self,
        instances: pd.DataFrame,
        quality: pd.DataFrame,
        args: argparse.Namespace,
    ) -> List[ReportArtifact]: ...
