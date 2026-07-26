"""segdiag: a model-evaluation framework for false-negative analysis of 3D
instance segmentation models on microscopy image stacks.

segdiag answers one recurring research question: *"Why is my segmentation
model missing cells (low instance recall), and which cells are most
responsible for it?"* - plus, since the 1.0 rearchitecture, two more:
*"is the raw input data itself the problem?"* and *"when the model
hallucinates a detection, what kind of mistake is it?"*

The toolkit is organized as three stages - see ``segdiag.core.pipeline``
(collect), ``segdiag.checks`` (pluggable analysis + report), and
``segdiag.core.writers`` (csv/parquet/html output) - fronted by a single
``segdiag run <check|all> --base-dir ...`` CLI (``segdiag.cli``). Run
``segdiag list-checks`` to see every registered check.

Every check's TP/FN/FP membership ultimately comes from a single,
well-tested instance-matching implementation in ``segdiag.core.matching``,
computed with the same one-to-one greedy matching algorithm, iteration
order, and IoU threshold as the lab's own ``metrics.py``
(``compute_object_metrics``), so segdiag's diagnostics agree with the
official evaluation numbers on what counts as a detection.
"""

__version__ = "1.0.0"
