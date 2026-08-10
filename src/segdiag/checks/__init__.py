"""The pluggable diagnostic checks of the segdiag pipeline.

Every check consumes the tables :func:`segdiag.core.pipeline.collect`
produces and returns a list of :class:`segdiag.core.report.ReportArtifact`.
Adding a new check needs exactly two changes: a new module implementing
:class:`segdiag.checks.base.Check`, and one line registering an instance of
it here - no CLI or I/O wiring required.
"""

from __future__ import annotations

from typing import Dict

from segdiag.checks.base import Check
from segdiag.checks.cell_count_agreement import CellCountAgreementCheck
from segdiag.checks.detection_probability import DetectionProbabilityCheck
from segdiag.checks.fn_characteristics import FnCharacteristicsCheck
from segdiag.checks.fn_contribution import FnContributionCheck
from segdiag.checks.fn_visualization import FnVisualizationCheck
from segdiag.checks.fp_root_cause import FpRootCauseCheck
from segdiag.checks.fp_visualization import FpVisualizationCheck
from segdiag.checks.gt_annotation_quality import GtAnnotationQualityCheck
from segdiag.checks.iou_distribution import IouDistributionCheck
from segdiag.checks.raw_image_quality import RawImageQualityCheck
from segdiag.checks.representative_case_gallery import RepresentativeCaseGalleryCheck
from segdiag.checks.story_closure import StoryClosureCheck

CHECKS: Dict[str, Check] = {
    c.name: c
    for c in [
        IouDistributionCheck(),
        FnVisualizationCheck(),
        FnCharacteristicsCheck(),
        DetectionProbabilityCheck(),
        FnContributionCheck(),
        StoryClosureCheck(),
        RawImageQualityCheck(),
        GtAnnotationQualityCheck(),
        FpRootCauseCheck(),
        FpVisualizationCheck(),
        CellCountAgreementCheck(),
        RepresentativeCaseGalleryCheck(),
    ]
}
