"""Unit tests for segdiag.core.schema."""

from dataclasses import asdict

import pandas as pd

from segdiag.core.schema import ImageQualityRecord, InstanceRecord


def _make_instance_record(**overrides) -> InstanceRecord:
    defaults = dict(
        dataset="my_dataset",
        sample="case01",
        model="unet_v9",
        slice_name="slice_0001.tif",
        z_index=0,
        role="gt",
        instance_id=1,
        volume=120,
        mean_intensity=42.0,
        centroid_y=10.5,
        centroid_x=20.5,
        bbox_min_row=5,
        bbox_min_col=15,
        bbox_max_row=15,
        bbox_max_col=25,
        classification="true_positive",
        best_iou=0.9,
        matched_instance_id=1,
    )
    defaults.update(overrides)
    return InstanceRecord(**defaults)


def test_instance_record_is_frozen():
    record = _make_instance_record()
    try:
        record.volume = 999  # type: ignore[misc]
        assert False, "InstanceRecord should be immutable"
    except AttributeError:
        pass


def test_instance_record_fp_subtype_defaults_to_none():
    record = _make_instance_record()
    assert record.fp_subtype is None


def test_instance_record_converts_to_dataframe_row_via_asdict():
    record = _make_instance_record()
    df = pd.DataFrame([asdict(record)])
    assert df.loc[0, "dataset"] == "my_dataset"
    assert df.loc[0, "classification"] == "true_positive"
    assert df.loc[0, "fp_subtype"] is None


def test_image_quality_record_fields_round_trip_through_dataframe():
    record = ImageQualityRecord(
        dataset="my_dataset",
        sample="case01",
        slice_name="slice_0001.tif",
        z_index=0,
        source="raw",
        mean_intensity=100.0,
        std_intensity=12.0,
        snr_estimate=8.3,
        saturation_pct=0.01,
        laplacian_variance=55.5,
        gt_instance_count=3,
        gt_touching_border_pct=0.0,
    )
    df = pd.DataFrame([asdict(record)])
    assert df.loc[0, "source"] == "raw"
    assert df.loc[0, "gt_instance_count"] == 3
