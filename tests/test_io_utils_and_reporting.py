"""Unit tests for segdiag.core.io_utils filtering and segdiag.core.reporting
output-path tagging.
"""

from pathlib import Path

from segdiag.core.io_utils import (
    extract_model_name,
    find_gt_folders,
    find_pred_folders,
    model_matches_exact,
    model_name_from_pred_dir,
    resolve_image_dir,
)
from segdiag.core.reporting import annotated_path, annotated_subdir, source_tag


def _make_dataset(root: Path) -> None:
    for sample in ("case01", "case02"):
        sample_dir = root / sample
        (sample_dir / "Flatten_561_mask").mkdir(parents=True)
        (sample_dir / "unet_v8.scroll-tiff").mkdir(parents=True)
        (sample_dir / "unet_v9.scroll-tiff").mkdir(parents=True)


def test_find_gt_folders_no_filter_returns_every_sample(tmp_path):
    _make_dataset(tmp_path)
    folders = find_gt_folders(tmp_path)
    assert {f.parent.name for f in folders} == {"case01", "case02"}


def test_find_gt_folders_sample_filter_restricts_to_match(tmp_path):
    _make_dataset(tmp_path)
    folders = find_gt_folders(tmp_path, sample_filter="case01")
    assert {f.parent.name for f in folders} == {"case01"}


def test_find_gt_folders_sample_filter_accepts_comma_list(tmp_path):
    _make_dataset(tmp_path)
    (tmp_path / "case03" / "Flatten_561_mask").mkdir(parents=True)
    folders = find_gt_folders(tmp_path, sample_filter="case01,case03")
    assert {f.parent.name for f in folders} == {"case01", "case03"}


def test_find_gt_folders_sample_filter_no_match_returns_empty(tmp_path):
    _make_dataset(tmp_path)
    folders = find_gt_folders(tmp_path, sample_filter="does-not-exist")
    assert folders == []


def test_find_pred_folders_no_filter_returns_every_model(tmp_path):
    _make_dataset(tmp_path)
    folders = find_pred_folders(tmp_path / "case01")
    assert {f.name for f in folders} == {"unet_v8.scroll-tiff", "unet_v9.scroll-tiff"}


def test_find_pred_folders_model_filter_restricts_to_match(tmp_path):
    _make_dataset(tmp_path)
    folders = find_pred_folders(tmp_path / "case01", model_filter="v9")
    assert [f.name for f in folders] == ["unet_v9.scroll-tiff"]


def test_model_name_from_pred_dir_strips_suffix(tmp_path):
    pred_dir = tmp_path / "unet_v9.scroll-tiff"
    assert model_name_from_pred_dir(pred_dir) == "unet_v9"


def test_model_matches_exact_no_filter_matches_everything():
    assert model_matches_exact("v12_unet_baseline", None) is True


def test_model_matches_exact_rejects_substring_sibling():
    # Unlike the substring `--model` filter, an exact match on
    # "v12_unet_baseline" must not also accept "v12_unet_baseline_dark".
    assert model_matches_exact("v12_unet_baseline_dark", "v12_unet_baseline") is False
    assert model_matches_exact("v12_unet_baseline", "v12_unet_baseline") is True


def test_model_matches_exact_accepts_comma_separated_list_case_insensitive():
    assert model_matches_exact("V12_Unet_Baseline", "v9,v12_unet_baseline") is True
    assert model_matches_exact("v13_unet_baseline", "v9,v12_unet_baseline") is False


def test_source_tag_reflects_filters(tmp_path):
    root = tmp_path / "my_dataset"
    root.mkdir()
    tag = source_tag(root, sample_filter="case01", model_filter="v9")
    assert tag == "my_dataset__case01__v9"


def test_source_tag_defaults_when_no_filters(tmp_path):
    root = tmp_path / "my_dataset"
    root.mkdir()
    tag = source_tag(root, sample_filter=None, model_filter=None)
    assert tag == "my_dataset__all-samples__all-models"


def test_annotated_path_prefixes_filename_with_source_tag(tmp_path):
    root = tmp_path / "my_dataset"
    root.mkdir()
    out_dir = tmp_path / "results"
    path = annotated_path(out_dir, root, "case01", "v9", "iou_distribution_analysis.png")
    assert path == out_dir / "my_dataset__case01__v9__iou_distribution_analysis.png"


def test_annotated_path_different_scopes_never_collide(tmp_path):
    root = tmp_path / "my_dataset"
    root.mkdir()
    out_dir = tmp_path / "results"
    path_v8 = annotated_path(out_dir, root, None, "v8", "report.png")
    path_v9 = annotated_path(out_dir, root, None, "v9", "report.png")
    assert path_v8 != path_v9


def test_annotated_subdir_is_created_and_tagged(tmp_path):
    root = tmp_path / "my_dataset"
    root.mkdir()
    out_dir = tmp_path / "results"
    sub = annotated_subdir(out_dir, root, "case01", None, "fn_diagnosis_3d")
    assert sub.exists()
    assert sub.name == "fn_diagnosis_3d__my_dataset__case01__all-models"


# --- analysis.py-aligned dynamic discovery -----------------------------------


def _make_analysis_style_dataset(root: Path) -> None:
    """Mirrors the real production layout: prediction folders named
    ``{image_folder}_{model}_mask.scroll-tif`` (or ``.scroll-tiff``).
    """
    sample_dir = root / "case01"
    (sample_dir / "Flatten_561").mkdir(parents=True)
    (sample_dir / "Flatten_561_mask").mkdir(parents=True)
    (sample_dir / "Flatten_561_unet_v8_mask.scroll-tif").mkdir(parents=True)
    (sample_dir / "Flatten_561_unet_v9_mask.scroll-tiff").mkdir(parents=True)


def test_find_pred_folders_accepts_both_scroll_tif_and_tiff(tmp_path):
    _make_analysis_style_dataset(tmp_path)
    folders = find_pred_folders(tmp_path / "case01")
    assert {f.name for f in folders} == {
        "Flatten_561_unet_v8_mask.scroll-tif",
        "Flatten_561_unet_v9_mask.scroll-tiff",
    }


def test_resolve_image_dir_strips_mask_suffix_by_default(tmp_path):
    _make_analysis_style_dataset(tmp_path)
    gt_dir = tmp_path / "case01" / "Flatten_561_mask"
    img_dir = resolve_image_dir(gt_dir)
    assert img_dir.name == "Flatten_561"


def test_resolve_image_dir_prefers_explicit_raw_name(tmp_path):
    _make_analysis_style_dataset(tmp_path)
    (tmp_path / "case01" / "custom_raw").mkdir()
    gt_dir = tmp_path / "case01" / "Flatten_561_mask"
    img_dir = resolve_image_dir(gt_dir, raw_name="custom_raw")
    assert img_dir.name == "custom_raw"


def test_resolve_image_dir_falls_back_to_other_non_mask_folder(tmp_path):
    sample_dir = tmp_path / "case01"
    (sample_dir / "weird_mask").mkdir(parents=True)
    (sample_dir / "images").mkdir(parents=True)
    img_dir = resolve_image_dir(sample_dir / "weird_mask")
    assert img_dir.name == "images"


def test_resolve_image_dir_returns_none_when_nothing_matches(tmp_path):
    sample_dir = tmp_path / "case01"
    (sample_dir / "only_mask").mkdir(parents=True)
    assert resolve_image_dir(sample_dir / "only_mask") is None


def test_extract_model_name_matches_analysis_py_convention(tmp_path):
    assert extract_model_name("Flatten_561", "Flatten_561_unet_v9_mask.scroll-tif") == "unet_v9"


def test_extract_model_name_falls_back_gracefully_for_legacy_convention(tmp_path):
    # The old segdiag README convention: a bare "<model>.scroll-tiff" folder
    # that doesn't follow the {image}_{model}_mask pattern at all.
    assert extract_model_name("Flatten_561", "unet_v9.scroll-tiff") == "unet_v9"


def test_model_name_from_pred_dir_uses_extract_model_name(tmp_path):
    pred_dir = tmp_path / "Flatten_561_unet_v9_mask.scroll-tif"
    assert model_name_from_pred_dir(pred_dir, img_dir_name="Flatten_561") == "unet_v9"

