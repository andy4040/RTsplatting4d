import json

import numpy as np
import pytest

from rt_port.evaluation import freeze_validation_roi, measure_view, measure_test_view, roi_sha256
from rt_port.selection import decide


def zeros(h=32, w=32):
    return np.zeros((3, h, w), dtype=np.float32)


def test_freeze_labels_connected_high_error_patches_once():
    prediction, target = zeros(), zeros()
    prediction[:, :8, :16] = 1.
    prediction[:, 24:, 24:] = .8
    labels, info = freeze_validation_roi(prediction, target, image="cam00", patch_size=8)
    assert info["regions"] == 2
    assert info["candidate_pixels"] == 192
    assert labels[0, 0] == labels[0, 8] != 0
    assert labels[31, 31] != labels[0, 0]
    assert labels[16, 16] == 0
    assert not labels.flags.writeable
    assert info["training_use"] is False
    assert info["roi_sha256"] == roi_sha256(labels)
    assert "cam00:1" in info["region_pixels"]
    json.dumps(info, allow_nan=False)


def test_partial_border_patch_uses_its_actual_area():
    prediction, target = zeros(5, 5), zeros(5, 5)
    prediction[:, 4, 4] = 1.
    labels, info = freeze_validation_roi(prediction, target, image="cam00", patch_size=4)
    assert info["baseline_full_mse"] == pytest.approx(.04)
    assert info["candidate_pixels"] == 1
    assert labels[4, 4] == 1


def test_empty_perfect_image_does_not_force_candidate():
    labels, info = freeze_validation_roi(zeros(), zeros(), image="cam00")
    assert info["regions"] == 0
    assert info["candidate_pixels"] == 0
    view = measure_view(zeros(), zeros(), zeros(), labels, image="cam00")
    assert view["regions"] == []
    assert view["outside_pixels"] == 1024
    assert view["full_control_mse"] == 0
    json.dumps(view, allow_nan=False)


def test_rawfloat_metrics_clamp_predictions_and_average_rgb_channels():
    control, rt, target = zeros(2, 2), zeros(2, 2), zeros(2, 2)
    control[0] = 2.
    control[1] = -1.
    control[2] = .5
    labels = np.ones((2, 2), dtype=bool)
    view = measure_view(control, rt, target, labels, image="cam00")
    assert view["full_control_mse"] == pytest.approx((1. + .25) / 3.)
    assert view["regions"][0]["control_mse"] == view["full_control_mse"]
    assert view["outside_pixels"] == 0
    assert view["outside_control_mse"] == 0
    assert view["outside_rt_mse"] == 0


def test_float_subjpeg_precision_is_preserved():
    target = np.full((3, 2, 2), .1, dtype=np.float64)
    control = target + .0001
    view = measure_view(control, target, target, np.ones((2, 2), np.int32), image="cam00")
    assert view["full_control_mse"] == pytest.approx(1e-8)
    assert view["full_control_mse"] > 0


def test_measurement_hash_protects_fixed_mask_pixels_not_just_counts():
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0, 0] = 1
    digest = roi_sha256(labels)
    labels[0, 0], labels[0, 1] = 0, 1
    with pytest.raises(ValueError, match="hash changed"):
        measure_view(zeros(4, 4), zeros(4, 4), zeros(4, 4), labels,
                     image="cam00", expected_roi_sha256=digest)


def test_label_hash_is_dtype_stable_but_spatially_sensitive():
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)
    assert roi_sha256(labels) == roi_sha256(labels.astype(">i8"))
    assert roi_sha256(labels) != roi_sha256(labels.reshape(1, 4))


def test_boolean_mask_uses_four_connected_components_and_namespaced_ids():
    labels = np.eye(3, dtype=bool)
    view = measure_view(zeros(3, 3), zeros(3, 3), zeros(3, 3), labels, image="cam09")
    assert len(view["regions"]) == 3
    assert [r["id"] for r in view["regions"]] == ["cam09:1", "cam09:2", "cam09:3"]
    assert all(r["pixels"] == 1 for r in view["regions"])


def test_hwc_and_chw_have_identical_metrics_and_hashes():
    control, target = zeros(), zeros()
    control[:, :8, :8] = .5
    labels_chw, info_chw = freeze_validation_roi(control, target, image="cam00", patch_size=8)
    labels_hwc, info_hwc = freeze_validation_roi(
        np.moveaxis(control, 0, -1), np.moveaxis(target, 0, -1),
        image="cam00", patch_size=8, channel_axis=-1)
    np.testing.assert_array_equal(labels_chw, labels_hwc)
    assert info_chw == info_hwc


@pytest.mark.parametrize("split", ["test", "train"])
def test_no_nonvalidation_split_in_selection_measurements(split):
    with pytest.raises(ValueError, match="never train/test"):
        measure_view(zeros(), zeros(), zeros(), np.zeros((32, 32), np.int32), image="cam19", split=split)


def test_explicit_final_test_metrics_match_validation_but_never_enter_selection():
    target = zeros()
    control = target + .2
    rt = target + .1
    labels = np.ones((32, 32), np.int32)
    test_view = measure_test_view(control, rt, target, labels, image="cam19")
    validation_view = measure_view(control, rt, target, labels, image="cam19")
    assert test_view["split"] == "test"
    assert {k: v for k, v in test_view.items() if k != "split"} == {
        k: v for k, v in validation_view.items() if k != "split"}
    with pytest.raises(ValueError, match="only split='validation'"):
        decide([{"iteration": 3000, "views": [test_view]}])


def test_final_test_full_image_report_needs_no_test_candidate_mining():
    view = measure_test_view(zeros(), zeros(), zeros(), np.zeros((32, 32), np.int32), image="cam19")
    assert view["split"] == "test"
    assert view["regions"] == []
    assert view["full_control_mse"] == view["full_rt_mse"] == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_prediction_rejected_before_clipping(bad):
    control = zeros()
    control[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite before clamping"):
        freeze_validation_roi(control, zeros(), image="cam00")


def test_uint8_and_out_of_range_gt_are_not_silently_accepted():
    with pytest.raises(ValueError, match="floating-point"):
        freeze_validation_roi(np.zeros((3, 8, 8), np.uint8), zeros(8, 8), image="cam00")
    target = zeros()
    target[0, 0, 0] = 2.
    with pytest.raises(ValueError, match="Ground truth"):
        freeze_validation_roi(zeros(), target, image="cam00")


def test_roi_and_image_shapes_and_label_values_are_validated():
    with pytest.raises(ValueError, match="dimensions"):
        measure_view(zeros(), zeros(), zeros(), np.zeros((3, 3), np.int32), image="cam00")
    with pytest.raises(ValueError, match="nonnegative"):
        roi_sha256(np.full((2, 2), -1, np.int32))
    with pytest.raises(ValueError, match="integer"):
        roi_sha256(np.full((2, 2), .5))


def test_frozen_baseline_metrics_integrate_with_multiview_whole_model_selector():
    target, control = zeros(), zeros()
    control[:, :16, :16] = .2
    treatment = control * .5
    frozen = {image: freeze_validation_roi(control, target, image=image, patch_size=8)
              for image in ("cam00", "cam09")}
    history = []
    for iteration in (1000, 2000, 3000):
        views = []
        for image, (labels, info) in frozen.items():
            views.append(measure_view(control, treatment, target, labels, image=image,
                                      expected_roi_sha256=info["roi_sha256"]))
        history.append({"iteration": iteration, "views": views})
    result = decide(history, {"require_roi_hash": True})
    assert result["selected_model"] == "rt_ex4dgs"
    assert result["latest"]["candidate"]["gain_db"] == pytest.approx(6.020599913)
    assert len(result["per_region"]) == 2  # Camera components are not merged as one physical object.
    json.dumps(result, allow_nan=False)
