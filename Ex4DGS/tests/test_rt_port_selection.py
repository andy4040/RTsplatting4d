import copy
import math

import pytest

from rt_port.selection import coredecide, decide, psnr_gain


def make_view(image, candidate_rt=.0075, outside_rt=.01, pixels=256):
    outside_pixels = 1024
    return {
        "image": image, "split": "validation", "roi_sha256": "a" * 64,
        "full_control_mse": .01,
        "full_rt_mse": (pixels * candidate_rt + outside_pixels * outside_rt) / (pixels + outside_pixels),
        "regions": [{"id": "window", "pixels": pixels, "control_mse": .01, "rt_mse": candidate_rt}],
        "outside_pixels": outside_pixels,
        "outside_control_mse": .01, "outside_rt_mse": outside_rt,
    }


def make_history(length=3):
    return [{"iteration": (i + 1) * 1000,
             "views": [make_view("cam00"), make_view("cam09")]}
            for i in range(length)]


def test_multiview_repeated_improvement_selects_final_whole_rt_model():
    result = decide(make_history(), {"require_roi_hash": True})
    assert result["accepted"]
    assert result["selected_model"] == "rt_ex4dgs"
    assert result["selection_scope"] == "whole_model"
    assert result["passes"] == 3
    assert result["latest"]["roi_hashes_verified"]
    assert result["per_region"][0]["id"] == "window"
    assert result["per_region"][0]["gain_db"] == pytest.approx(10 * math.log10(.01 / .0075))
    assert result == coredecide(make_history(), {"require_roi_hash": True})


def test_candidate_improvement_cannot_override_worse_whole_image():
    history = make_history()
    for observation in history:
        for view in observation["views"]:
            view["outside_pixels"] = 100000
            view["outside_rt_mse"] = .0101  # Within outside limit, but dominates full image.
            view["full_rt_mse"] = (256 * .0075 + 100000 * .0101) / 100256
    result = decide(history)
    assert result["selected_model"] == "ex4dgs"
    assert "insufficient_global_mean_psnr_gain" in result["latest"]["failure_reasons"]
    assert result["latest"]["candidate"]["gain_db"] > .5


def test_outside_harm_rejects_even_with_global_improvement():
    history = make_history()
    for observation in history:
        observation["views"] = [make_view("cam00", .001, .0103), make_view("cam09", .001, .0103)]
    result = decide(history)
    assert not result["accepted"]
    assert result["latest"]["global_mean_psnr_gain_db"] > 0
    assert "pooled_outside_mse_regression" in result["latest"]["failure_reasons"]


def test_outside_harm_in_one_camera_cannot_hide_behind_another():
    history = make_history()
    for observation in history:
        observation["views"] = [make_view("cam00", .001, .0103), make_view("cam09", .001, .009)]
    result = decide(history)
    assert result["latest"]["outside_relative_mse_increase"] < .02
    assert "outside_mse_regression_in_validation_view" in result["latest"]["failure_reasons"]
    assert not result["accepted"]


@pytest.mark.parametrize("split", ["test", "train", None])
def test_test_or_training_metrics_are_never_used_for_selection(split):
    history = make_history(5)
    history[0]["views"][0]["split"] = split  # Even before the last-three window.
    with pytest.raises(ValueError, match="only split='validation'"):
        decide(history)


def test_empty_regions_or_no_history_cannot_select_rt():
    history = make_history()
    for observation in history:
        for view in observation["views"]:
            view["regions"] = []
    result = decide(history)
    assert not result["accepted"]
    assert result["latest"]["candidate"]["gain_db"] is None
    assert not decide([])["accepted"]
    assert not decide(make_history(2))["accepted"]


def test_tiny_region_and_one_validation_camera_are_insufficient():
    history = make_history()
    for observation in history:
        observation["views"][1]["regions"][0]["pixels"] = 127
    assert not decide(history)["accepted"]
    for observation in history:
        observation["views"] = observation["views"][:1]
    assert not decide(history)["accepted"]


def test_one_camera_candidate_regression_rejects():
    history = make_history()
    for observation in history:
        observation["views"] = [make_view("cam00", .001), make_view("cam09", .0101)]
    result = decide(history)
    assert result["latest"]["candidate"]["gain_db"] > .5
    assert "candidate_regression_in_validation_view" in result["latest"]["failure_reasons"]
    assert not result["accepted"]


def test_final_regression_rejects_even_after_two_passes():
    history = make_history()
    history[-1]["views"] = [make_view("cam00", .02), make_view("cam09", .02)]
    result = decide(history)
    assert result["passes"] == 2
    assert not result["accepted"]
    assert result["reason"].startswith("final_observation_failed")


def test_last_window_and_configured_pass_frequency():
    history = make_history(5)
    history[2]["views"] = [make_view("cam00", .02), make_view("cam09", .02)]
    result = decide(history)
    assert result["accepted"]
    assert result["window_iterations"] == [3000, 4000, 5000]
    assert result["passes"] == 2
    assert not decide(history, {"required_passes": 3})["accepted"]
    assert decide(history, {"decision_observations": 2, "required_passes": 2})["accepted"]
    assert not decide(history, {"min_gain_db": 2.0})["accepted"]


def test_candidate_mse_is_pixel_pooled_not_average_region_db():
    history = make_history()
    for observation in history:
        for view in observation["views"]:
            view["regions"] = [
                {"id": "large", "pixels": 12800, "control_mse": .1, "rt_mse": .1},
                {"id": "tiny", "pixels": 128, "control_mse": .01, "rt_mse": .0001},
            ]
    result = decide(history)
    assert result["latest"]["candidate"]["gain_db"] < .1
    assert not result["accepted"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1])
def test_nonfinite_or_negative_metrics_raise(value):
    history = make_history()
    history[0]["views"][0]["regions"][0]["rt_mse"] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        decide(history)


def test_zero_mse_is_finite_and_equal_perfect_predictions_have_no_gain():
    assert psnr_gain(0, 0) == 0
    assert math.isfinite(psnr_gain(.01, 0))
    assert math.isfinite(psnr_gain(0, .01))


@pytest.mark.parametrize("change", ["pixels", "hash", "region", "image"])
def test_changing_validation_roi_or_camera_is_rejected(change):
    history = make_history()
    view = history[-1]["views"][0]
    if change == "pixels":
        view["regions"][0]["pixels"] += 1
    elif change == "hash":
        view["roi_sha256"] = "b" * 64
    elif change == "region":
        view["regions"][0]["id"] = "different"
    else:
        view["image"] = "cam19"
    with pytest.raises(ValueError, match="ROI identities changed"):
        decide(history)


def test_production_requires_mask_hash_and_duplicate_iterations_are_invalid():
    history = make_history()
    del history[0]["views"][0]["roi_sha256"]
    with pytest.raises(ValueError, match="roi_sha256 is required"):
        decide(history, {"require_roi_hash": True})
    history = make_history()
    history[1]["iteration"] = history[0]["iteration"]
    with pytest.raises(ValueError, match="strictly increasing"):
        decide(history)


def test_invalid_configuration_and_duplicate_regions_raise():
    with pytest.raises(ValueError, match="exceeds"):
        decide([], {"required_passes": 4})
    with pytest.raises(ValueError, match="integer"):
        decide([], {"decision_observations": 1.5})
    history = make_history()
    view = history[0]["views"][0]
    view["regions"].append(copy.deepcopy(view["regions"][0]))
    with pytest.raises(ValueError, match="Region IDs"):
        decide(history)
