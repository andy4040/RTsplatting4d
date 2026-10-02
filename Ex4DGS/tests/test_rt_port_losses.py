"""Check the intentional supervision delta and fidelity of RT loss equations."""

import pytest
import torch

from rt_port.losses import (
    candidate_consistency,
    compute_rt_loss,
    gate_transmission,
    masked_positive_loss,
)
from rt_port.vendor.image_utils import local_variance


def test_positive_mask_has_no_outside_gradient_and_uses_selected_mean():
    opacity = torch.tensor([[[0.2, 0.8], [0.4, 0.6]]], requires_grad=True)
    mask = torch.tensor([[True, False], [False, True]])
    loss = masked_positive_loss(opacity, mask)
    expected = -torch.log(torch.tensor([0.8, 0.4]) + 1e-10).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.equal(opacity.grad[0][~mask], torch.zeros(2))
    assert torch.all(opacity.grad[0][mask] > 0)
    # Padding with unknown pixels cannot dilute the positive supervision.
    padded = torch.nn.functional.pad(opacity.detach(), (0, 4, 0, 4))
    padded_mask = torch.nn.functional.pad(mask, (0, 4, 0, 4), value=False)
    torch.testing.assert_close(masked_positive_loss(padded, padded_mask), loss.detach())


def test_empty_candidate_losses_are_zero_finite_and_have_zero_gradients():
    opacity = torch.rand(1, 3, 4, requires_grad=True)
    transmission = torch.rand(1, 3, 4, requires_grad=True)
    scatter = torch.rand(3, 3, 4, requires_grad=True)
    empty = torch.zeros(1, 3, 4, dtype=torch.bool)
    loss = masked_positive_loss(opacity, empty) + candidate_consistency(transmission, scatter, empty)
    assert loss.item() == 0 and torch.isfinite(loss)
    loss.backward()
    for tensor in (opacity, transmission, scatter):
        assert torch.equal(tensor.grad, torch.zeros_like(tensor))


def test_consistency_matches_original_spatial_sums_and_only_positive_pixels():
    transmission = torch.tensor([[[0.2, 0.9], [0.4, 0.6]]], requires_grad=True)
    scatter = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2).requires_grad_()
    mask = torch.tensor([[True, False], [True, True]])
    tran_values = transmission[0][mask]
    scat_values = scatter[:, mask]
    expected = ((tran_values - tran_values.mean()) ** 2).sum()
    expected = expected + ((scat_values - scat_values.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1).mean()
    actual = candidate_consistency(transmission, scatter, mask)
    torch.testing.assert_close(actual, expected)
    duplicated = candidate_consistency(
        transmission.repeat(1, 1, 2), scatter.repeat(1, 1, 2), mask.repeat(1, 2),
    )
    torch.testing.assert_close(duplicated, 2 * actual)
    actual.backward()
    assert torch.count_nonzero(transmission.grad[0][~mask]) == 0
    assert torch.count_nonzero(scatter.grad[:, ~mask]) == 0


def test_gate_preserves_forward_image_and_suppresses_only_transmission_gradients():
    torch.manual_seed(3)
    transmission = torch.rand(3, 5, 6, requires_grad=True)
    scatter = torch.rand(3, 5, 6, requires_grad=True)
    specular = torch.rand(3, 5, 6, requires_grad=True)
    opacity = torch.full((1, 5, 6), 0.5, requires_grad=True)
    gated = gate_transmission(transmission, scatter, specular, opacity, 4.0)
    torch.testing.assert_close(gated, transmission + scatter + specular)
    expected_gradient = torch.exp(-4.0 * local_variance(specular.detach(), weights=1 - opacity.detach()))
    gated.sum().backward()
    torch.testing.assert_close(transmission.grad, expected_gradient)
    assert torch.all(transmission.grad > 0) and torch.all(transmission.grad < 1)
    torch.testing.assert_close(scatter.grad, torch.ones_like(scatter))
    torch.testing.assert_close(specular.grad, torch.ones_like(specular))
    assert opacity.grad is None


def test_zero_gate_scale_leaves_transmission_gradients_unchanged():
    transmission = torch.rand(3, 4, 4, requires_grad=True)
    gated = gate_transmission(transmission, torch.zeros_like(transmission), torch.rand_like(transmission), torch.ones(1, 4, 4), 0)
    gated.sum().backward()
    torch.testing.assert_close(transmission.grad, torch.ones_like(transmission))


@pytest.mark.parametrize("mask", [torch.ones(2, 2), torch.ones(2, 2, dtype=torch.uint8)])
def test_masks_require_explicit_boolean_semantics(mask):
    with pytest.raises(TypeError, match="boolean"):
        masked_positive_loss(torch.zeros(1, 2, 2), mask)


def test_mask_resolution_mismatch_is_not_silently_broadcast():
    with pytest.raises(ValueError, match="resolution"):
        masked_positive_loss(torch.zeros(1, 3, 4), torch.ones(1, 3, 1, dtype=torch.bool))


def _package():
    transmission = torch.full((3, 4, 5), 0.2, requires_grad=True)
    scatter = torch.full((3, 4, 5), 0.1, requires_grad=True)
    specular = torch.full((3, 4, 5), 0.05, requires_grad=True)
    normal = torch.zeros(3, 4, 5)
    normal[2] = 1
    return {
        "render_tran": transmission,
        "render_scat": scatter,
        "final_tran": transmission,
        "final_scat": scatter,
        "final_spec": specular,
        "final_rendering": transmission + scatter + specular,
        "surface_opacity": torch.full((1, 4, 5), 0.8, requires_grad=True),
        "transmissivity": torch.full((1, 4, 5), 0.7, requires_grad=True),
        "surface_normal": normal,
        "surface_depth_normal": normal.clone(),
        "foreground": torch.ones(1, 4, 5),
        "visibility_filter": torch.tensor([True, False, True]),
    }


def test_composed_loss_omits_opaque_supervision_and_preserves_occupancy_decay():
    package = _package()
    occupancy = torch.tensor([[0.2], [0.9], [0.6]], requires_grad=True)
    loss, metrics = compute_rt_loss(package, torch.zeros(3, 4, 5), torch.zeros(1, 4, 5, dtype=torch.bool), occupancy, {}, 100)
    assert set(metrics) == {"pbr", "occupancy", "norm", "mask", "consistency", "total"}
    assert metrics["occupancy"] == pytest.approx(0.001 * 0.4)
    assert metrics["mask"] == metrics["consistency"] == metrics["norm"] == 0.0
    loss.backward()
    assert torch.isfinite(loss)
    assert package["transmissivity"].grad.abs().sum() == 0
    assert occupancy.grad[1] == 0


def test_no_visible_gaussians_do_not_produce_nan():
    package = _package()
    package["visibility_filter"][:] = False
    loss, metrics = compute_rt_loss(package, torch.zeros(3, 4, 5), torch.zeros(4, 5, dtype=torch.bool), torch.rand(3, 1), {}, 1)
    assert torch.isfinite(loss) and metrics["occupancy"] == 0


def test_phase_schedule_keeps_initial_photo_stages_and_resolves_mask_minus_one():
    package = _package()
    target = torch.zeros(3, 4, 5)
    mask = torch.ones(4, 5, dtype=torch.bool)
    config = {"init_until_iter": 10, "norm_loss_from_iter": 4, "mask_loss_from_iter": -1, "lambda_dssim": 0.0}
    occupancy = torch.ones(3, 1)
    _, early = compute_rt_loss(package, target, mask, occupancy, config, 3)
    _, middle = compute_rt_loss(package, target, mask, occupancy, config, 7)
    _, late = compute_rt_loss(package, target, mask, occupancy, config, 10)
    assert early["diff"] == pytest.approx(0.2)
    assert "norm" not in early and "mask" not in early
    assert middle["diff"] == pytest.approx(0.35)
    assert "norm" in middle and "mask" not in middle
    assert late["pbr"] == pytest.approx(0.35)
    assert late["mask"] > 0 and "consistency" in late
    assert config["mask_loss_from_iter"] == -1  # no config mutation


def test_lpips_schedule_requires_callable_instead_of_silent_disabling():
    package = _package()
    target = torch.zeros(3, 4, 5)
    mask = torch.ones(4, 5, dtype=torch.bool)
    occupancy = torch.ones(3, 1)
    with pytest.raises(ValueError, match="lpips_fn"):
        compute_rt_loss(package, target, mask, occupancy, {}, 15000)
    _, metrics = compute_rt_loss(package, target, mask, occupancy, {}, 15000, lpips_fn=lambda a, b: (a - b).abs().mean())
    assert metrics["lpips"] == pytest.approx(0.01 * 0.35)


@pytest.mark.parametrize("include_none", [False, True])
def test_optional_distortion_requires_actual_renderer_statistic(include_none):
    package = _package()
    if include_none:
        package['volume_dist'] = None
    with pytest.raises(ValueError, match="volume_dist"):
        compute_rt_loss(package, torch.zeros(3, 4, 5), torch.ones(4, 5, dtype=torch.bool), torch.ones(3, 1), {"dist_loss_weight": 0.1}, 1)
