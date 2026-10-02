"""RT-Splatting training losses with positive-only candidate supervision.

The equations and stage ordering follow RT-Splatting ``train.py`` at upstream
3f45b3cac4be04db9f3092234666b695991b268a. The intentional changes are (1) mask
BCE averaged only over candidate pixels, (2) no opaque/transmissivity target
outside candidates, and (3) empty selections contribute differentiable zero.
Consistency retains the original spatial SUM, including averaging the three
scatter channel sums. This module does not infer candidates or select models.
"""

from collections.abc import Mapping

import torch

from utils.loss_utils import l1_loss, ssim
from rt_port.vendor.image_utils import local_variance


def _single_channel(image, name):
    if image.ndim == 3 and image.shape[0] == 1:
        return image[0]
    if image.ndim == 2:
        return image
    raise ValueError(f"{name} must have shape [H,W] or [1,H,W]")


def _candidate_pixels(mask, reference):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("candidate_mask must be a boolean tensor; no implicit thresholding")
    mask = _single_channel(mask, "candidate_mask")
    if mask.shape != reference.shape:
        raise ValueError("candidate_mask must match the rendered image resolution")
    if mask.device != reference.device:
        raise ValueError("candidate_mask and rendered image must be on the same device")
    return mask


def masked_positive_loss(surface_opacity, mask):
    """Official BCE(1-opacity, 1), averaged only over positive candidates.

    Pixels outside ``mask`` provide no transparent OR opaque supervision.
    Uses the official BCE epsilon rather than PyTorch BCE's log clamping.
    """
    opacity = _single_channel(surface_opacity, "surface_opacity")
    selected = opacity[_candidate_pixels(mask, opacity)]
    if selected.numel() == 0:
        return opacity.sum() * 0.0
    return -torch.log(1.0 - selected + 1e-10).mean()


def candidate_consistency(transmissivity, scatter, mask):
    """Official global-within-image consistency, not per component or mean.

    All disconnected candidates in this image share the same means, exactly
    as the official code. Splitting by material/component is a separate change.
    """
    transmission = _single_channel(transmissivity, "transmissivity")
    if scatter.ndim != 3 or scatter.shape[0] != 3 or scatter.shape[1:] != transmission.shape:
        raise ValueError("scatter must have shape [3,H,W] matching transmissivity")
    candidates = _candidate_pixels(mask, transmission)
    transmission_values = transmission[candidates]
    if transmission_values.numel() == 0:
        return transmission.sum() * 0.0 + scatter.sum() * 0.0
    transmission_term = (transmission_values - transmission_values.mean()).square().sum()
    scatter_values = scatter[:, candidates]
    scatter_term = (
        (scatter_values - scatter_values.mean(dim=-1, keepdim=True))
        .square().sum(dim=-1).mean()
    )
    return transmission_term + scatter_term


def gate_transmission(final_tran, final_scat, final_spec, surface_opacity, local_var_scale=4.0):
    """Official specular-aware gradient gate; forward radiance is unchanged.

    The vendored official ``local_variance`` returns a weighted local standard
    deviation despite its name. Do not replace it with torch.var or square it.
    The gate is detached so it changes transmission gradients only.
    """
    opacity = _single_channel(surface_opacity, "surface_opacity").unsqueeze(0)
    spec_variance = local_variance(final_spec, weights=1 - opacity.detach())
    spec_complexity = 1 - torch.exp(-local_var_scale * spec_variance.detach())
    detached_tran = final_tran.detach() * spec_complexity + final_tran * (1 - spec_complexity)
    return final_scat + detached_tran + final_spec


def _option(config, key, default):
    return config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)


def compute_rt_loss(package, target, candidate_mask, occupancy, config, iteration, *, lpips_fn=None):
    """Return ``(loss_tensor, detached_float_metrics)`` for one RT phase step.

    ``iteration`` is the RT continuation phase's iteration, NOT an implicitly
    added Ex4DGS checkpoint iteration. The runner must record this schedule.
    ``config`` uses official RT-Splatting loss option names and defaults.
    ``package`` uses official renderer keys. LPIPS is dependency-injected so
    importing this module does not load/download a pretrained network. When
    its scheduled nonzero term is reached, omitting ``lpips_fn`` is an error,
    never a silent change of objective.
    """
    if iteration < 0:
        raise ValueError("iteration must be a nonnegative RT phase step")
    init_until = int(_option(config, "init_until_iter", 0))
    norm_from = int(_option(config, "norm_loss_from_iter", 0))
    mask_from = int(_option(config, "mask_loss_from_iter", -1))
    if mask_from == -1:
        mask_from = init_until
    dssim_weight = float(_option(config, "lambda_dssim", 0.2))
    terms = {}

    # Preserve the official stage ordering, including the initial transmission
    # reconstruction before normal regularization starts.
    if iteration < norm_from:
        prediction = package["render_tran"]
        photo_key = "diff"
    elif iteration < init_until:
        prediction = package["final_rendering"]
        photo_key = "diff"
    else:
        photo_key = "pbr"
        if iteration < mask_from:
            prediction = package["final_rendering"]
        else:
            prediction = gate_transmission(
                package["final_tran"], package["final_scat"], package["final_spec"],
                package["surface_opacity"], float(_option(config, "local_var_scale", 4.0)),
            )
    terms[photo_key] = (1.0 - dssim_weight) * l1_loss(prediction, target) + dssim_weight * (1.0 - ssim(prediction, target))

    lpips_weight = float(_option(config, "lambda_lpips", 0.01))
    if photo_key == "pbr" and lpips_weight > 0 and iteration >= int(_option(config, "lpips_loss_from_iter", 15000)):
        if lpips_fn is None:
            raise ValueError("The scheduled LPIPS term requires an explicit lpips_fn")
        terms["lpips"] = lpips_weight * lpips_fn(prediction, target).mean()

    occupancy_weight = float(_option(config, "occupancy_decay_weight", 0.001))
    if occupancy_weight > 0 and iteration >= init_until:
        visible = package["visibility_filter"]
        if visible.dtype != torch.bool or visible.ndim != 1 or visible.shape[0] != occupancy.shape[0]:
            raise ValueError("visibility_filter must be bool [number_of_gaussians]")
        visible_occupancy = occupancy[visible]
        terms["occupancy"] = occupancy_weight * (visible_occupancy.mean() if visible_occupancy.numel() else occupancy.sum() * 0.0)

    normal_weight = float(_option(config, "norm_loss_weight", 0.05))
    if normal_weight > 0 and iteration >= norm_from:
        error = 1 - (package["surface_normal"] * package["surface_depth_normal"]).sum(dim=0, keepdim=True)
        terms["norm"] = normal_weight * (error * package["foreground"]).mean()

    distance_weight = float(_option(config, "dist_loss_weight", 0.0))
    if distance_weight > 0 and iteration >= int(_option(config, "dist_loss_from_iter", 0)):
        # The Ex4DGS rasterizer does not supply the official 2DGS distortion
        # statistic. Fail if this optional loss is requested without that data.
        if package.get("volume_dist") is None:
            raise ValueError("The Ex4DGS rasterizer does not provide the official volume_dist statistic")
        terms["dist"] = distance_weight * package["volume_dist"].mean()

    mask_weight = float(_option(config, "mask_loss_weight", 0.01))
    if mask_weight > 0 and iteration >= mask_from:
        terms["mask"] = mask_weight * masked_positive_loss(package["surface_opacity"], candidate_mask)

    # Intentionally NO BCE(transmissivity[~mask], 0). The candidate complement
    # is unknown, not an opaque target; multiplying that loss by zero is unsafe
    # on an empty complement and would obscure the actual changed objective.
    consistency_weight = float(_option(config, "consistency_loss_weight", 0.000002))
    if consistency_weight > 0 and iteration >= init_until:
        terms["consistency"] = consistency_weight * candidate_consistency(
            package["transmissivity"], package["render_scat"], candidate_mask,
        )

    loss = sum(terms.values())
    metrics = {name: float(value.detach().item()) for name, value in terms.items()}
    metrics["total"] = float(loss.detach().item())
    return loss, metrics
