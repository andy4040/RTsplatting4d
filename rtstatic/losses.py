"""RT losses adapted to single-frame training; empty-mask reductions are guarded."""
import torch


def gate_transmission(transmission, complexity):
    weight = complexity.detach().clamp(0, 1)
    return transmission.detach() * weight + transmission * (1 - weight)


def loss_for_view(pkg, target, mask, model, opt, step, mode, warmup):
    from utils.loss_utils import binary_cross_entropy, l1_loss, lpips, ssim
    from utils.image_utils import local_variance

    if mode == "gs":
        prediction = pkg["final_rendering"]
    elif step < warmup:
        prediction = pkg["render_tran"]
    else:
        transmitted = pkg["final_tran"]
        if mode == "rt":
            variance = local_variance(pkg["final_spec"], weights=1 - pkg["surface_opacity"].detach())
            complexity = 1 - torch.exp(-opt.local_var_scale * variance.detach())
            transmitted = gate_transmission(transmitted, complexity)
        prediction = pkg["final_scat"] + transmitted + pkg["final_spec"]
    losses = {"rgb": (1 - opt.lambda_dssim) * l1_loss(prediction, target)
              + opt.lambda_dssim * (1 - ssim(prediction, target))}
    if step >= warmup:
        normal_error = 1 - (pkg["surface_normal"] * pkg["surface_depth_normal"]).sum(0, keepdim=True)
        losses["normal"] = opt.norm_loss_weight * (normal_error * pkg["foreground"]).mean()
    if opt.lambda_lpips > 0 and step >= opt.lpips_loss_from_iter:
        losses["lpips"] = opt.lambda_lpips * lpips(prediction, target)
    if mode != "gs" and step >= warmup:
        visible = pkg["visibility_filter"]
        if visible.any():
            losses["occupancy"] = opt.occupancy_decay_weight * model.get_occupancy[visible].mean()
        losses["mask"] = opt.mask_loss_weight * binary_cross_entropy(1 - pkg["surface_opacity"], mask.float())
        if (~mask).any():
            losses["opaque_transmission"] = opt.transmissivity_loss_weight * binary_cross_entropy(pkg["transmissivity"][~mask], 0)
        if mask.any():
            values = pkg["transmissivity"][mask]
            scatter = pkg["render_scat"][:, mask.squeeze(0)]
            # Per-pixel mean keeps this adapted regularizer independent of mask area/resolution.
            losses["consistency"] = opt.consistency_loss_weight * (
                (values - values.mean()).square().mean() +
                (scatter - scatter.mean(-1, keepdim=True)).square().mean())
    return sum(losses.values()), losses
