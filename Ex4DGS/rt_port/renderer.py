"""A shared-geometry Ex4DGS / official RT-Splatting rendering adapter.

Deferred rendering/shading equations adapted from RT-Splatting's
gaussian_renderer/__init__.py at commit 3f45b3cac4be04db9f3092234666b695991b268a.
See vendor provenance and the original licenses included with this port.

Transmission uses the original Ex4DGS 3D rasterizer (sigma * optical alpha).
Deferred surface attributes use the official RT-Splatting 2D rasterizer (sigma)
on a differentiable surfel view of the SAME centers/rotations/scales. There is
no new surface collection or fixed-depth plane. This hybrid deliberately keeps
Ex4DGS 3D volume support and is not the paper's all-2D rendering equivalence.

Ex4DGS's rasterizer does not supply RT volume distortion, normal, or weighted
optical-opacity maps. Unsupported keys are None, never fabricated zero losses.
"""
import math

import torch
from torch.nn import functional as F


def render(viewpoint_camera, base, material, pipe, bg_color, timestamp=None,
           training=False, near=.01, far=300., scaling_modifier=1., init_stage=False):
    from diff_surfel_anych import GaussianRasterizationSettings, GaussianRasterizer
    from gaussian_renderer import render as ex_render
    from rt_port.vendor.color_utils import reflect
    from rt_port.vendor.point_utils import camera_rays, depth_to_normal_sobel

    if getattr(pipe, "compute_cov3D_python", False) or getattr(pipe, "convert_SHs_python", False):
        raise ValueError("RT port requires original CUDA SH/covariance paths (Python flags false)")
    timestamp = viewpoint_camera.timestamp if timestamp is None else timestamp
    xyz = base.get_xyz_at_t(timestamp, training=training)
    occupancy = material.occupancy_at(timestamp, training=training)
    opacity = material.get_opacity
    volume = ex_render(viewpoint_camera, base, pipe, bg_color, timestamp=timestamp,
                       scaling_modifier=scaling_modifier, training=training, near=near,
                       far=far, override_opacity=occupancy * opacity)

    means2D = torch.zeros_like(xyz, requires_grad=True) + 0
    if torch.is_grad_enabled():
        means2D.retain_grad()
    settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * .5),
        tanfovy=math.tan(viewpoint_camera.FoVy * .5),
        bg=bg_color, scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=base.active_sh_degree, campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    scales, rotations = material.surfel_geometry(timestamp)
    inside = material.get_inside_mask.to(occupancy.dtype)
    extras = torch.cat((material.get_roughness, material.get_language_feature,
                        inside, inside * material.get_reflectance, opacity,
                        inside * material.get_transmissivity), dim=-1)
    render_scat, surface_extras, radii, allmap = rasterizer(
        means3D=xyz, means2D=means2D, shs=base.get_features(), extras=extras,
        opacities=occupancy, scales=scales, rotations=rotations, cov3D_precomp=None,
    )
    roughness, feature, foreground, reflectance, surface_opacity, transmissivity = (
        surface_extras.split([1, 4, 1, 1, 1, 1], dim=0))
    foreground = foreground.detach()
    surface_alpha = allmap[1:2]
    surface_normal = allmap[2:5]
    surface_normal = (surface_normal.movedim(0, -1)
                      @ viewpoint_camera.world_view_transform[:3, :3].T).movedim(-1, 0)
    surface_normal = F.normalize(surface_normal, dim=0)
    median_depth = torch.nan_to_num(allmap[5:6], 0, 0)
    # This follows the official expected-depth normalization and cleanup.
    expected_depth = torch.nan_to_num(allmap[0:1] / surface_alpha, 0, 0)
    depth_ratio = float(getattr(pipe, "depth_ratio", .0))
    surface_depth = expected_depth * (1 - depth_ratio) + depth_ratio * median_depth
    surface_depth_normal = depth_to_normal_sobel(
        viewpoint_camera, surface_depth.movedim(0, -1)).movedim(-1, 0)
    surface_depth_normal = surface_depth_normal * surface_alpha.detach()

    h, w = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
    selected = (foreground.flatten() > .05).nonzero(as_tuple=True)[0]
    render_spec = xyz.new_zeros((3, h, w))
    render_attenuation = xyz.new_zeros((1, h, w))
    _, viewdirs = camera_rays(viewpoint_camera)
    viewdirs = F.normalize(viewdirs, dim=-1)
    reflected = F.normalize(reflect(-viewdirs, surface_normal.movedim(0, -1)), dim=-1)
    if len(selected):
        spec_light, attenuation = material.shade(
            reflected.reshape(-1, 3)[selected],
            roughness.movedim(0, -1).reshape(-1, 1)[selected],
            feature.movedim(0, -1).reshape(-1, material.gsfeat_dim)[selected],
        )
        render_spec = render_spec.reshape(3, -1).index_copy(
            1, selected, spec_light.T).reshape(3, h, w)
        render_attenuation = render_attenuation.reshape(1, -1).index_copy(
            1, selected, attenuation.T).reshape(1, h, w)
    render_attenuation = 1 - (1 - render_attenuation) * foreground

    # Exact official radiometric composition. Gradient gating belongs to loss.
    render_tran = volume["render"]
    final_tran = render_tran * transmissivity
    final_scat = render_scat * (1 - transmissivity)
    final_spec = render_spec * reflectance
    final_rendering = final_tran + final_scat
    if not init_stage:
        final_tran = final_tran * render_attenuation
        final_scat = final_scat * render_attenuation
        final_rendering = final_tran + final_scat + final_spec

    return {
        "final_rendering": final_rendering, "render": final_rendering,
        "final_tran": final_tran, "final_scat": final_scat, "final_spec": final_spec,
        "transmission": final_tran, "reflection": final_spec,
        "render_spec": render_spec, "render_tran": render_tran, "render_scat": render_scat,
        "feature": feature, "roughness": roughness, "reflectance": reflectance,
        "transmissivity": transmissivity, "attenuation": render_attenuation,
        "foreground": foreground, "surface_alpha": surface_alpha,
        "surface_depth": surface_depth, "surface_normal": surface_normal,
        "surface_depth_normal": surface_depth_normal, "surface_dist": allmap[6:7],
        "surface_opacity": surface_opacity,
        "volume_alpha": volume["acc"], "volume_depth": volume["depth"],
        "volume_normal": None, "volume_depth_normal": None, "volume_dist": None,
        "volume_opacity": None,
        "occupancy": occupancy, "optical_opacity": opacity,
        "viewspace_points": means2D, "visibility_filter": radii > 0, "radii": radii,
        "volume_viewspace_points": volume["viewspace_points"],
        "volume_visibility_filter": volume["visibility_filter"],
        "volume_radii": volume["radii"], "dominent_idxs": volume["dominent_idxs"],
    }
