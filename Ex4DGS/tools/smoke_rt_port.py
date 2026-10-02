"""Exercise the production RT port on CUDA with a small analytic Ex4DGS scene.

Run from the Ex4DGS environment after installing both upstream CUDA renderers:
    python tools/smoke_rt_port.py
No dataset, saved experiment, or server process is modified. Network dimensions
are the exact production dimensions; only the scene/image size is small.
"""
import argparse
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.nn import functional as F

from gaussian_renderer import render as ex_render
from rt_port.model import RTMaterialModel, initial_surfel_axes, quaternion_product
from rt_port.renderer import render
from rt_port.losses import compute_rt_loss
from rt_port.vendor.sph_utils import cart2sph
from utils.general_utils import build_rotation
from utils.graphics_utils import getProjectionMatrix


class AnalyticExModel(nn.Module):
    """Small differentiable implementation of the Ex4DGS rendering interface."""
    def __init__(self):
        super().__init__()
        device = "cuda"
        self._xyz = nn.Parameter(torch.tensor([
            [-.48, -.25, 2.7], [0., -.23, 3.0], [.47, -.21, 2.8],
            [-.42, .24, 3.0], [.06, .26, 2.8], [.49, .23, 3.1],
        ], device=device))
        self._xyz_disp = nn.Parameter(torch.full_like(self._xyz, .015))
        self._scaling = nn.Parameter(torch.tensor([[.32, .29, .035]], device=device).log().repeat(6, 1))
        self._rotation = nn.Parameter(torch.tensor([[1., .03, -.02, .01]], device=device).repeat(6, 1))
        self._rotation_motion = nn.Parameter(torch.full((6, 4), .003, device=device))
        self._opacity = nn.Parameter(torch.full((6, 1), 1.2, device=device))
        colors = torch.tensor([[.3, .7, .2], [.8, .2, .3], [.2, .3, .8],
                               [.6, .5, .2], [.3, .6, .8], [.8, .4, .7]], device=device)
        self._features = nn.Parameter(((colors - .5) / .28209479177387814)[:, None, :])
        self.active_sh_degree = self.max_sh_degree = 0
        self.kernel_size = .1

    def get_xyz_at_t(self, t, mode=0, training=False):
        return self._xyz + self._xyz_disp * t

    def get_scaling(self, mode=0):
        return self._scaling.exp()

    def get_rotation_at_t(self, t, mode=0):
        return self._rotation + t * self._rotation_motion

    def get_opacity_at_t(self, t, mode=0, training=False):
        return self._opacity.sigmoid()

    def get_features(self, mode=0):
        return self._features


def make_camera(width, height):
    camera = SimpleNamespace(image_width=width, image_height=height,
                             FoVx=.75, FoVy=.60, timestamp=1.)
    camera.world_view_transform = torch.eye(4, device="cuda")
    camera.projection_matrix = getProjectionMatrix(.01, 300., camera.FoVx,
                                                   camera.FoVy).T.cuda()
    camera.full_proj_transform = camera.projection_matrix
    camera.camera_center = torch.zeros(3, device="cuda")
    return camera


def test_frame_mapping():
    scales = torch.tensor([[.1, .3, .2], [.3, .1, .2], [.3, .2, .1]], device="cuda")
    normal_axis, tangent_axes, permutation = initial_surfel_axes(scales)
    original = F.normalize(torch.tensor([[.9, .2, .1, .15]], device="cuda").repeat(3, 1), dim=-1)
    mapped = build_rotation(quaternion_product(original, permutation))
    original_frames = build_rotation(original)
    expected_normals = original_frames.gather(2, normal_axis[:, None, None].expand(-1, 3, 1))[:, :, 0]
    expected_tangents = original_frames.gather(2, tangent_axes[:, None, :].expand(-1, 3, -1))
    torch.testing.assert_close(mapped[:, :, 2], expected_normals, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(mapped[:, :, :2], expected_tangents, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.linalg.det(mapped), torch.ones(3, device="cuda"), atol=1e-6, rtol=1e-6)


def official_shading_reference(material, directions, roughness, features):
    """Direct upstream equations, with no chunking/checkpointing."""
    feature_map = F.normalize(features, dim=-1).reshape(-1, 1, 4)
    wo_xy = ((cart2sph(directions[:, material.XYZ])[:, 1:]
              / directions.new_tensor([[math.pi, 2 * math.pi]]))[:, [1, 0]])
    spec_feat = material.dir_encoding(wo_xy[None, :, None, :], roughness,
                                      index=0).reshape(-1, 16)
    wrap = (spec_feat.reshape(-1, 16, 1) @ feature_map).reshape(-1, 64)
    output = material.light_mlp(torch.cat([wrap, spec_feat], dim=-1))
    return torch.exp(output[:, :3] + math.log(.5)), torch.sigmoid(output[:, 3:4])


def assert_nonzero_gradient(parameter, label):
    if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum() == 0:
        raise AssertionError(f"{label}: expected finite nonzero gradient")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA and the compiled upstream rasterizers are required")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    test_frame_mapping()
    base = AnalyticExModel()
    material_config = {"rand_init": True, "shading_chunk_size": 65536,
                       "checkpoint_shading": True}
    material = RTMaterialModel(base, material_config)
    assert material.dir_encoding.fm.shape == (1, 1, 512, 1024, 16)
    assert material.light_mlp[0].in_features == 80
    assert material.light_mlp[0].out_features == 256
    assert not any(key.startswith("base.") for key in material.state_dict())
    assert {id(p) for p in base.parameters()}.isdisjoint({id(p) for p in material.parameters()})

    directions = F.normalize(torch.randn(317, 3, device="cuda"), dim=-1)
    roughness = torch.rand(317, 1, device="cuda") * .7 + .1
    features = torch.randn(317, 4, device="cuda")
    material.shading_chunk_size = 113  # Exercise chunk boundaries in this parity probe.
    with torch.no_grad():
        expected = official_shading_reference(material, directions, roughness, features)
        actual = material.shade(directions, roughness, features)
    material.shading_chunk_size = material_config["shading_chunk_size"]
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)

    camera = make_camera(args.width, args.height)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False,
                           convert_SHs_python=False, depth_ratio=0.)
    background = torch.zeros(3, device="cuda")
    with torch.no_grad():
        saved_alpha = material._opacity.clone()
        material._opacity.fill_(20.)  # sigmoid rounds to float32 1, only in this parity test.
        baseline = ex_render(camera, base, pipe, background, near=.01, far=300.)["render"]
        opaque = render(camera, base, material, pipe, background)["render_tran"]
        torch.testing.assert_close(opaque, baseline, atol=1e-6, rtol=1e-6)
        material._opacity.copy_(saved_alpha)

    # Both original geometry/color and new RT materials must receive gradients.
    original_count = len(base._xyz)
    package = render(camera, base, material, pipe, background, training=True)
    assert len(base._xyz) == original_count == len(material._opacity)
    assert package["volume_dist"] is None
    for name, value in package.items():
        if torch.is_tensor(value) and value.is_floating_point():
            assert torch.isfinite(value).all(), name
    torch.testing.assert_close(package["render"], package["final_tran"] + package["final_scat"] + package["final_spec"])
    ys, xs = torch.meshgrid(torch.linspace(0, 1, args.height, device="cuda"),
                           torch.linspace(0, 1, args.width, device="cuda"), indexing="ij")
    target = torch.stack((xs * .7, ys * .8, .2 + xs * ys * .5))
    mask = ((xs > .2) & (xs < .8) & (ys > .15) & (ys < .85))
    loss, metrics = compute_rt_loss(package, target, mask, package["occupancy"],
                                    {"lambda_lpips": 0., "init_until_iter": 0}, 1)
    loss.backward()
    for parameter, label in ((base._xyz, "geometry"), (base._xyz_disp, "motion"),
                             (base._scaling, "scale"), (base._rotation, "rotation"),
                             (base._features, "base color"), (base._opacity, "occupancy"),
                             (material._opacity, "optical opacity"),
                             (material._transmissivity, "transmissivity"),
                             (material._reflectance, "reflectance"),
                             (material._language_feature, "material feature"),
                             (material.dir_encoding.fm, "SphMip"),
                             (material.light_mlp[0].weight, "lighting MLP")):
        assert_nonzero_gradient(parameter, label)
    for parameter in list(base.parameters()) + list(material.parameters()):
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()
    base_optimizer = torch.optim.Adam(base.parameters(), lr=1e-4)
    material_optimizer = torch.optim.Adam(material.optimizer_groups(), eps=1e-15)
    base_optimizer.step()
    material_optimizer.step()
    base_optimizer.zero_grad(set_to_none=True)
    material_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        before_save = render(camera, base, material, pipe, background)["render"]
    buffer = io.BytesIO()
    torch.save({"base": base.state_dict(), "material": material.state_dict(),
                "config": material_config, "optimizer": material_optimizer.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer)
    reloaded_base = AnalyticExModel()
    reloaded_base.load_state_dict(payload["base"])
    reloaded_material = RTMaterialModel(reloaded_base, payload["config"])
    reloaded_material.load_state_dict(payload["material"])
    with torch.no_grad():
        reloaded = render(camera, reloaded_base, reloaded_material, pipe, background)["render"]
    torch.testing.assert_close(reloaded, before_save, atol=1e-6, rtol=1e-6)
    print(json.dumps({"status": "passed", "production_light_architecture": True,
                      "transmission_alpha_one_matches_ex": True,
                      "shading_matches_official_equations": True,
                      "geometry_motion_material_gradients": True,
                      "checkpoint_roundtrip": True, "losses": metrics,
                      "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                      "image_size": [args.width, args.height]}, indent=2))


if __name__ == "__main__":
    main()
