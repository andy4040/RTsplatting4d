"""Physical ordering and gradient contracts on a small synthetic CUDA scene."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import numpy as np
import torch

from gaussian_renderer import render
from rt_pipeline.surfaces import GlassSurfaces, CombinedGaussians, world_rays
from utils.graphics_utils import getProjectionMatrix
from utils.sh_utils import RGB2SH


class Background:
    def __init__(self, z=4., opacity=.99):
        self._xyz = torch.tensor([[0., 0., z]], device='cuda')
        self.kernel_size = .1
        self.active_sh_degree = self.max_sh_degree = 3
        self.opacity = torch.tensor([[opacity]], device='cuda')
        self.scale = torch.tensor([[1., 1., .01]], device='cuda')

    def get_xyz_at_t(self, t, **kw): return self._xyz
    def get_scaling(self, **kw): return self.scale
    def get_rotation_at_t(self, t, **kw): return torch.tensor([[1., 0., 0., 0.]], device='cuda')
    def get_opacity_at_t(self, t, **kw): return self.opacity
    def get_features(self, **kw):
        features = torch.zeros((1, 16, 3), device='cuda')
        features[:, 0] = RGB2SH(torch.tensor([[.8, .2, .1]], device='cuda'))
        return features


view = torch.eye(4, device='cuda')
projection = getProjectionMatrix(.01, 100., 1., 1.).T.cuda()
camera = SimpleNamespace(image_height=64, image_width=64, FoVx=1., FoVy=1., timestamp=0,
    world_view_transform=view, full_proj_transform=projection, camera_center=torch.zeros(3, device='cuda'))
pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
bg = torch.zeros(3, device='cuda')
seeds = dict(centers=[[0., 0., 2.]], quaternions=[[1., 0., .1, 0.]],
             local_points=[[-.3, -.3, 0.], [.3, -.3, 0.], [-.3, .3, 0.], [.3, .3, 0.]],
             point_scales=[[.4, .4, .004]]*4, move_limits=[.5], group_ids=[0]*4)
base = Background()
surface = GlassSurfaces(seeds)
with torch.no_grad():
    surface.environment_sh[:, 1:] = .15
none = torch.zeros(1, dtype=torch.bool, device='cuda')
original = render(camera, base, pipe, bg, near=.01, far=100.)['render']
disabled = surface(camera, base, pipe, bg, .01, 100., enabled=none)['render']
torch.testing.assert_close(original, disabled, atol=0, rtol=0)
proxy = CombinedGaussians(base, surface)
assert len(proxy.get_xyz_at_t(0)) == 5
torch.testing.assert_close(proxy.get_xyz_at_t(0)[:1], base._xyz)
result = surface(camera, base, pipe, bg, .01, 100.)
target = torch.rand_like(result['render'])
loss = (result['render']-target).square().mean() + .01*(1-result['surface_coverage']).mean()
loss.backward()
for name in ('translation', 'quaternions', 'occupancy_logits', 'opacity_logits', 'reflectance_logits', 'environment_sh'):
    gradient = getattr(surface, name).grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
with torch.no_grad():
    front = surface(camera, base, pipe, bg, .01, 100.)['surface_coverage'][0, 32, 32].item()
    blocker = Background(z=1., opacity=.99999)
    blocker.scale[:, :2] = 10.
    behind = surface(camera, blocker, pipe, bg, .01, 100.)['surface_coverage'][0, 32, 32].item()
    assert front > .1 and behind < front*.02, (front, behind)
    surface.opacity_logits.fill_(-20.)
    transparent = render(camera, CombinedGaussians(base, surface, True), pipe, bg, near=.01, far=100.)['render']
    torch.testing.assert_close(original, transparent, atol=1e-5, rtol=1e-5)
    clone = GlassSurfaces(surface.seeds()); clone.load_state_dict(surface.state_dict())
    torch.testing.assert_close(surface(camera, base, pipe, bg)['render'], clone(camera, base, pipe, bg)['render'], atol=0, rtol=0)
assert torch.isfinite(world_rays(camera)).all()
print('PASS: independent surface geometry, exact disabled baseline, transparent transmission, foreground occlusion, finite nonzero geometry/material gradients, exact checkpoint reload.')
