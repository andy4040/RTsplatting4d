"""Independent thin planar Gaussian surfaces; an RT-inspired research extension.

This is NOT the official RT-Splatting renderer: it uses Ex4DGS thin 3D
ellipsoids and a shared degree-3 reflection environment instead of 2D surfels
and Sph-Mip shading. Occupancy/opacity and surface/volume passes are separate.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from gaussian_renderer import render
from utils.general_utils import build_rotation
from utils.sh_utils import eval_sh, RGB2SH


def world_rays(camera):
    h, w = camera.image_height, camera.image_width
    y, x = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
    local = torch.stack(((2*(x+.5)/w-1)*math.tan(camera.FoVx/2),
                         (2*(y+.5)/h-1)*math.tan(camera.FoVy/2), torch.ones_like(x)), -1)
    rotation = camera.world_view_transform.inverse()[:3, :3]
    return F.normalize(local.float() @ rotation, dim=-1)


class CombinedGaussians:
    """Append independent surfaces, preserving every background Gaussian's identity."""
    def __init__(self, base, surface, transmission=False, enabled=None):
        self.base, self.surface = base, surface
        self.transmission, self.enabled = transmission, enabled
        self._xyz = base._xyz
        self.kernel_size = base.kernel_size
        self.active_sh_degree = base.active_sh_degree
        self.max_sh_degree = base.max_sh_degree

    def get_xyz_at_t(self, t, **kwargs):
        return torch.cat((self.base.get_xyz_at_t(t), self.surface.positions()), 0)

    def get_scaling(self, **kwargs):
        return torch.cat((self.base.get_scaling(), self.surface.scales()), 0)

    def get_rotation_at_t(self, t, **kwargs):
        return torch.cat((self.base.get_rotation_at_t(t), self.surface.rotations()), 0)

    def get_opacity_at_t(self, t, **kwargs):
        occupancy = self.surface.occupancy(self.enabled)
        if self.transmission:
            occupancy = occupancy * self.surface.opacity_logits.sigmoid()[self.surface.group_ids]
        return torch.cat((self.base.get_opacity_at_t(t), occupancy), 0)

    def get_features(self, **kwargs):
        base = self.base.get_features()
        extra = torch.zeros((len(self.surface.group_ids), *base.shape[1:]), device=base.device)
        # A thin transparent interface absorbs light; no arbitrary diffuse RGB texture.
        extra[:, 0] = RGB2SH(torch.zeros_like(extra[:, 0]))
        return torch.cat((base, extra), 0)


class GlassSurfaces(nn.Module):
    def __init__(self, seeds):
        super().__init__()
        for name in ('centers', 'quaternions', 'local_points', 'point_scales', 'move_limits'):
            self.register_buffer('initial_'+name, torch.as_tensor(seeds[name], dtype=torch.float32, device='cuda'))
        self.register_buffer('group_ids', torch.as_tensor(seeds['group_ids'], dtype=torch.long, device='cuda'))
        groups = len(self.initial_centers)
        self.translation = nn.Parameter(torch.zeros_like(self.initial_centers))
        self.quaternions = nn.Parameter(self.initial_quaternions.clone())
        self.log_scale = nn.Parameter(torch.zeros((groups, 2), device='cuda'))
        self.occupancy_logits = nn.Parameter(torch.full((len(self.group_ids), 1), 1.4, device='cuda'))
        self.opacity_logits = nn.Parameter(torch.full((groups, 1), -4.6, device='cuda'))
        self.reflectance_logits = nn.Parameter(torch.full((groups, 1), -3., device='cuda'))
        self.roughness_logits = nn.Parameter(torch.full((groups, 1), -2., device='cuda'))
        self.environment_sh = nn.Parameter(torch.zeros((3, 16), device='cuda'))
        self.register_buffer('enabled', torch.ones(groups, dtype=torch.bool, device='cuda'))

    @property
    def groups(self):
        return len(self.initial_centers)

    def positions(self):
        ids = self.group_ids
        scale = self.log_scale.clamp(-.7, .7).exp()[ids]
        local = self.initial_local_points * torch.cat((scale, torch.ones_like(scale[:, :1])), -1)
        rotated = torch.bmm(build_rotation(self.quaternions)[ids], local.unsqueeze(-1)).squeeze(-1)
        center = self.initial_centers + self.translation.tanh()*self.initial_move_limits[:, None]
        return rotated + center[ids]

    def rotations(self):
        return F.normalize(self.quaternions, dim=-1)[self.group_ids]

    def scales(self):
        scale = self.log_scale.clamp(-.7, .7).exp()[self.group_ids]
        return self.initial_point_scales * torch.cat((scale, torch.ones_like(scale[:, :1])), -1)

    def occupancy(self, enabled=None):
        enabled = self.enabled if enabled is None else enabled
        return self.occupancy_logits.sigmoid() * enabled[self.group_ids, None]

    def regularization(self):
        q = F.normalize(self.quaternions, dim=-1)
        q0 = F.normalize(self.initial_quaternions, dim=-1)
        return (self.translation.tanh().square().mean() + .1*self.log_scale.square().mean()
                + .1*(1-(q*q0).sum(-1).square()).mean()
                + .02*self.environment_sh[:, 1:].square().mean()
                + .02*self.opacity_logits.sigmoid().mean())

    def forward(self, camera, base, pipe, background, near=.01, far=300., enabled=None):
        enabled = self.enabled if enabled is None else enabled
        if not bool(enabled.any()):
            return render(camera, base, pipe, background, near=near, far=far)
        assert not pipe.compute_cov3D_python and not pipe.convert_SHs_python
        transmission = render(camera, CombinedGaussians(base, self, True, enabled), pipe,
                              background, near=near, far=far)['render']
        proxy = CombinedGaussians(base, self, False, enabled)
        nbase = len(base.get_xyz_at_t(camera.timestamp))
        zeros = torch.zeros((nbase, 3), device='cuda')
        normals = build_rotation(self.quaternions)[:, :, 2][self.group_ids]
        # Face all contributing normals toward this camera before aggregation.
        facing = ((camera.camera_center-self.positions())*normals).sum(-1, keepdim=True)
        normals = normals * torch.where(facing >= 0, 1., -1.)
        normal_buffer = render(camera, proxy, pipe, torch.zeros_like(background),
            override_color=torch.cat((zeros, normals), 0), near=near, far=far)['render']
        ids = self.group_ids
        materials = torch.cat((self.reflectance_logits.sigmoid()[ids],
                               torch.ones((len(ids), 1), device='cuda'),
                               self.roughness_logits.sigmoid()[ids]), -1)
        material_buffer = render(camera, proxy, pipe, torch.zeros_like(background),
            override_color=torch.cat((zeros, materials), 0), near=near, far=far)['render']
        weight = material_buffer[:1].clamp(0, 1)
        coverage = material_buffer[1:2].clamp(0, 1)
        normal = F.normalize(normal_buffer.permute(1, 2, 0), dim=-1)
        view = -world_rays(camera)
        reflection_direction = F.normalize(2*(normal*view).sum(-1, keepdim=True)*normal-view, dim=-1)
        roughness = (material_buffer[2]/coverage[0].clamp_min(1e-5)).clamp(0, 1)
        # Degree-dependent attenuation approximates a rough reflected environment.
        degree = torch.tensor([0]+[1]*3+[2]*5+[3]*7, device='cuda')
        attenuation = torch.exp(-roughness[..., None].square()*degree*(degree+1))
        sh = self.environment_sh[None, None]*attenuation[..., None, :]
        environment = torch.sigmoid(eval_sh(3, sh, reflection_direction)).permute(2, 0, 1)
        reflected = weight*environment
        return {'render':(1-weight)*transmission+reflected, 'transmission':transmission,
                'reflection':reflected, 'reflection_weight':weight, 'surface_coverage':coverage}

    def seeds(self):
        result = {name:getattr(self, 'initial_'+name).detach().cpu() for name in
                  ('centers', 'quaternions', 'local_points', 'point_scales', 'move_limits')}
        result['group_ids'] = self.group_ids.cpu()
        return result
