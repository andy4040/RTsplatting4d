"""Restricted Gaussian RT adapter, inspired by (not a reproduction of) RT-Splatting.

Frozen Ex4DGS geometry + temporal support act as geometric occupancy. Selected
Gaussians gain optical opacity and reflection SHs. Surface passes use occupancy;
transmission uses occupancy * optical opacity. All Gaussians occlude the surface
passes, so selected points behind opaque surfaces cannot simply shine through.
3D ellipsoid shortest axes approximate normals; no surfel conversion/refraction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from gaussian_renderer import render
from utils.general_utils import build_rotation
from utils.sh_utils import eval_sh


class CandidateOptics(nn.Module):
    def __init__(self, candidates, duration=1):
        super().__init__()
        candidates = torch.as_tensor(candidates,dtype=torch.bool,device='cuda')
        self.register_buffer('candidates',candidates)
        self.register_buffer('ids',candidates.nonzero().flatten())
        self.duration = max(1,int(duration)-1)
        n = len(self.ids)
        self.opacity_logits = nn.Parameter(torch.full((n,1),4.6,device='cuda'))
        self.reflectance_logits = nn.Parameter(torch.full((n,1),-4.6,device='cuda'))
        self.spec_sh = nn.Parameter(torch.zeros(n,3,9,device='cuda'))
        self.spec_time = nn.Parameter(torch.zeros(n,3,3,device='cuda'))

    def optical_values(self):
        alpha = torch.ones(len(self.candidates),1,device=self.ids.device)
        reflectance = torch.zeros_like(alpha)
        alpha = alpha.index_copy(0,self.ids,self.opacity_logits.sigmoid())
        reflectance = reflectance.index_copy(0,self.ids,self.reflectance_logits.sigmoid())
        return alpha, reflectance

    def forward(self, camera, pc, pipe, background, near=.01, far=300.):
        if pc.get_xyz_at_t(camera.timestamp).shape[0] != len(self.candidates):
            raise ValueError('Topology changed after candidate selection')
        opacity, reflectance = self.optical_values()
        occupancy = pc.get_opacity_at_t(camera.timestamp).detach()
        kwargs = dict(near=near,far=far)
        trans = render(camera,pc,pipe,background,override_opacity=occupancy*opacity,**kwargs)['render']
        if not len(self.ids):
            return {'render':trans,'transmission':trans,'reflection':torch.zeros_like(trans),
                    'reflection_weight':torch.zeros_like(trans[:1])}
        xyz = pc.get_xyz_at_t(camera.timestamp).detach()[self.ids]
        scales = pc.get_scaling().detach()[self.ids]
        rotations = pc.get_rotation_at_t(camera.timestamp).detach()[self.ids]
        frames = build_rotation(rotations)
        normals = frames.gather(2,scales.argmin(-1)[:,None,None].expand(-1,3,1)).squeeze(-1)
        view = F.normalize(camera.camera_center[None]-xyz,dim=-1)
        direction = F.normalize(2*(normals*view).sum(-1,keepdim=True)*normals-view,dim=-1)
        time = float(camera.timestamp)/self.duration
        basis = torch.tensor([math.sin(math.pi*time),math.cos(math.pi*time)-1,time],device=xyz.device)
        color = torch.sigmoid(eval_sh(2,self.spec_sh,direction)+(self.spec_time*basis[None,None]).sum(-1))
        colors = torch.zeros(len(self.candidates),3,device=xyz.device).index_copy(0,self.ids,color)
        black = torch.zeros_like(background)
        reflection = render(camera,pc,pipe,black,override_color=colors*reflectance,
                            override_opacity=occupancy,**kwargs)['render']
        weight = render(camera,pc,pipe,black,override_color=reflectance.expand(-1,3).contiguous(),
                        override_opacity=occupancy,**kwargs)['render'][:1].clamp(0,1)
        transmission = (1-weight)*trans
        return {'render':transmission+reflection,'transmission':transmission,
                'reflection':reflection,'reflection_weight':weight}

    def regularization(self):
        if not len(self.ids):
            return self.opacity_logits.sum()*0
        return ((1-self.opacity_logits.sigmoid()).mean() + self.reflectance_logits.sigmoid().mean()
                + .1*self.spec_time.square().mean() + .01*self.spec_sh.square().mean())
