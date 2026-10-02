"""Exercise installed CUDA operators on synthetic inputs; no model training."""
import json
import sys
from pathlib import Path
sys.path.insert(0, '/workspace/RTsplatting4d')
import torch
import nvdiffrast.torch as dr
from simple_knn._C import distCUDA2
from diff_surfel_anych import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getProjectionMatrix
from scene.gaussian_model import SphMipEncoding

torch.manual_seed(0)
result = {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(), 'training_started': False}
distance = distCUDA2(torch.rand(64, 3, device='cuda'))
assert torch.isfinite(distance).all() and (distance > 0).all()
result['simple_knn_cuda'] = 'passed'
encoder = SphMipEncoding(n_levels=2, plane_size=8, feature_dim=4, rand_init=True).cuda()
coords = torch.rand(1, 2, 1, 2, device='cuda')
levels = torch.zeros(1, 2, 1, device='cuda')
encoded = encoder(coords, levels)
encoded.sum().backward()
assert torch.isfinite(encoded).all() and torch.isfinite(encoder.fm.grad).all()
result['official_SphMipEncoding_forward_backward'] = 'passed'
projection = getProjectionMatrix(znear=.01, zfar=100., fovX=1., fovY=1.).T.cuda()
settings = GaussianRasterizationSettings(32,32,0.54630249,0.54630249,torch.zeros(3,device='cuda'),1.,torch.eye(4,device='cuda'),projection,0,torch.zeros(3,device='cuda'),False,False)
for channels in [1, 9]:
    means = torch.tensor([[0.,0.,2.]], device='cuda', requires_grad=True)
    extras = torch.full((1,channels),.5,device='cuda',requires_grad=True)
    outputs = GaussianRasterizer(settings)(means3D=means, means2D=torch.zeros_like(means,requires_grad=True),
        opacities=torch.full((1,1),.5,device='cuda',requires_grad=True),
        shs=torch.ones(1,1,3,device='cuda',requires_grad=True),extras=extras,
        scales=torch.full((1,2),.1,device='cuda',requires_grad=True),
        rotations=torch.tensor([[1.,0.,0.,0.]],device='cuda',requires_grad=True))
    color,extra,radii,depth = outputs
    assert radii.max() > 0 and all(torch.isfinite(x).all() for x in outputs)
    (color.mean()+extra.mean()+depth.mean()).backward()
    assert torch.isfinite(means.grad).all() and torch.isfinite(extras.grad).all()
    result[f'diff_surfel_{channels}_channels_forward_backward'] = 'passed'
torch.cuda.synchronize()
Path('/workspace/runtime-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
