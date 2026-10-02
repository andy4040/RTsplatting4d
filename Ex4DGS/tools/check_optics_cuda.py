"""CUDA contract tests using real cameras and initialization, without training."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from arguments import ModelParams,OptimizationParams,PipelineParams
from scene import Scene,getmodel
from gaussian_renderer import render
from rt_pipeline.optics import CandidateOptics

p=argparse.ArgumentParser();lp=ModelParams(p);op=OptimizationParams(p);pp=PipelineParams(p)
args=p.parse_args();args.loader='residual_n3dv';args.end_timestamp=0;args.resolution=8
Path(args.model_path).mkdir(parents=True,exist_ok=True)
pc=getmodel()(args.sh_degree,args.start_duration,args.time_interval,args.time_pad)
scene=Scene(args,pc);pc.training_setup(op.extract(args))
cam=scene.train_cameras[1.0][0];pipe=pp.extract(args);bg=torch.zeros(3,device='cuda')
for p in vars(pc).values():
    if isinstance(p,torch.nn.Parameter):p.requires_grad_(False)
n=len(pc.get_xyz_at_t(0));none=CandidateOptics(np.zeros(n,bool))
original=render(cam,pc,pipe,bg,near=.01,far=300.)['render'];adapted=none(cam,pc,pipe,bg,near=.01,far=300.)['render']
torch.testing.assert_close(original,adapted,atol=1e-6,rtol=1e-6)
some=CandidateOptics(np.arange(n)%2==0)
result=some(cam,pc,pipe,bg)
assert all(torch.isfinite(v).all() for v in result.values())
result['render'].mean().backward()
assert some.opacity_logits.grad.abs().sum()>0
assert some.reflectance_logits.grad.abs().sum()>0
assert some.spec_sh.grad.abs().sum()>0
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in some.parameters())
alpha,r=some.optical_values()
assert torch.all(alpha[~some.candidates]==1) and torch.all(r[~some.candidates]==0)
print('PASS: empty gate equals baseline; non-candidates unchanged; opacity/reflection/SH finite nonzero gradients')
with torch.no_grad():
    pc.xyz_error_min_timestamp.fill_(0)
    pc.xyz_error_min_timestamp[0]=-1
    pc.prune_invisible()
    assert len(pc.get_xyz_at_t(0))==n-1
    pc.prune_small()
    assert len(pc.get_xyz_at_t(0))==n-1
print('PASS: invisible/small-point pruning with no dynamic Gaussians')
