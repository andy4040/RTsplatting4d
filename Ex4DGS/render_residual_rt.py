"""Render saved baseline/RT checkpoints without optimization or candidate updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from arguments import ModelParams,OptimizationParams,PipelineParams
from scene import Scene,getmodel
from gaussian_renderer import render
from rt_pipeline.optics import CandidateOptics
from rt_pipeline.report import write_preview,selected_cameras

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--source',required=True);parser.add_argument('--baseline-only',action='store_true')
    cli=parser.parse_args();config=json.loads((cli.run/'experiment.json').read_text())
    p=argparse.ArgumentParser();lp=ModelParams(p);op=OptimizationParams(p);pp=PipelineParams(p);args=p.parse_args([])
    for k,v in config['ex4dgs'].items():setattr(args,k,v)
    args.source_path=cli.source;args.model_path=str(cli.run/'reload');args.loader='residual_n3dv'
    Path(args.model_path).mkdir(exist_ok=True)
    pc=getmodel()(args.sh_degree,args.start_duration,args.time_interval,args.time_pad,
        interp_type=args.interp_type,rot_interp_type=args.rot_interp_type,time_pad_type=args.time_pad_type,var_pad=args.var_pad,kernel_size=args.kernel_size)
    scene=Scene(args,pc)
    state,_=torch.load(cli.run/'baseline_final.pth')
    pc.restore(state,op.extract(args));background=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda');pipe=pp.extract(args)
    if cli.baseline_only:
        render_function=lambda c:render(c,pc,pipe,background,near=args.near,far=args.far)
        label='reloaded_baseline'
    else:
        state=torch.load(cli.run/'rt_latest.pth')
        adapter=CandidateOptics(state['adapter']['candidates'],duration=state['duration'])
        adapter.load_state_dict(state['adapter']);adapter.eval()
        render_function=lambda c:adapter(c,pc,pipe,background,args.near,args.far)
        label='reloaded_rt'
    result=write_preview(cli.run/'reload/previews',label,selected_cameras(scene.test_cameras[1.0],config['pipeline']['probe_frames']),render_function)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
