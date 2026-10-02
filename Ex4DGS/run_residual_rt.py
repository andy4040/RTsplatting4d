"""Run Ex4DGS -> persistent train residuals -> independent surface trials."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from train import training
from rt_pipeline.pipeline import ResidualMonitor, train_optics

RUN_OUTPUT = None


def main():
    global RUN_OUTPUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    cli=parser.parse_args()
    config=json.loads(Path(cli.config).read_text())
    cfg=config['pipeline']
    observations=cfg['observe_iterations']
    assert observations==sorted(set(observations))
    assert len(observations)>=cfg['min_observations']
    assert observations[-1]==cfg['baseline_iterations']
    output=Path(cli.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing experiment')
    output.mkdir(parents=True,exist_ok=True)
    RUN_OUTPUT = output
    defaults=argparse.ArgumentParser()
    lp=ModelParams(defaults);op=OptimizationParams(defaults);pp=PipelineParams(defaults)
    args=defaults.parse_args([])
    for key,value in config['ex4dgs'].items():
        if not hasattr(args,key): raise ValueError(f'Unknown Ex4DGS option: {key}')
        setattr(args,key,value)
    args.source_path=cli.source;args.model_path=str(output);args.iterations=cfg['baseline_iterations']
    args.loader='residual_n3dv'
    args.test_iterations=[];args.save_iterations=[args.iterations];args.checkpoint_iterations=[]
    args.extract_every=getattr(args,'extract_every',1)
    (output/'experiment.json').write_text(json.dumps(config,indent=2))
    (output/'previews').mkdir(exist_ok=True)
    (output/'previews/index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Ex4DGS 학습 진행 중</h1><p>1,000회마다 테스트 시점의 중간 결과가 추가됩니다. 첫 결과가 아직 준비되지 않았습니다.</p>',encoding='utf-8')
    safe_state(False)
    monitor=ResidualMonitor(cfg,output,args.near,args.far)
    scene=training(lp.extract(args),op.extract(args),pp.extract(args),[],args.save_iterations,[],None,-1,args,observer=monitor)
    # Authoritative checkpoint is after all end-of-iteration callbacks/mutations.
    torch.save((scene.gaussians.capture(),args.iterations),output/'baseline_final.pth')
    if config.get('post_baseline', 'rt_port') == 'rt_port':
        import gc
        from run_rt_ex4dgs import run as run_rt_trial
        del scene
        gc.collect(); torch.cuda.empty_cache()
        run_rt_trial(SimpleNamespace(baseline_run=output, source=cli.source,
            config=Path(config.get('rt_port_config', 'configs/coffee_rt_port.json')),
            output=output/'rt_port_trial', resume=False))
        return
    if config.get('post_baseline') == 'surface_hypotheses':
        import gc
        from run_surface_hypotheses import main as run_surface_trial
        del scene
        gc.collect(); torch.cuda.empty_cache()
        run_surface_trial(SimpleNamespace(baseline_run=output, source=cli.source,
            config=Path(config.get('surface_config', 'configs/coffee_surface_hypotheses.json')),
            output=output/'surface_trial'))
        return
    if config.get('post_baseline') != 'legacy_adapter_smoke_only':
        raise ValueError('Unknown post-baseline stage')
    background=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
    candidates=monitor.select(scene,pp.extract(args),background)
    train_optics(scene,pp.extract(args),background,candidates,cfg,output,args.near,args.far)
    (output/'complete.json').write_text(json.dumps({'status':'complete','baseline_iterations':args.iterations,'rt_iterations_requested':cfg['rt_iterations'],'candidate_points':int(candidates.sum())},indent=2))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if RUN_OUTPUT is not None:
            path=RUN_OUTPUT/'progress.json'
            try: state=json.loads(path.read_text())
            except (OSError,ValueError): state={}
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            path.write_text(json.dumps(state,indent=2))
        raise
