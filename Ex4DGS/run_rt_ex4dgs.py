"""Matched Ex4DGS continuation and shared-geometry RT port, validation selection.

This entry point never imports the historical independent-plane experiment.
The initial pilot explicitly operates at one timestamp; temporal quality has
not been established. Use --resume to continue an interrupted paired trial.
"""
import argparse
import html
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from gaussian_renderer import render as ex_render
from rt_pipeline.report import load_image, save_image
from rt_port.evaluation import freeze_validation_roi, measure_view, measure_test_view, roi_sha256
from rt_port.losses import compute_rt_loss
from rt_port.model import RTMaterialModel
from rt_port.renderer import render as rt_render
from rt_port.runtime import load_baseline, load_training_masks, atomic_json, atomic_save, sha256_file
from rt_port.selection import decide
from utils.loss_utils import l1_loss, ssim


def finite_gradients(optimizers):
    """Check gradients before either optimizer updates a model with bad values."""
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Nonfinite gradient in {group.get('name', 'unnamed')}")


def assert_same_state(left, right, require_independent=True):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        if require_independent and left.numel(): assert left.data_ptr() != right.data_ptr()
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: assert_same_state(left[key], right[key], require_independent)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_same_state(a, b, require_independent)
    else:
        assert left == right


def save_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def draw_training_sample(schedule, rng, count, random_background, background):
    if not schedule:
        schedule.extend(range(count)); rng.shuffle(schedule)
    return schedule.pop(), torch.rand_like(background) if random_background else background


@torch.no_grad()
def evaluate(step, cameras, targets, rois, roi_meta, control, treatment, material, pipe,
             background, args, folder, split='validation'):
    folder.mkdir(parents=True, exist_ok=True)
    views, rows = [], []
    for camera in cameras:
        name = camera.image_name
        target = targets[name] if name in targets else load_image(camera)
        prediction = ex_render(camera, control, pipe, background, near=args.near, far=args.far)['render']
        package = rt_render(camera, treatment, material, pipe, background, near=args.near, far=args.far)
        labels = rois[name] if name in rois else np.zeros(target.shape[1:], dtype=np.int32)
        measure = measure_view if split == 'validation' else measure_test_view
        metrics = measure(prediction, package['render'], target, labels, image=name,
                          expected_roi_sha256=roi_meta.get(name, {}).get('roi_sha256'))
        views.append(metrics)
        for key, value in [('gt', target), ('ex4dgs', prediction), ('rt', package['render']),
                           ('transmission', package['final_tran']), ('reflection', package['final_spec'])]:
            save_image(value, folder/f'{name}_{key}.jpg')
        # JPEGs are illustrative only; PSNR is computed from unclipped target and
        # clamped float predictions, independently of image encoding.
        control_psnr = -10*np.log10(max(metrics['full_control_mse'], 1e-12))
        rt_psnr = -10*np.log10(max(metrics['full_rt_mse'], 1e-12))
        row = f'<h2>{html.escape(name)}: Ex4DGS {control_psnr:.3f} dB / RT {rt_psnr:.3f} dB</h2><div class="row">'
        for key in ['gt', 'ex4dgs', 'rt', 'transmission', 'reflection']:
            row += f'<figure><figcaption>{key}</figcaption><a href="{name}_{key}.jpg"><img src="{name}_{key}.jpg"></a></figure>'
        rows.append(row+'</div>')
        del package, prediction
    result = {'iteration': step, 'split': split, 'views': views}
    atomic_json(folder/'metrics.json', result)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px}.row{display:flex;flex-wrap:wrap}figure{width:45%;margin:1%}img{width:100%}pre{white-space:pre-wrap}</style>'
        f'<h1>Ex4DGS / RT 이식 비교: {step:,}회 추가 학습</h1><p>같은 체크포인트에서 같은 횟수 추가 학습. 후보 밖은 미확인 영역입니다. split: {split}</p>'
        +''.join(rows)+'<details><summary>영역별 수치</summary><pre>'+html.escape(json.dumps(result, indent=2))+'</pre></details>', encoding='utf-8')
    return result


def publish_index(preview_root, history, decision=None):
    links = ''.join(f'<p><a href="step_{h["iteration"]:06d}/index.html">추가 {h["iteration"]:,}회 비교</a></p>' for h in history)
    if decision is not None:
        links += '<p><a href="final_test/index.html">선택 완료 후 test 결과</a></p><pre>'+html.escape(json.dumps(decision, indent=2))+'</pre>'
    (preview_root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Ex4DGS + RT-Splatting</h1><p>기존 가우시안을 공유하는 반사·투과 이식. 검증으로 전체 모델을 선택하며, 부분 이미지를 합성하지 않습니다.</p>'
        '<p><a href="candidates/index.html">학습 후보 및 고정 검증 영역</a></p>'+links, encoding='utf-8')


def validate_config(cfg):
    if cfg['iterations_per_branch'] <= 0 or cfg['evaluate_every'] <= 0:
        raise ValueError('Training/evaluation intervals must be positive')
    if set(cfg['validation_cameras']) & set(cfg['test_cameras']):
        raise ValueError('Validation and test overlap')
    if len(set(cfg['validation_cameras'])) != len(cfg['validation_cameras']) or len(set(cfg['test_cameras'])) != len(cfg['test_cameras']):
        raise ValueError('Duplicate split camera')
    if cfg['rt_loss'].get('dist_loss_weight', 0) != 0:
        raise ValueError('Ex volume rasterizer lacks the official distortion statistic')
    decide([], cfg['decision'])  # Validate thresholds before GPU work.


def run(cli):
    output, baseline_run = Path(cli.output), Path(cli.baseline_run)
    cfg = json.loads(Path(cli.config).read_text(encoding='utf-8'))
    validate_config(cfg)
    resume = bool(getattr(cli, 'resume', False))
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError('Refusing to overwrite a trial. Use a new output or --resume.')
    if resume and not (output/'latest.pth').exists():
        raise FileNotFoundError('Resume requires latest.pth')
    if resume and (output/'complete.json').exists():
        raise ValueError('Trial already complete')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    scene, args, opt, pipe, baseline_cfg, baseline_iteration, create_model = load_baseline(
        baseline_run, cli.source, output)
    if opt.static_reg != 0 or opt.motion_reg != 0:
        raise ValueError('This first-frame continuation requires baseline static/motion regularizers zero')
    if opt.lambda_dssim != cfg['rt_loss'].get('lambda_dssim', .2):
        raise ValueError('Both branches must use the same photometric DSSIM coefficient')
    pipe.depth_ratio = 0.0
    control = scene.gaussians
    treatment = create_model()
    state, _ = torch.load(baseline_run/'baseline_final.pth')
    treatment.restore(state, opt); del state
    # Check exact initial parameter AND optimizer equality without GPU aliasing.
    for left, right in zip(control.optimizer.param_groups, treatment.optimizer.param_groups):
        assert left['name'] == right['name']
        for a, b in zip(left['params'], right['params']):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            if a.numel(): assert a.data_ptr() != b.data_ptr()
    assert_same_state(control.optimizer.state_dict(), treatment.optimizer.state_dict())
    material = RTMaterialModel(treatment, cfg['material'])
    material_optimizer = torch.optim.Adam(material.optimizer_groups(), lr=0.0, eps=1e-15)
    train = sorted(scene.train_cameras[1.0], key=lambda c:c.image_name)
    heldout = scene.test_cameras[1.0]
    validation = [c for c in heldout if c.image_name.split('_')[0] in cfg['validation_cameras']]
    test = [c for c in heldout if c.image_name.split('_')[0] in cfg['test_cameras']]
    assert len(validation) == len(cfg['validation_cameras']) and len(test) == len(cfg['test_cameras'])
    assert len(validation)+len(test) == len(heldout)
    assert set(c.image_name for c in train).isdisjoint(c.image_name for c in validation+test)
    background = torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.], device='cuda')
    targets = {c.image_name:load_image(c) for c in train+validation}  # No test target used in training/selection.
    preview_root = baseline_run/'previews'/output.name
    preview_root.mkdir(parents=True, exist_ok=True)
    candidate_preview = preview_root/'candidates'; candidate_preview.mkdir(exist_ok=True)
    baseline_hash = sha256_file(baseline_run/'baseline_final.pth')
    protocol = {'baseline_sha256': baseline_hash, 'baseline_iteration': baseline_iteration,
        'train':[c.image_name for c in train], 'validation':[c.image_name for c in validation], 'test':[c.image_name for c in test],
        'test_used_for_selection':False, 'prior_test_exposure':'cam19 was displayed in earlier baseline previews; not a pristine unseen test',
        'method':'shared Ex4DGS 3D volume and official RT 2D deferred surface rasterizer',
        'original_rt_commit':'3f45b3cac4be04db9f3092234666b695991b268a',
        'mask_semantics':'positive-only train candidates; complement unknown',
        'candidate_bce_normalization':'mean over selected positive pixels', 'consistency_reduction':'official spatial SUM',
        'validation_roi':'baseline validation residual patches, frozen before continuation, evaluation-only, no cross-view glass identity assumed',
        'branch_topology':'fixed; no point creation/pruning after completed baseline',
        'same_checkpoint':True, 'same_optimizer_state':True, 'same_camera_order':True,
        'same_iteration_budget':True, 'same_background_sequence':True, 'same_wall_clock_budget':False,
        'rt_loss_iteration_clock':'additional RT phase; LPIPS starts at RT phase step15000',
        'whole_scene_rt_scope':True, 'separate_glass_planes':False,
        'temporal_scope':'current runner validates only frame0; no dynamic-sequence claim',
        'surface_near_clip':0.2, 'volume_near_clip':args.near,
        'known_changes':['3D forward transmission / 2D deferred surface on same Ex geometry',
                         'original Ex optimizer for geometry/color/occupancy; official Adam rates for new RT parameters',
                         'restored 30k point set and fixed topology instead of official RT from-scratch densification',
                         'positive-only candidate mask supervision with empty guards',
                         'shortest-axis surfel frame chosen at initialization; no new depths or planes',
                         'memory checkpointing/chunking of unchanged light MLP']}
    # Count actual near-clip mismatches for this dataset rather than hiding them.
    with torch.no_grad():
        xyz = control.get_xyz_at_t(0)
        protocol['near_clip_counts'] = {}
        for c in train+validation:
            z = (xyz @ c.world_view_transform[:3,:3] + c.world_view_transform[3,:3])[:,2]
            protocol['near_clip_counts'][c.image_name] = int(((z>args.near)&(z<=.2)).sum())
    schedule, history, start_step, resumed_rng = [], [], 0, None
    rng = random.Random(cfg['seed'])
    if resume:
        saved = torch.load(output/'latest.pth')
        if saved['config'] != cfg or saved['protocol']['baseline_sha256'] != baseline_hash:
            raise ValueError('Resume configuration or original checkpoint changed')
        control.restore(saved['control'], opt); treatment.restore(saved['treatment'], opt)
        material.load_state_dict(saved['material']); material_optimizer.load_state_dict(saved['material_optimizer'])
        assert_same_state(material.state_dict(), saved['material'], require_independent=False)
        assert_same_state(material_optimizer.state_dict(), saved['material_optimizer'], require_independent=False)
        assert_same_state(control.optimizer.state_dict(), saved['control'][16], require_independent=False)
        assert_same_state(treatment.optimizer.state_dict(), saved['treatment'][16], require_independent=False)
        history, schedule, start_step = saved['history'], saved['schedule'], saved['iteration']
        rng.setstate(saved['sampler_rng']); resumed_rng = saved['rng']
        del saved
        masks = {}
        for c in train:
            with np.load(output/'training_masks'/f'{c.image_name}.npz') as data: masks[c.image_name] = data['positive'].copy()
        roi_meta = json.loads((output/'roi_metadata.json').read_text())
        rois = {}
        for c in validation:
            with np.load(output/'validation_rois'/f'{c.image_name}.npz') as data: rois[c.image_name] = data['labels'].copy()
    else:
        masks, _ = load_training_masks(train, baseline_run, baseline_cfg['pipeline'], output/'training_masks')
        rois, roi_meta = {}, {}
        (output/'validation_rois').mkdir(exist_ok=True)
        with torch.no_grad():
            for c in validation:
                prediction = ex_render(c, control, pipe, background, near=args.near, far=args.far)['render']
                labels, meta = freeze_validation_roi(prediction, targets[c.image_name], image=c.image_name, **cfg['validation_roi'])
                rois[c.image_name], roi_meta[c.image_name] = labels, meta
                np.savez_compressed(output/'validation_rois'/f'{c.image_name}.npz', labels=labels)
        atomic_json(output/'roi_metadata.json', roi_meta)
    # Hash the actual frozen training masks and evaluation labels into protocol.
    protocol['training_mask_sha256'] = {c.image_name:sha256_file(output/'training_masks'/f'{c.image_name}.npz') for c in train}
    protocol['validation_roi_sha256'] = {name:meta['roi_sha256'] for name,meta in roi_meta.items()}
    for name, labels in rois.items():
        if roi_sha256(labels) != roi_meta[name]['roi_sha256']:
            raise ValueError(f'Frozen validation ROI modified: {name}')
    if resume:
        original_protocol = json.loads((output/'protocol.json').read_text())
        if original_protocol != protocol: raise ValueError('Frozen masks or protocol changed on resume')
    atomic_json(output/'protocol.json', protocol); atomic_json(output/'experiment.json', cfg)
    pseudo = {name:torch.as_tensor(mask, device='cuda') for name,mask in masks.items()}
    candidate_rows = []
    for c in train+validation:
        mask = pseudo[c.image_name] if c in train else torch.as_tensor(rois[c.image_name]>0, device='cuda')
        overlay = targets[c.image_name]*.7 + mask[None]*torch.tensor([.3,0.,.3], device='cuda')[:,None,None]
        save_image(overlay, candidate_preview/f'{c.image_name}.jpg')
        candidate_rows.append(f'<h2>{c.image_name} ({"train candidate" if c in train else "validation ROI"})</h2><img width="80%" src="{c.image_name}.jpg">')
    (candidate_preview/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>학습 후보 / 고정 검증 영역</h1><p>보라색은 고오차 영역입니다. 유리 정답이나 깊이 정답이 아닙니다. 검증 영역은 손실에 사용하지 않습니다.</p>'+''.join(candidate_rows), encoding='utf-8')
    publish_index(preview_root, history)
    lpips_fn = None
    if cfg['iterations_per_branch'] >= cfg['rt_loss'].get('lpips_loss_from_iter',15000) and cfg['rt_loss'].get('lambda_lpips',.01)>0:
        from rt_port.vendor.loss_utils import lpips
        lpips_fn = lpips
        # Preload the network before restoring the checkpoint RNG state.
        with torch.no_grad(): lpips_fn(targets[train[0].image_name], targets[train[0].image_name])
    if resumed_rng is not None: restore_rng(resumed_rng)
    started = time.time()
    progress_path = baseline_run/'progress.json'
    log_path = output/'loss.jsonl'
    if resume and log_path.exists():
        # Remove only uncheckpointed log entries; authoritative state is latest.pth.
        entries = [line for line in log_path.read_text().splitlines() if json.loads(line)['iteration'] <= start_step]
        log_path.write_text('\n'.join(entries)+'\n')
    with log_path.open('a', encoding='utf-8') as log:
        for step in range(start_step+1, cfg['iterations_per_branch']+1):
            index, random_bg = draw_training_sample(schedule, rng, len(train), opt.random_background, background)
            camera = train[index]; target = targets[camera.image_name]
            control.update_learning_rate(baseline_iteration+step); treatment.update_learning_rate(baseline_iteration+step)
            control.optimizer.zero_grad(set_to_none=True)
            prediction = ex_render(camera, control, pipe, random_bg, near=args.near, far=args.far, training=True)['render']
            control_loss = (1-opt.lambda_dssim)*l1_loss(prediction,target)+opt.lambda_dssim*(1-ssim(prediction,target))
            if not torch.isfinite(control_loss): raise FloatingPointError('Nonfinite Ex4DGS loss')
            control_loss.backward(); finite_gradients([control.optimizer]); control.optimizer.step()
            control_value = control_loss.item(); del prediction, control_loss
            treatment.optimizer.zero_grad(set_to_none=True); material_optimizer.zero_grad(set_to_none=True)
            package = rt_render(camera,treatment,material,pipe,random_bg,near=args.near,far=args.far,training=True,
                                init_stage=step<cfg['rt_loss'].get('init_until_iter',0))
            loss, metrics = compute_rt_loss(package,target,pseudo[camera.image_name],package['occupancy'],cfg['rt_loss'],step,lpips_fn=lpips_fn)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite RT loss')
            loss.backward(); finite_gradients([treatment.optimizer,material_optimizer])
            treatment.optimizer.step(); material_optimizer.step()
            del package, loss
            row = {'iteration':step,'image':camera.image_name,'background':random_bg.detach().cpu().tolist(),
                   'control_loss':control_value,'rt_loss':metrics,'elapsed_seconds':time.time()-started,
                   'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
            log.write(json.dumps(row)+'\n')
            if step%25==0 or step==1:
                log.flush(); atomic_json(progress_path, {'stage':'rt_port_trial','status':'running','iteration':step,
                    'iterations_per_branch':cfg['iterations_per_branch'],'preview':f'{output.name}/index.html',**row})
                print(json.dumps(row),flush=True)
            if step%cfg['evaluate_every']==0 or step==cfg['iterations_per_branch']:
                observation = evaluate(step,validation,targets,rois,roi_meta,control,treatment,material,pipe,background,args,preview_root/f'step_{step:06d}')
                history.append(observation)
                atomic_json(output/'validation_history.json',history)
                interim = decide(history,cfg['decision'])
                atomic_json(output/'latest_decision.json',interim)
                publish_index(preview_root,history)
                atomic_save(output/'latest.pth',{'format_version':1,'iteration':step,'control':control.capture(),
                    'treatment':treatment.capture(),'material':material.state_dict(),'material_optimizer':material_optimizer.state_dict(),
                    'config':cfg,'baseline_config':baseline_cfg,'protocol':protocol,'history':history,
                    'schedule':schedule,'sampler_rng':rng.getstate(),'rng':save_rng()})
                if getattr(cli,'stop_after',None) == step and step < cfg['iterations_per_branch']:
                    atomic_json(output/'stopped.json',{'iteration':step,'reason':'requested checkpoint stop'})
                    atomic_json(progress_path,{'stage':'rt_port_trial','status':'checkpoint_saved','iteration':step})
                    return
    decision = decide(history,cfg['decision'])
    atomic_json(output/'decision.json',decision)  # Freeze decision before any test rendering.
    is_rt = decision['selected_model']=='rt_ex4dgs'
    atomic_save(output/'selected.pth',{'format_version':1,'selected_model':decision['selected_model'],
        'base':treatment.capture() if is_rt else control.capture(), 'material':material.state_dict() if is_rt else None,
        'config':cfg,'baseline_config':baseline_cfg,'protocol':protocol,'decision':decision,'iteration':cfg['iterations_per_branch']})
    test_metrics = evaluate(cfg['iterations_per_branch'],test,{}, {}, {},control,treatment,material,pipe,background,args,preview_root/'final_test',split='test')
    atomic_json(output/'test_metrics.json',test_metrics)
    publish_index(preview_root,history,decision)
    completed = {'stage':'rt_port_trial','status':'complete','selected_model':decision['selected_model'],
                 'iterations_per_branch':cfg['iterations_per_branch'],'preview':f'{output.name}/index.html'}
    atomic_json(output/'complete.json',completed); atomic_json(progress_path,completed)
    print(json.dumps(completed),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run',type=Path,required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/coffee_rt_port.json'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int,help='Diagnostic: return at this saved evaluation checkpoint')
    cli=parser.parse_args()
    try:
        run(cli)
    except Exception as exc:
        if cli.output.exists():
            atomic_json(cli.output/'failure.json',{'error':f'{type(exc).__name__}: {exc}'})
        if cli.baseline_run.exists():
            atomic_json(cli.baseline_run/'progress.json',{'stage':'rt_port_trial','status':'failed','error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__=='__main__': main()
