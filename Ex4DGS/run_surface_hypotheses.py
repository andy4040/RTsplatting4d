"""Reuse a completed Ex4DGS baseline; test independent glass-surface hypotheses."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene, getmodel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from rt_pipeline.report import load_image, save_image, write_preview
from rt_pipeline.surface_candidates import load_pseudo_masks, initialize_surfaces
from rt_pipeline.surface_decisions import decide_group, psnr_gain
from rt_pipeline.surfaces import GlassSurfaces, CombinedGaussians


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def atomic_save(path, value):
    path = Path(path); temp = path.with_suffix('.tmp')
    torch.save(value, temp); temp.replace(path)


@torch.no_grad()
def publish_candidates(folder, output, records, cameras, surface, base, pipe, bg, near, far):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for camera in cameras:
        name = camera.image_name
        for suffix in ('_mask.png', '_overlay.jpg'):
            shutil.copy2(output/'pseudo_masks'/f'{name}{suffix}', folder/f'{name}{suffix}')
        images = [f'{name}_overlay.jpg']
        if surface is not None:
            colors = torch.cat((torch.zeros((len(base.get_xyz_at_t(0)), 3), device='cuda'),
                                torch.ones((len(surface.group_ids), 3), device='cuda')))
            coverage = render(camera, CombinedGaussians(base, surface), pipe, torch.zeros_like(bg),
                override_color=colors, near=near, far=far)['render'][:1]
            overlay = .7*load_image(camera) + coverage*torch.tensor([0., .3, .3], device='cuda')[:, None, None]
            save_image(overlay, folder/f'{name}_surface.jpg')
            images.append(f'{name}_surface.jpg')
        rows.append('<h2>'+name+'</h2>'+''.join(f'<img style="width:47%;margin:1%" src="{image}">' for image in images))
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>임시 유리 후보와 표면 가설</h1>'
        '<p>보라색: 지속 고오차 후보. 청록색: 별도로 초기화한 3D 표면의 투영. 후보 밖은 미확인 영역입니다. '
        '표면 깊이는 정답이 아니며, 여러 학습 카메라의 후보 영역과 겹치는 얇은 평면으로 초기화했습니다.</p>'
        '<pre>'+html.escape(json.dumps(records, indent=2))+'</pre>'+''.join(rows), encoding='utf-8')


def load_baseline(baseline_run, source, output, resolution=None):
    experiment = json.loads((baseline_run/'experiment.json').read_text())
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
    args = parser.parse_args([])
    for key, value in experiment['ex4dgs'].items():
        setattr(args, key, value)
    args.source_path = source; args.model_path = str(output/'scene'); args.loader = 'residual_n3dv'
    if resolution is not None:
        args.resolution = resolution
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    def create_model():
        return getmodel()(args.sh_degree, args.start_duration, args.time_interval, args.time_pad,
            interp_type=args.interp_type, rot_interp_type=args.rot_interp_type, time_pad_type=args.time_pad_type,
            var_pad=args.var_pad, kernel_size=args.kernel_size)
    base = create_model()
    scene = Scene(args, base, shuffle=False)
    state, iteration = torch.load(baseline_run/'baseline_final.pth')
    assert iteration == experiment['pipeline']['baseline_iterations']
    base.restore(state, op.extract(args))
    assert base._xyz_motion.numel() == 0, 'First-frame surface pilot has not validated moving glass'
    assert all(c.timestamp == 0 for c in scene.train_cameras[1.0]), 'Single-frame pilot only'
    return scene, args, op.extract(args), pp.extract(args), experiment, iteration, create_model


@torch.no_grad()
def freeze_validation_rois(surface, base, cameras, pipe, bg, near, far, threshold, folder):
    """Geometry-only projections, fixed before any branch optimization or validation RGB read."""
    folder.mkdir(parents=True, exist_ok=True)
    rois = {}
    for camera in cameras:
        regions = []
        for group in range(surface.groups):
            enabled = torch.arange(surface.groups, device='cuda') == group
            proxy = CombinedGaussians(base, surface, enabled=enabled)
            colors = torch.cat((torch.zeros((len(base.get_xyz_at_t(0)), 3), device='cuda'),
                                torch.ones((len(surface.group_ids), 3), device='cuda')))
            coverage = render(camera, proxy, pipe, torch.zeros_like(bg), override_color=colors,
                              near=near, far=far)['render'][0]
            mask = coverage > threshold
            regions.append(mask)
            save_image(mask[None].float(), folder/f'{camera.image_name}_group{group}.png')
        rois[camera.image_name] = regions
        np.savez_compressed(folder/f'{camera.image_name}.npz', masks=torch.stack(regions).cpu().numpy())
    return rois


@torch.no_grad()
def evaluate_validation(step, surface, control, treatment, cameras, targets, rois, pipe, bg, near, far):
    groups = {str(i):[] for i in range(surface.groups)}
    outside_control_sum = outside_treatment_sum = 0.
    outside_pixels = 0
    for camera in cameras:
        target = targets[camera.image_name]
        c = render(camera, control, pipe, bg, near=near, far=far)['render'].clamp(0, 1)
        t = surface(camera, treatment, pipe, bg, near, far)['render'].clamp(0, 1)
        c_error = (c-target).square().mean(0)
        t_error = (t-target).square().mean(0)
        union = torch.stack(rois[camera.image_name]).any(0)
        outside_control_sum += c_error[~union].sum().item()
        outside_treatment_sum += t_error[~union].sum().item()
        outside_pixels += int((~union).sum())
        for group, mask in enumerate(rois[camera.image_name]):
            count = int(mask.sum())
            if count:
                enabled = surface.enabled.clone(); enabled[group] = False
                removed = surface(camera, treatment, pipe, bg, near, far, enabled=enabled)['render'].clamp(0, 1)
                cm = c_error[mask].mean().item(); tm = t_error[mask].mean().item()
                rm = (removed-target).square().mean(0)[mask].mean().item()
                row = {'image':camera.image_name, 'pixels':count, 'control_mse':cm, 'surface_mse':tm,
                       'gain_db':psnr_gain(cm, tm), 'ablation_gain_db':psnr_gain(rm, tm)}
            else:
                row = {'image':camera.image_name, 'pixels':0, 'gain_db':None, 'ablation_gain_db':None}
            groups[str(group)].append(row)
    increase = (outside_treatment_sum-outside_control_sum)/max(outside_control_sum, 1e-12)
    return {'iteration':step, 'groups':groups, 'outside_pixels':outside_pixels,
            'outside_relative_mse_increase':increase, 'test_used':False}


@torch.no_grad()
def merged_validation_guard(surface, control, treatment, cameras, targets, rois, pipe, bg, near, far, decision_cfg):
    """Removal changes the composite: recheck the actual selected model before test."""
    gains = []; c_out = t_out = 0.
    active = surface.enabled.nonzero(as_tuple=True)[0].tolist()
    if not active:
        return {'passes':False, 'reason':'no accepted groups'}
    for camera in cameras:
        target = targets[camera.image_name]
        c = render(camera, control, pipe, bg, near=near, far=far)['render'].clamp(0, 1)
        t = surface(camera, treatment, pipe, bg, near, far)['render'].clamp(0, 1)
        ce = (c-target).square().mean(0); te = (t-target).square().mean(0)
        mask = torch.stack([rois[camera.image_name][i] for i in active]).any(0)
        if int(mask.sum()) >= decision_cfg['min_validation_pixels']:
            gains.append(psnr_gain(ce[mask].mean().item(), te[mask].mean().item()))
        c_out += ce[~mask].sum().item(); t_out += te[~mask].sum().item()
    outside = (t_out-c_out)/max(c_out, 1e-12)
    passed = (len(gains) >= decision_cfg['min_validation_views']
              and np.mean(gains) >= decision_cfg['min_gain_db']
              and min(gains) >= decision_cfg['min_per_view_gain_db']
              and outside <= decision_cfg['max_outside_mse_increase'])
    return {'passes':bool(passed), 'per_view_gain_db':gains, 'outside_relative_mse_increase':outside}


def main(cli):
    output = cli.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing surface trial')
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(cli.config.read_text())
    assert set(cfg['validation_cameras']).isdisjoint(cfg['test_cameras'])
    assert cfg['decision']['required_passes'] <= cfg['decision']['decision_observations']
    atomic_json(output/'experiment.json', cfg)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    scene, args, opt, pipe, baseline_cfg, baseline_iteration, create_model = load_baseline(cli.baseline_run, cli.source, output)
    base = scene.gaussians
    train = sorted(scene.train_cameras[1.0], key=lambda c:c.image_name)
    heldout = scene.test_cameras[1.0]
    validation = [c for c in heldout if c.image_name.split('_')[0] in cfg['validation_cameras']]
    test = [c for c in heldout if c.image_name.split('_')[0] in cfg['test_cameras']]
    assert len(validation) == len(cfg['validation_cameras']) and len(test) == len(cfg['test_cameras'])
    assert set(c.image_name for c in train).isdisjoint(c.image_name for c in validation+test)
    assert len(validation)+len(test) == len(heldout)
    bg = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda')
    preview_root = cli.baseline_run/'previews'
    progress_path = cli.baseline_run/'progress.json'
    atomic_json(progress_path, {'stage':'surface_initialization', 'status':'running'})
    baseline_hash = hashlib.sha256((cli.baseline_run/'baseline_final.pth').read_bytes()).hexdigest()
    atomic_json(output/'protocol.json', {'baseline_sha256':baseline_hash,
        'train':[c.image_name for c in train], 'validation':[c.image_name for c in validation],
        'test':[c.image_name for c in test], 'test_used_for_selection':False,
        'prior_test_exposure':'All three held-out views were already shown in earlier baseline previews.',
        'same_optimizer_state':True, 'same_camera_order':True, 'same_iteration_budget':True,
        'same_wall_clock_budget':False, 'branch_topology':'fixed',
        'method':'RT-inspired independent planar thin 3D Gaussian surfaces; no official RT reproduction claim'})
    masks, patches = load_pseudo_masks(train, cli.baseline_run, baseline_cfg['pipeline'], output/'pseudo_masks')
    seeds, records = initialize_surfaces(train, base, pipe, bg, masks, patches, baseline_cfg['pipeline']['patch_size'],
        cfg['initialization'], args.near, args.far, output/'surfaces')
    if not records:
        publish_candidates(preview_root/'surface_candidates', output, records, train, None, base, pipe, bg, args.near, args.far)
        result = {'status':'skipped_no_supported_surfaces', 'interpretation':'No supported initialization, not proof that glass is absent.'}
        atomic_json(output/'decision.json', result)
        atomic_save(output/'selected.pth', {'background':base.capture(), 'selected_model':'baseline',
                    'baseline_config':baseline_cfg, 'trial_config':cfg, 'decision':result})
        write_preview(preview_root, 'surface_no_candidates_final_test', test,
            lambda c:render(c, base, pipe, bg, near=args.near, far=args.far), result)
        atomic_json(output/'complete.json', result); atomic_json(progress_path, result)
        return
    surface = GlassSurfaces(seeds)
    publish_candidates(preview_root/'surface_candidates', output, records, train, surface, base, pipe, bg, args.near, args.far)
    torch.save({'seeds':surface.seeds(), 'state_dict':surface.state_dict()}, output/'surface_initial.pth')
    rois = freeze_validation_rois(surface, base, validation, pipe, bg, args.near, args.far,
        cfg['decision']['fixed_roi_coverage_threshold'], output/'fixed_validation_rois')
    # Validation targets are first loaded after hypotheses and evaluation regions freeze.
    targets = {c.image_name:load_image(c) for c in train+validation}
    pseudo = {name:torch.as_tensor(mask, device='cuda') for name, mask in masks.items()}
    control = base
    treatment = create_model()
    state, _ = torch.load(cli.baseline_run/'baseline_final.pth')
    treatment.restore(state, opt); del state
    assert control._xyz.data_ptr() != treatment._xyz.data_ptr()
    torch.testing.assert_close(control._xyz, treatment._xyz, rtol=0, atol=0)
    surface_optimizer = torch.optim.Adam(surface.parameters(), lr=cfg['optimization']['surface_lr'])
    schedule = []; rng = random.Random(cfg['seed'])
    history = []
    started = time.time()
    with (output/'loss.jsonl').open('w') as log:
        for step in range(1, cfg['iterations_per_branch']+1):
            if not schedule:
                schedule = list(train); rng.shuffle(schedule)
            camera = schedule.pop(); target = targets[camera.image_name]
            random_bg = torch.rand(3, device='cuda') if opt.random_background else bg
            control.update_learning_rate(baseline_iteration+step)
            treatment.update_learning_rate(baseline_iteration+step)
            control.optimizer.zero_grad(set_to_none=True)
            prediction = render(camera, control, pipe, random_bg, near=args.near, far=args.far)['render']
            control_loss = (1-opt.lambda_dssim)*l1_loss(prediction, target)+opt.lambda_dssim*(1-ssim(prediction, target))
            if not torch.isfinite(control_loss): raise FloatingPointError('Non-finite control loss')
            control_loss.backward(); control.optimizer.step()
            del prediction
            treatment.optimizer.zero_grad(set_to_none=True); surface_optimizer.zero_grad(set_to_none=True)
            package = surface(camera, treatment, pipe, random_bg, args.near, args.far)
            photo = (1-opt.lambda_dssim)*l1_loss(package['render'], target)+opt.lambda_dssim*(1-ssim(package['render'], target))
            mask = pseudo[camera.image_name]
            positive_loss = (1-package['surface_coverage'][0][mask]).mean() if mask.any() else photo*0
            regularization = surface.regularization()
            loss = photo + cfg['optimization']['positive_mask_weight']*positive_loss + cfg['optimization']['geometry_regularization_weight']*regularization
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite surface loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surface.parameters(), cfg['optimization']['gradient_clip'], error_if_nonfinite=True)
            treatment.optimizer.step(); surface_optimizer.step()
            del package
            if step == 1 or step % 10 == 0:
                row = {'iteration':step, 'camera':camera.image_name, 'control_photo':control_loss.item(),
                    'surface_photo':photo.item(), 'positive_mask':positive_loss.item(), 'total_surface_loss':loss.item()}
                log.write(json.dumps(row)+'\n'); log.flush()
            if step == 1 or step % 100 == 0:
                status = {'stage':'surface_hypothesis_trial', 'iteration':step, 'total':cfg['iterations_per_branch'],
                    'status':'running', 'groups':surface.groups, 'elapsed_seconds':time.time()-started}
                atomic_json(progress_path, status); atomic_json(output/'progress.json', status)
                print(json.dumps(status), flush=True)
            if step % cfg['preview_every'] == 0 or step == cfg['iterations_per_branch']:
                observation = evaluate_validation(step, surface, control, treatment, validation, targets, rois, pipe, bg, args.near, args.far)
                history.append(observation)
                atomic_json(output/'validation_history.json', history)
                write_preview(preview_root, f'surface_control_{step:06d}', validation,
                    lambda c:render(c, control, pipe, bg, near=args.near, far=args.far), {'stage':'control', 'split':'validation'})
                write_preview(preview_root, f'surface_trial_{step:06d}', validation,
                    lambda c:surface(c, treatment, pipe, bg, args.near, args.far), {'stage':'surface_trial', 'split':'validation', 'evidence':observation})
                atomic_save(output/'latest.pth', {'iteration':step, 'control':control.capture(),
                    'treatment':treatment.capture(), 'surface_seeds':surface.seeds(), 'surface':surface.state_dict(),
                    'surface_optimizer':surface_optimizer.state_dict(), 'validation_history':history,
                    'rng':rng.getstate(), 'schedule':[c.image_name for c in schedule],
                    'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state(), 'config':cfg})
    decisions = {str(i):decide_group(history, i, cfg['decision']) for i in range(surface.groups)}
    surface.enabled.copy_(torch.tensor([decisions[str(i)]['accepted'] for i in range(surface.groups)], device='cuda'))
    merged = merged_validation_guard(surface, control, treatment, validation, targets, rois, pipe, bg, args.near, args.far, cfg['decision'])
    selected = 'surface' if merged['passes'] else 'control'
    if selected == 'control':
        surface.enabled.fill_(False)
    decision = {'status':'complete', 'groups':decisions, 'merged_validation':merged, 'selected_model':selected,
        'retained_groups':surface.enabled.nonzero(as_tuple=True)[0].tolist(), 'test_used_for_decision':False,
        'interpretation':'Acceptance indicates useful representation, not a verified glass label.'}
    atomic_json(output/'decision.json', decision)
    atomic_save(output/'selected.pth', {'background':treatment.capture() if selected == 'surface' else control.capture(),
        'surface_seeds':surface.seeds(), 'surface':surface.state_dict(), 'selected_model':selected,
        'baseline_config':baseline_cfg, 'trial_config':cfg, 'decision':decision})
    def final_render(camera):
        return surface(camera, treatment, pipe, bg, args.near, args.far) if selected == 'surface' else render(camera, control, pipe, bg, near=args.near, far=args.far)
    # Test RGB is first read here, after the selected checkpoint and decision are saved.
    final = write_preview(preview_root, 'surface_selected_final_test', test, final_render,
        {'stage':'final_test', 'selection':decision, 'split':'test'})
    atomic_json(output/'final_test.json', final)
    atomic_json(output/'complete.json', decision)
    atomic_json(progress_path, {'stage':'surface_hypothesis_trial', 'status':'complete', 'selected_model':selected})
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    try:
        main(cli)
    except Exception as exc:
        if cli.output.exists():
            atomic_json(cli.output/'failure.json', {'status':'failed', 'error':repr(exc)})
        atomic_json(cli.baseline_run/'progress.json', {'stage':'surface_hypothesis_trial', 'status':'failed', 'error':repr(exc)})
        raise
