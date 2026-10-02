"""Restore Ex4DGS and reuse training residuals without a plane-model dependency."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene, getmodel
from rt_pipeline.hard_regions import persistent_candidates, expand_patches
from rt_pipeline.report import load_image, save_image


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def create_args(experiment, source, output, resolution=None):
    """Rebuild recorded Ex arguments without needing a checkpoint file.

    Returns (args, optimization_options, pipeline_options). ``experiment`` is
    the original baseline experiment object with its ``ex4dgs`` settings.
    """
    parser = argparse.ArgumentParser()
    ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    args = parser.parse_args([])
    for key, value in experiment['ex4dgs'].items():
        if not hasattr(args, key):
            raise ValueError(f'Unknown recorded Ex4DGS option: {key}')
        setattr(args, key, value)
    args.source_path = str(source)
    args.model_path = str(Path(output) / 'scene')
    args.loader = 'residual_n3dv'
    if resolution is not None:
        args.resolution = resolution
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    return args, optimization.extract(args), pipeline.extract(args)


def create_model(args):
    """Construct the same official Ex model class/configuration as baseline."""
    return getmodel()(
        args.sh_degree, args.start_duration, args.time_interval, args.time_pad,
        interp_type=args.interp_type, rot_interp_type=args.rot_interp_type,
        time_pad_type=args.time_pad_type, var_pad=args.var_pad,
        kernel_size=args.kernel_size,
    )


def load_baseline(baseline_run, source, output, resolution=None):
    """Restore the authoritative completed baseline, including Adam moments.

    The returned factory creates independent empty Ex models for restoring a
    second branch. No plane or material adapter is created here. Topology is
    kept fixed only after the baseline densification schedule has ended.
    """
    baseline_run = Path(baseline_run)
    experiment = json.loads((baseline_run / 'experiment.json').read_text(encoding='utf-8'))
    args, optimization, pipeline = create_args(experiment, source, output, resolution)
    factory = lambda: create_model(args)
    baseline = factory()
    scene = Scene(args, baseline, shuffle=False)
    state, iteration = torch.load(baseline_run / 'baseline_final.pth', weights_only=False)
    expected = int(experiment['pipeline']['baseline_iterations'])
    if iteration != expected:
        raise ValueError(f'Expected completed baseline step {expected}, got {iteration}')
    if iteration < optimization.densify_until_iter:
        raise ValueError('Fixed-topology continuation requires a post-densification baseline')
    baseline.restore(state, optimization)
    del state
    if baseline._xyz_motion.numel() != 0:
        raise ValueError('This RT continuation pilot has only been validated at frame zero, without dynamic Gaussians')
    cameras = scene.train_cameras[1.0] + scene.test_cameras[1.0]
    if not cameras or any(float(camera.timestamp) != 0.0 for camera in cameras):
        raise ValueError('This RT continuation pilot supports only timestamp zero')
    if baseline._xyz.shape[0] == 0:
        raise ValueError('The baseline checkpoint contains no static Gaussians')
    return scene, args, optimization, pipeline, experiment, iteration, factory


def load_training_masks(cameras, baseline_run, config, output):
    """Reuse fixed late TRAIN residuals as positive-only pseudo supervision.

    This is the existing persistent-candidate computation, without importing
    surface initialization code. No validation/test RGB determines these masks.
    The caller passes the already-verified training camera split only.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    masks, patches = {}, {}
    for camera in cameras:
        history = []
        for iteration in config['observe_iterations']:
            path = Path(baseline_run) / 'residuals' / f'{camera.image_name}_{iteration:06d}.npz'
            with np.load(path) as values:
                if int(values['iteration']) != iteration:
                    raise ValueError(f'Residual checkpoint mismatch: {path}')
                history.append({'mse': values['mse'].copy(), 'global_mse': float(values['global_mse'])})
        selected, stats = persistent_candidates(
            history, ratio=config['error_ratio'], floor=config['mse_floor'],
            persistence=config['persistence'], max_improvement=config['max_improvement'],
            min_observations=config['min_observations'],
        )
        size = int(config['patch_size'])
        expected_shape = ((camera.image_height + size - 1) // size, (camera.image_width + size - 1) // size)
        if selected.shape != expected_shape:
            raise ValueError('Residual grid and current camera resolution differ; regenerate consistent residuals')
        mask = expand_patches(selected, camera.image_height, camera.image_width, size)
        masks[camera.image_name] = mask
        patches[camera.image_name] = selected
        np.savez_compressed(output / f'{camera.image_name}.npz', positive=mask, candidate_patches=selected, **stats)
        device_mask = torch.as_tensor(mask, device='cuda')[None]
        save_image(device_mask.float(), output / f'{camera.image_name}_mask.png')
        overlay = load_image(camera) * .7 + device_mask * torch.tensor([.3, 0., .3], device='cuda')[:, None, None]
        save_image(overlay, output / f'{camera.image_name}_overlay.jpg')
    atomic_json(output / 'semantics.json', {
        'positive': 'temporary glass hypothesis',
        'outside': 'unknown, not a negative glass label',
        'source': 'late train residuals only',
        'positive_only_loss': True,
    })
    return masks, patches
