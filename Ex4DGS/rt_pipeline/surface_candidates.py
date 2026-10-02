"""Image-space pseudo masks and explicitly uncertain foreground plane seeds."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label
from scipy.spatial.transform import Rotation

from gaussian_renderer import render
from rt_pipeline.hard_regions import persistent_candidates, expand_patches
from rt_pipeline.report import load_image, save_image


def load_pseudo_masks(cameras, baseline_run, config, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    masks, patches = {}, {}
    for camera in cameras:
        history = []
        for iteration in config['observe_iterations']:
            with np.load(Path(baseline_run)/'residuals'/f'{camera.image_name}_{iteration:06d}.npz') as values:
                assert int(values['iteration']) == iteration
                history.append({'mse':values['mse'].copy(), 'global_mse':float(values['global_mse'])})
        selected, stats = persistent_candidates(history, ratio=config['error_ratio'], floor=config['mse_floor'],
            persistence=config['persistence'], max_improvement=config['max_improvement'], min_observations=config['min_observations'])
        mask = expand_patches(selected, camera.image_height, camera.image_width, config['patch_size'])
        masks[camera.image_name] = mask
        patches[camera.image_name] = selected
        np.savez_compressed(output/(camera.image_name+'.npz'), positive=mask, candidate_patches=selected, **stats)
        save_image(torch.as_tensor(mask, device='cuda')[None].float(), output/(camera.image_name+'_mask.png'))
        overlay = load_image(camera)*.7 + torch.as_tensor(mask, device='cuda')[None]*torch.tensor([.3, 0., .3], device='cuda')[:, None, None]
        save_image(overlay, output/(camera.image_name+'_overlay.jpg'))
    (output/'semantics.json').write_text(json.dumps({'positive':'temporary glass hypothesis',
        'outside':'unknown, not a negative glass label', 'source':'late train residuals only',
        'positive_only_loss':True}, indent=2))
    return masks, patches


def project(points, camera):
    world_view = camera.world_view_transform.detach().cpu().numpy()
    xyz = points @ world_view[:3, :3] + world_view[3, :3]
    z = xyz[:, 2]
    x = (xyz[:, 0]/np.maximum(z, 1e-8)/math.tan(camera.FoVx/2)+1)*camera.image_width/2-.5
    y = (xyz[:, 1]/np.maximum(z, 1e-8)/math.tan(camera.FoVy/2)+1)*camera.image_height/2-.5
    valid = (z > .01) & (x >= 0) & (x < camera.image_width) & (y >= 0) & (y < camera.image_height)
    return np.clip(x.astype(int), 0, camera.image_width-1), np.clip(y.astype(int), 0, camera.image_height-1), valid


@torch.no_grad()
def initialize_surfaces(cameras, base, pipe, background, masks, patches, patch_size, cfg, near, far, output):
    components = []
    for camera in cameras:
        labels, count = label(patches[camera.image_name])
        for index in range(1, count+1):
            ij = np.argwhere(labels == index)
            if len(ij) >= cfg['min_component_patches']:
                components.append((len(ij), camera, ij))
    components.sort(key=lambda v:(-v[0], v[1].image_name))
    covered = {c.image_name:np.zeros_like(patches[c.image_name]) for c in cameras}
    results = {k:[] for k in ('centers','quaternions','local_points','point_scales','move_limits','group_ids')}
    records = []
    xyz = base.get_xyz_at_t(0).detach().cpu().numpy()
    for _, camera, ij in components:
        if len(records) >= cfg['max_groups']:
            break
        if covered[camera.image_name][ij[:, 0], ij[:, 1]].mean() > .5:
            continue
        # A bounded planar sheet starts in front of the reconstructed background.
        # The background depth only brackets hypotheses, never labels a glass point.
        package = render(camera, base, pipe, background, near=near, far=far)
        dominant = package['dominent_idxs'].squeeze().cpu().numpy()
        pixels = ij[:, ::-1]*patch_size + patch_size*.5-.5
        pixels[:, 0] = pixels[:, 0].clip(0, camera.image_width-1)
        pixels[:, 1] = pixels[:, 1].clip(0, camera.image_height-1)
        ids = dominant[pixels[:, 1].astype(int), pixels[:, 0].astype(int)].astype(int)
        ids = ids[(ids >= 0) & (ids < len(xyz))]
        if len(ids) < 3:
            continue
        view = camera.world_view_transform.cpu().numpy()
        depths = (xyz[ids] @ view[:3, :3] + view[3, :3])[:, 2]
        depths = depths[depths > near*4]
        if len(depths) < 3:
            continue
        background_depth = float(np.median(depths))
        fx = camera.image_width/(2*math.tan(camera.FoVx/2))
        fy = camera.image_height/(2*math.tan(camera.FoVy/2))
        local_ray = np.c_[(pixels[:, 0]+.5-camera.image_width/2)/fx,
                         (pixels[:, 1]+.5-camera.image_height/2)/fy, np.ones(len(pixels))]
        inverse = np.linalg.inv(view)
        camera_rotation = inverse[:3, :3].T
        choices = []
        for fraction in cfg['depth_fractions']:
            depth = background_depth*fraction
            points = local_ray*depth @ inverse[:3, :3] + inverse[3, :3]
            votes = np.zeros(len(points), dtype=int)
            for other in cameras:
                x, y, valid = project(points, other)
                votes += valid & masks[other.image_name][y, x]
            keep = votes >= cfg['min_train_views']
            choices.append((int(keep.sum()), float(votes[keep].mean()) if keep.any() else 0., fraction, points, keep))
        choices.sort(key=lambda choice:(-choice[0], -choice[1], choice[2]))
        count, _, fraction, points, keep = choices[0]
        if count < cfg['min_surface_points']:
            continue
        points = points[keep]
        if len(points) > cfg['max_points_per_group']:
            # Avoid sparse holes from subsampling; skip oversized components instead.
            continue
        center = points.mean(0)
        local_points = (points-center) @ camera_rotation
        quat_xyzw = Rotation.from_matrix(camera_rotation).as_quat()
        quat = quat_xyzw[[3, 0, 1, 2]]
        depth = background_depth*fraction
        sx, sy = patch_size*depth/fx*.75, patch_size*depth/fy*.75
        group = len(records)
        results['centers'].append(center)
        results['quaternions'].append(quat)
        results['move_limits'].append(depth*cfg['translation_bound_fraction'])
        results['local_points'].extend(local_points)
        results['point_scales'].extend(np.tile([sx, sy, min(sx, sy)*.02], (len(points), 1)))
        results['group_ids'].extend([group]*len(points))
        support_names = []
        for other in cameras:
            x, y, valid = project(points, other)
            if valid.any():
                covered[other.image_name][y[valid]//patch_size, x[valid]//patch_size] = True
            if np.count_nonzero(valid & masks[other.image_name][y, x]) >= cfg['min_surface_points']:
                support_names.append(other.image_name)
        records.append({'group':group, 'anchor_camera':camera.image_name, 'points':len(points),
            'background_depth_bracket':background_depth, 'initial_depth_fraction':fraction,
            'train_support_views':support_names,
            'depth_is_measured_glass_depth':False, 'prior':'camera-facing thin planar sheet; learnable rigid pose and extent',
            'depth_choices':[{'fraction':v[2], 'supported_points':v[0]} for v in choices]})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'initialization.json').write_text(json.dumps({'groups':records, 'uses_validation_rgb':False,
        'uses_test':False, 'limitation':'Residual-mask frustum overlap does not uniquely determine glass depth.'}, indent=2))
    return {k:np.asarray(v) for k, v in results.items()}, records
