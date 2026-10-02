"""Exercise acceptance with synthetic validation targets; not scientific evidence."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from run_surface_hypotheses import load_baseline, evaluate_validation, merged_validation_guard
from rt_pipeline.surfaces import GlassSurfaces
from rt_pipeline.surface_decisions import decide_group

root = Path(__file__).resolve().parents[1]
trial = root/'checks/surface_smoke_final'
scene, args, opt, pipe, _, _, create_model = load_baseline(root/'checks/smoke_recovery_01',
    '/workspace/RTsplatting4d/data/coffee_martini_f000', root/'checks/surface_acceptance')
state = torch.load(trial/'latest.pth')
control = scene.gaussians; control.restore(state['control'], opt)
treatment = create_model(); treatment.restore(state['treatment'], opt)
surface = GlassSurfaces(state['surface_seeds']); surface.load_state_dict(state['surface'])
cfg = state['config']
cameras = [c for c in scene.test_cameras[1.0] if c.image_name.split('_')[0] in cfg['validation_cameras']]
bg = torch.zeros(3, device='cuda')
rois = {c.image_name:[torch.as_tensor(m, device='cuda') for m in np.load(trial/'fixed_validation_rois'/f'{c.image_name}.npz')['masks']] for c in cameras}
with torch.no_grad():
    synthetic_targets = {c.image_name:surface(c, treatment, pipe, bg, args.near, args.far)['render'].clamp(0, 1).detach() for c in cameras}
    history = [evaluate_validation(i, surface, control, treatment, cameras, synthetic_targets, rois, pipe, bg, args.near, args.far) for i in (1, 2, 3)]
    decisions = [decide_group(history, i, cfg['decision']) for i in range(surface.groups)]
    assert all(d['accepted'] for d in decisions), decisions
    assert merged_validation_guard(surface, control, treatment, cameras, synthetic_targets, rois, pipe, bg, args.near, args.far, cfg['decision'])['passes']
    surface.enabled.fill_(False)
    assert not merged_validation_guard(surface, control, treatment, cameras, synthetic_targets, rois, pipe, bg, args.near, args.far, cfg['decision'])['passes']
result = {'status':'passed', 'synthetic_targets':True, 'quality_evidence':False,
          'checks':['positive adoption path', 'merged model guard', 'all rejected fallback']}
(root/'checks/surface_acceptance/result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
