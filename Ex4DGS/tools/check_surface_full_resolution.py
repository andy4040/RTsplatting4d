"""One finite-gradient update at production resolution with a snapshot of the live baseline."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from run_surface_hypotheses import load_baseline
from rt_pipeline.surfaces import GlassSurfaces
from rt_pipeline.report import load_image
from utils.loss_utils import l1_loss, ssim

root = Path(__file__).resolve().parents[1]
scene, args, opt, pipe, _, _, _ = load_baseline(root/'checks/smoke_recovery_01',
    '/workspace/RTsplatting4d/data/coffee_martini_f000', root/'checks/surface_full_resolution', resolution=2)
snapshot, iteration = torch.load(root/'runs/coffee_f000_residual_rt/baseline_latest.pth')
scene.gaussians.restore(snapshot, opt)
initial = torch.load(root/'checks/surface_smoke_01/surface_initial.pth')
surface = GlassSurfaces(initial['seeds']); surface.load_state_dict(initial['state_dict'])
camera = scene.train_cameras[1.0][0]; target = load_image(camera)
torch.cuda.reset_peak_memory_stats()
started = time.time()
package = surface(camera, scene.gaussians, pipe, torch.zeros(3, device='cuda'), args.near, args.far)
loss = .8*l1_loss(package['render'], target) + .2*(1-ssim(package['render'], target)) + .01*(1-package['surface_coverage']).mean()
loss.backward()
for model_parameters in [surface.parameters(), (p for p in vars(scene.gaussians).values() if isinstance(p, torch.nn.Parameter))]:
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model_parameters)
scene.gaussians.optimizer.step()
result = {'status':'passed', 'baseline_snapshot_iteration':iteration,
    'resolution':camera.resolution, 'background_points':len(scene.gaussians.get_xyz_at_t(0)),
    'surface_points':len(surface.group_ids), 'loss':float(loss), 'seconds':time.time()-started,
    'peak_allocated_gb':torch.cuda.max_memory_allocated()/1024**3,
    'baseline_checkpoint_modified':False}
(root/'checks/surface_full_resolution/result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
