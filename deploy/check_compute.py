"""Time unchanged RT forward/backward under CUDA math settings (no training)."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtstatic.upstream import activate
activate(None)
import torch
from gaussian_renderer import render
from rtstatic.engine import load_model, make_camera
from rtstatic.data import validate_scene
from rtstatic.losses import loss_for_view

run = ROOT / "runs/coffee_rt_100k"
model, state = load_model(run / "checkpoint.pt")
config = state["config"]
opt = SimpleNamespace(**config["optimization"])
model.training_setup(opt)
scene = ROOT / "data/coffee_f000"
manifest = validate_scene(scene)
infos = [info for info in manifest["cameras"] if info["split"] == "train"]
cameras = [make_camera(scene, infos[index]) for index in (0, 8, 16)]
pipe = SimpleNamespace(init_stage=False, depth_ratio=0.)
background = torch.zeros(3, device="cuda")
report = {"checkpoint_step": state["step"], "gaussians": len(model.get_xyz), "modes": {}}
for label, tf32, benchmark in [("default", False, False), ("tf32", True, False), ("tf32_cudnn_tuned", True, True)]:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = benchmark
    times, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for repeat in range(2):
        for camera in cameras:
            model.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            package = render(camera, model, pipe, background)
            loss, _ = loss_for_view(package, camera.original_image.cuda(),
                camera.gt_transparent_mask.cuda(), model, opt, 16000, "rt", config["warmup"])
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss in math-settings check")
            if repeat:
                times.append(elapsed)
                losses.append(float(loss.detach()))
            del package, loss
    report["modes"][label] = dict(seconds=times, losses=losses,
        mean_seconds=sum(times)/len(times), peak_vram_bytes=torch.cuda.max_memory_allocated())
    print(json.dumps({label: report["modes"][label]}), flush=True)
(ROOT / "deploy/compute_check.json").write_text(json.dumps(report, indent=2))
