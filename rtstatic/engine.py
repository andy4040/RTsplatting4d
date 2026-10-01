"""Official GaussianModel + RT renderer with an explicit held-out-camera protocol.

The `gs` control uses the same surfel rasterizer with a single opacity and SH
appearance. It is a controlled 2DGS-style baseline, not official 3DGS results.
"""
import argparse
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .data import load_mask, read_json, validate_scene, write_json
from .metrics import all_metrics
from .upstream import activate


def make_camera(root, info):
    import torch
    from scene.cameras import Camera
    root = Path(root)
    rgb = np.array(Image.open(root / "images" / f"{info['name']}.png").convert("RGB"), copy=True)
    mask = load_mask(root / "masks" / f"{info['name']}.png", (info["width"], info["height"]))
    k = np.asarray(info["K"])
    return Camera(colmap_id=int(info["name"][3:]), R=np.asarray(info["R"]).T, T=np.asarray(info["T"]),
                  FoVx=2 * math.atan(info["width"] / (2 * k[0, 0])),
                  FoVy=2 * math.atan(info["height"] / (2 * k[1, 1])),
                  image=torch.from_numpy(rgb).permute(2, 0, 1).float() / 255,
                  gt_alpha_mask=None, gt_transparent_mask=torch.from_numpy(mask.copy())[None],
                  image_name=info["name"], uid=int(info["name"][3:]), data_device="cpu")


def render_gs(camera, model, pipe, background):
    """One forward rasterization; no learned reflection/transmission decomposition."""
    import torch
    import torch.nn.functional as functional
    from diff_surfel_anych import GaussianRasterizationSettings, GaussianRasterizer
    from utils.point_utils import depth_to_normal_sobel

    means2d = torch.zeros_like(model.get_xyz, requires_grad=True)
    if torch.is_grad_enabled():
        means2d.retain_grad()
    settings = GaussianRasterizationSettings(
        image_height=camera.image_height, image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
        bg=background, scale_modifier=1., viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform, sh_degree=model.active_sh_degree,
        campos=camera.camera_center, prefiltered=False, debug=False)
    rgb, _, radii, maps = GaussianRasterizer(settings)(
        means3D=model.get_xyz, means2D=means2d, shs=model.get_features,
        extras=torch.ones_like(model.get_occupancy), opacities=model.get_occupancy,
        scales=model.get_scaling, rotations=model.get_rotation, cov3D_precomp=None)
    alpha = maps[1:2]
    normal = (maps[2:5].movedim(0, -1) @ camera.world_view_transform[:3, :3].T).movedim(-1, 0)
    normal = functional.normalize(normal, dim=0)
    depth = torch.nan_to_num(maps[0:1] / alpha, nan=0, posinf=0, neginf=0)
    depth_normal = depth_to_normal_sobel(camera, depth.movedim(0, -1)).movedim(-1, 0) * alpha.detach()
    return dict(final_rendering=rgb, surface_normal=normal, surface_depth_normal=depth_normal,
                foreground=torch.ones_like(alpha), viewspace_points=means2d,
                visibility_filter=radii > 0, radii=radii)


def plain_scalars(value):
    """Keep upstream NumPy learning-rate scalars out of portable checkpoints."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain_scalars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(plain_scalars(item) for item in value)
    return value


def read_checkpoint(path):
    import importlib
    import pickle
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        if not hasattr(torch.serialization, "safe_globals"):
            raise ValueError("This early NumPy optimizer checkpoint needs PyTorch >= 2.6 to load safely")
    # Early optimizer checkpoints contain NumPy float64 learning rates from
    # the official scheduler. Allow only scalar reconstruction, including the
    # NumPy 1.x module name when reading on a NumPy 2.x workstation.
    core = "numpy._core.multiarray" if int(np.__version__.split(".")[0]) >= 2 else "numpy.core.multiarray"
    scalar = importlib.import_module(core).scalar
    safe = [scalar, (scalar, "numpy.core.multiarray.scalar"),
            (scalar, "numpy._core.multiarray.scalar"), np.dtype,
            type(np.dtype("float64")), type(np.dtype("float32"))]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location="cpu", weights_only=True)


def save_model(model, path, config, model_args, step, training_state=None):
    import torch
    # Upstream capture() omits optical/material parameters and the reflection MLP.
    # Save every Gaussian parameter plus both neural modules as tensor state dicts.
    parameters = {name: value.detach().cpu() for name, value in vars(model).items()
                  if isinstance(value, torch.nn.Parameter)}
    state = dict(format_version=1, parameters=parameters,
                    light_mlp={k: v.cpu() for k, v in model.light_mlp.state_dict().items()},
                    dir_encoding={k: v.cpu() for k, v in model.dir_encoding.state_dict().items()},
                    active_sh_degree=model.active_sh_degree, model_args=vars(model_args),
                    config=config, step=step)
    if training_state is not None:
        state["training_state"] = dict(training_state,
            optimizer=model.optimizer.state_dict(),
            buffers={name: getattr(model, name).detach().cpu() for name in
                     ["max_radii2D", "xyz_gradient_accum", "denom", "last_update"]},
            spatial_lr_scale=model.spatial_lr_scale,
            python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all())
    temporary = Path(path).with_suffix(".tmp")
    torch.save(plain_scalars(state), temporary)
    temporary.replace(path)


def load_model(path):
    import torch
    from scene.gaussian_model import GaussianModel
    # Files contain tensor state dicts and plain metadata, not pickled modules.
    state = read_checkpoint(path)
    if state["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    model = GaussianModel(state["model_args"]["sh_degree"], SimpleNamespace(**state["model_args"]))
    for name, tensor in state["parameters"].items():
        setattr(model, name, torch.nn.Parameter(tensor.cuda()))
    model.light_mlp.load_state_dict(state["light_mlp"])
    model.dir_encoding.load_state_dict(state["dir_encoding"])
    model.active_sh_degree = state["active_sh_degree"]
    return model, state


def evaluate_model(model, scene, manifest, mode, out, lpips_enabled=False):
    import torch
    from gaussian_renderer import render
    from utils.loss_utils import lpips
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    renderer = render_gs if mode == "gs" else render
    pipe = SimpleNamespace(init_stage=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    per_view = {}
    with torch.no_grad():
        for info in manifest["cameras"]:
            if info["split"] != "test":
                continue
            camera = make_camera(scene, info)
            package = renderer(camera, model, pipe, background)
            prediction = package["final_rendering"].clamp(0, 1)
            target = camera.original_image.to("cuda")
            pred = prediction.permute(1, 2, 0).cpu().numpy()
            gt = target.permute(1, 2, 0).cpu().numpy()
            mask = camera.gt_transparent_mask[0].numpy()
            result = all_metrics(pred, gt, mask)
            result["lpips_full"] = float(lpips(prediction, target)) if lpips_enabled else None
            per_view[info["name"]] = result
            folder = out / info["name"]
            folder.mkdir()
            np.save(folder / "prediction.npy", pred)
            for key, image in [("prediction", pred), ("target", gt)]:
                Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(folder / f"{key}.png")
            for key in ["final_tran", "final_scat", "final_spec"]:
                if key in package:
                    layer = package[key].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    Image.fromarray(np.rint(layer * 255).astype(np.uint8)).save(folder / f"{key}.png")
            error = np.mean(np.abs(pred - gt), axis=-1)
            heat = np.zeros_like(pred)
            heat[..., 0] = (error * 5).clip(0, 1)
            Image.fromarray(np.rint(np.concatenate([gt, pred, heat], axis=1) * 255).astype(np.uint8)).save(folder / "comparison.png")
            ys, xs = np.nonzero(mask)
            if len(xs):
                crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
                Image.fromarray(np.rint(np.concatenate([gt[crop], pred[crop]], axis=1) * 255).astype(np.uint8)).save(folder / "window_crop.png")
    report = dict(mode=mode, scene_fingerprint=manifest["fingerprint"], frame=manifest["frame"],
                  per_view=per_view, ground_truth="observed image including reflections; no clean transmission GT",
                  metric_notes="PSNR uses selected pixels only (cap 120 dB); SSIM uses fully contained 11x11 windows. LPIPS, if requested, is full-image only.")
    write_json(out / "metrics.json", report)
    return report


def save_preview(model, scene, manifest, mode, out, step, config, elapsed):
    """Read-only held-out render; preserve RNG state for subsequent training."""
    import shutil
    import torch
    preview = out / "previews" / f"step_{step:06d}"
    if (preview / "training_summary.json").exists():
        return
    python_rng, torch_rng = random.getstate(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    try:
        evaluate_model(model, scene, manifest, mode, preview / "evaluation", config["lpips"])
        write_json(preview / "config.json", config)
        shutil.copyfile(out / "losses.jsonl", preview / "losses.jsonl")
        write_json(preview / "training_summary.json", dict(steps=step, seconds=elapsed,
            gaussians=len(model.get_xyz), peak_vram_bytes=torch.cuda.max_memory_allocated(),
            intermediate=True))
        print(f"Saved held-out preview at iteration {step}: {preview}", flush=True)
    finally:
        random.setstate(python_rng)
        torch.set_rng_state(torch_rng)
        torch.cuda.set_rng_state_all(cuda_rng)


def trim_resume_log(out, checkpoint_step):
    """Preserve discarded unsaved steps separately instead of plotting them twice."""
    import json
    path = out / "losses.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    kept, discarded = [], []
    for line in lines:
        (kept if json.loads(line)["step"] <= checkpoint_step else discarded).append(line)
    if discarded:
        archive = out / f"discarded_after_{checkpoint_step}_{time.time_ns()}.jsonl"
        archive.write_text("\n".join(discarded) + "\n", encoding="utf-8")
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return len(discarded)


def train(args):
    # Validate data before expensive CUDA initialization, refuse output reuse.
    scene, out = Path(args.scene).resolve(), Path(args.out).resolve()
    manifest = validate_scene(scene)
    resume = getattr(args, "resume", None)
    if out.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {out}")
    if resume and Path(resume).resolve().parent != out:
        raise ValueError("Resume checkpoint must be in the original output directory")
    if args.iterations <= args.warmup or args.warmup < 1:
        raise ValueError("iterations must exceed warmup >= 1")
    stop_step = getattr(args, "stop_at", None) or args.iterations
    preview_every = getattr(args, "preview_every", 0)
    if not args.warmup < stop_step <= args.iterations or preview_every < 0:
        raise ValueError("Require warmup < stop-at <= iterations and preview-every >= 0")
    if args.mode not in ("gs", "rt", "rt_no_gating"):
        raise ValueError("Invalid mode")
    commit = activate(args.upstream)
    import torch
    from arguments import ModelParams, OptimizationParams
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud
    from .losses import loss_for_view

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    parser = argparse.ArgumentParser()
    mp, op = ModelParams(parser), OptimizationParams(parser)
    defaults = parser.parse_args([])
    model_args, opt = mp.extract(defaults), op.extract(defaults)
    model_args.source_path, model_args.model_path = str(scene), str(out)
    if args.env_radius is not None:
        if args.env_radius <= 0:
            raise ValueError("env-radius must be positive")
        model_args.env_scope_radius = args.env_radius
        model_args.env_scope_center = list(args.env_center)
        model_args.xyz_axis = [0, 1, 2]
    opt.iterations = args.iterations
    opt.position_lr_max_steps = args.iterations
    opt.densify_until_iter = min(15000, args.iterations // 2)
    opt.lambda_lpips = 0.01 if args.lpips else 0.0
    config = dict(scene=str(scene), mode=args.mode, seed=args.seed, iterations=args.iterations,
                  stop_at=stop_step, preview_every=preview_every,
                  warmup=args.warmup, lpips=args.lpips, upstream_commit=commit,
                  scene_fingerprint=manifest["fingerprint"], optimization=vars(opt),
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(),
                  baseline="single-opacity SH surfel control; not official 3DGS benchmark",
                  checkpoint_purpose="inference and training resume with optimizer/RNG state")
    out.mkdir(parents=True, exist_ok=bool(resume))
    if resume:
        previous = read_json(out / "config.json")
        for key in ["scene_fingerprint", "mode", "seed", "iterations", "warmup", "lpips", "upstream_commit"]:
            if previous[key] != config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
    write_json(out / "config.json", config)
    train_infos = [c for c in manifest["cameras"] if c["split"] == "train"]
    write_json(out / "split_audit.json", dict(train=[c["name"] for c in train_infos],
               test=[c["name"] for c in manifest["cameras"] if c["split"] == "test"],
               initialization=manifest["initialization_cameras"], frame=manifest["frame"]))
    cameras = [make_camera(scene, c) for c in train_infos]
    cloud = np.load(scene / "points.npz", allow_pickle=False)
    restored = None
    if resume:
        model, restored = load_model(resume)
        if "training_state" not in restored:
            raise ValueError("Checkpoint has no optimizer state; it is inference-only")
        model.spatial_lr_scale = restored["training_state"]["spatial_lr_scale"]
        model_args = SimpleNamespace(**restored["model_args"])
    else:
        model = GaussianModel(model_args.sh_degree, model_args)
        model.create_from_pcd(BasicPointCloud(cloud["xyz"], cloud["colors"], np.zeros_like(cloud["xyz"])), manifest["extent"])
    if args.mode == "gs" and not resume:
        # Match the initial effective volume opacity of RT: 0.1 occupancy * 0.5 opacity.
        with torch.no_grad():
            model._occupancy.fill_(math.log(.05 / .95))
    model.training_setup(opt)
    renderer = render_gs if args.mode == "gs" else render
    pipe = SimpleNamespace(init_stage=True, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    last_reset = -100000
    start_step, elapsed_before = 1, 0.
    if restored:
        state = restored["training_state"]
        model.optimizer.load_state_dict(state["optimizer"])
        for name, value in state["buffers"].items():
            setattr(model, name, value.cuda())
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        last_reset, elapsed_before = state["last_reset"], state["elapsed_seconds"]
        start_step = restored["step"] + 1
        if restored["step"] > stop_step:
            raise ValueError("Checkpoint is already beyond the requested stop step")
        discarded = trim_resume_log(out, restored["step"])
        events_path = out / "resume_events.json"
        events = read_json(events_path) if events_path.exists() else []
        events.append(dict(checkpoint_step=restored["step"], stop_at=stop_step,
                           discarded_log_rows=discarded, unix_time=time.time()))
        write_json(events_path, events)
    start = time.perf_counter()
    if restored and getattr(args, "preview_on_resume", False):
        save_preview(model, scene, manifest, args.mode, out, restored["step"], config, elapsed_before)
    with (out / "losses.jsonl").open("a" if resume else "w", encoding="utf-8") as log:
        import json
        for step in range(start_step, stop_step + 1):
            model.update_learning_rate(step)
            if step % 1000 == 0:
                model.oneupSHdegree()
            camera = cameras[random.randrange(len(cameras))]
            pipe.init_stage = step < args.warmup
            package = renderer(camera, model, pipe, background)
            loss, components = loss_for_view(package, camera.original_image.cuda(),
                camera.gt_transparent_mask.cuda(), model, opt, step, args.mode, args.warmup)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}/{camera.image_name}")
            loss.backward()
            with torch.no_grad():
                if step < opt.densify_until_iter:
                    visible = package["visibility_filter"]
                    model.max_radii2D[visible] = torch.maximum(model.max_radii2D[visible], package["radii"][visible])
                    model.add_densification_stats(package["viewspace_points"], visible, step)
                    if step > opt.densify_from_iter and step % opt.densification_interval == 0:
                        size_threshold = 20 if step > opt.occupancy_reset_interval else None
                        model.densify_and_prune(opt.densify_grad_threshold, opt.occupancy_cull,
                                               manifest["extent"], size_threshold, last_reset)
                        if not len(model.get_xyz):
                            raise RuntimeError("All Gaussians pruned; check geometry and scene scale")
                    if step % opt.occupancy_reset_interval == 0:
                        model.reset_occupancy()
                        last_reset = step
                    if args.mode != "gs" and step >= opt.occupancy_reset_interval and step % opt.occupancy_reset_interval == opt.occupancy_reset_interval // 2:
                        model.reset_opacity()
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 100 == 0 or step == stop_step:
                row = dict(step=step, camera=camera.image_name, loss=float(loss.detach()),
                           points=len(model.get_xyz), elapsed_seconds=elapsed_before + time.perf_counter() - start,
                           components={k: float(v.detach()) for k, v in components.items()})
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(f"{args.mode} {step}/{stop_step}: loss={row['loss']:.6f}, points={row['points']}", flush=True)
            if step % args.save_every == 0 or step == stop_step:
                save_model(model, out / "checkpoint.pt", config, model_args, step,
                           dict(last_reset=last_reset, elapsed_seconds=elapsed_before + time.perf_counter() - start))
            if preview_every and step % preview_every == 0 and step < stop_step:
                save_preview(model, scene, manifest, args.mode, out, step, config,
                             elapsed_before + time.perf_counter() - start)
    torch.cuda.synchronize()
    write_json(out / "training_summary.json", dict(seconds=elapsed_before + time.perf_counter() - start, steps=stop_step,
               gaussians=len(model.get_xyz), peak_vram_bytes=torch.cuda.max_memory_allocated()))
    return evaluate_model(model, scene, manifest, args.mode, out / "evaluation", args.lpips)


def evaluate(args):
    activate(args.upstream)
    model, state = load_model(args.checkpoint)
    scene = args.scene or state["config"]["scene"]
    manifest = validate_scene(scene)
    if manifest["fingerprint"] != state["config"]["scene_fingerprint"]:
        raise ValueError("Scene images/calibration/masks changed since training")
    if state["step"] < state["config"]["warmup"]:
        raise ValueError("Checkpoint is still in warmup; final RT evaluation is not valid")
    return evaluate_model(model, scene, manifest, state["config"]["mode"], args.out, args.lpips)
