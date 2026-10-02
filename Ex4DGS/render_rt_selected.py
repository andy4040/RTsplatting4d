"""Render a standalone selected.pth without training or changing its decision.

Example:
  python render_rt_selected.py --checkpoint runs/trial/selected.pth \
      --source /workspace/data/coffee_martini_f000 --output runs/selected_preview

The source dataset supplies cameras and target photos; the baseline run and its
baseline_final.pth are not required. --cameras cam00,cam09 overrides the stored
final-test camera list for inspection only, never for a new selection decision.
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render as ex_render
from rt_pipeline.report import load_image, save_image
from rt_port.model import RTMaterialModel
from rt_port.renderer import render as rt_render
from scene import Scene, getmodel


def validate_checkpoint(payload):
    required = {"format_version", "selected_model", "base", "material", "config",
                "baseline_config", "protocol", "decision", "iteration"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Expected a complete RT-port selected.pth checkpoint")
    if payload["format_version"] != 1:
        raise ValueError(f"Unsupported selected checkpoint version: {payload['format_version']}")
    if payload["selected_model"] not in {"ex4dgs", "rt_ex4dgs"}:
        raise ValueError("Unknown selected model")
    if payload["decision"].get("selected_model") != payload["selected_model"]:
        raise ValueError("Stored selection and decision disagree")
    if not isinstance(payload["base"], (tuple, list)) or len(payload["base"]) != 40:
        raise ValueError("Checkpoint does not contain the complete Ex4DGS capture tuple")
    if payload["selected_model"] == "rt_ex4dgs":
        if not isinstance(payload["material"], dict):
            raise ValueError("Selected RT model is missing its material/lighting state")
    elif payload["material"] is not None:
        raise ValueError("An Ex4DGS-only selection must not contain active RT materials")
    if not isinstance(payload["baseline_config"], dict) or "ex4dgs" not in payload["baseline_config"]:
        raise ValueError("Missing baseline configuration needed to reconstruct the model")
    if not payload["protocol"].get("test"):
        raise ValueError("Missing final-test camera list in the stored protocol")


def make_args(baseline_config, source, output):
    parser = argparse.ArgumentParser(add_help=False)
    ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    args = parser.parse_args([])
    known = vars(args)
    unknown = set(baseline_config["ex4dgs"]) - set(known)
    if unknown:
        raise ValueError(f"Unrecognized stored Ex4DGS options: {sorted(unknown)}")
    for key, value in baseline_config["ex4dgs"].items():
        setattr(args, key, value)
    args.source_path = str(source)
    args.model_path = str(output / "scene_metadata")
    args.loader = "residual_n3dv"
    Path(args.model_path).mkdir()
    return args, optimization.extract(args), pipeline.extract(args)


def choose_cameras(scene, protocol, override=None):
    cameras = scene.train_cameras[1.0] + scene.test_cameras[1.0]
    by_name = {camera.image_name: camera for camera in cameras}
    if len(by_name) != len(cameras):
        raise ValueError("Duplicate image names in supplied dataset")
    # Relocation is allowed, changing the trained camera split is not implicit.
    for split in ("train", "validation", "test"):
        missing = set(protocol.get(split, [])) - set(by_name)
        if missing:
            raise ValueError(f"Dataset is missing saved {split} images: {sorted(missing)}")
    if override is None:
        return [by_name[name] for name in protocol["test"]]
    requested = [token.strip() for token in override.split(",")]
    if not requested or any(not token for token in requested) or len(set(requested)) != len(requested):
        raise ValueError("--cameras must be a nonempty comma-separated list without duplicates")
    selected = []
    for token in requested:
        matches = [camera for camera in cameras if camera.image_name == token
                   or camera.image_name.split("_", 1)[0] == token]
        if not matches:
            raise ValueError(f"Camera not found in dataset: {token}")
        selected.extend(matches)
    if len({camera.image_name for camera in selected}) != len(selected):
        raise ValueError("--cameras selects the same image more than once")
    return sorted(selected, key=lambda camera: (camera.timestamp, camera.image_name))


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def run(cli):
    checkpoint = Path(cli.checkpoint).resolve()
    source = Path(cli.source).resolve()
    output = Path(cli.output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError("Output must be a new folder; existing artifacts are never overwritten")
    # Checkpoints are executable Python pickle artifacts: load only trusted own
    # experiment checkpoints. Explicit CUDA relocation supports copied files.
    payload = torch.load(checkpoint, map_location="cuda")
    validate_checkpoint(payload)
    output.mkdir(parents=True)
    args, optimization, pipe = make_args(payload["baseline_config"], source, output)
    base = getmodel()(args.sh_degree, args.start_duration, args.time_interval, args.time_pad,
                      interp_type=args.interp_type, rot_interp_type=args.rot_interp_type,
                      time_pad_type=args.time_pad_type, var_pad=args.var_pad,
                      kernel_size=args.kernel_size)
    scene = Scene(args, base, shuffle=False)
    base.restore(payload["base"], optimization)
    cameras = choose_cameras(scene, payload["protocol"], cli.cameras)
    pipe.depth_ratio = 0.
    background = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device="cuda")
    is_rt = payload["selected_model"] == "rt_ex4dgs"
    material = None
    if is_rt:
        material = RTMaterialModel(base, payload["config"]["material"])
        material.load_state_dict(payload["material"], strict=True)
        material.eval()
    result = {
        "format_version": 1, "checkpoint": str(checkpoint),
        "checkpoint_sha256": hash_file(checkpoint), "source": str(source),
        "selected_model": payload["selected_model"],
        "test_fixture": bool(payload.get("test_fixture", False)),
        "fixture_purpose": payload.get("fixture_purpose"),
        "continuation_iteration": payload["iteration"],
        "saved_decision": payload["decision"], "selection_recomputed": False,
        "camera_source": "explicit_inspection_override" if cli.cameras else "saved_final_test_protocol",
        "psnr_definition": "-10log10(mean RGB squared error); prediction clamped [0,1] before metric, before PNG quantization",
        "views": [],
    }
    # Release duplicate serialized material tensors before full-resolution shading.
    del payload
    rows = []
    weighted_mse, total_values = 0., 0
    for camera in cameras:
        name = camera.image_name
        if Path(name).name != name or any(character in name for character in ("/", "\\")):
            raise ValueError("Unsafe image name for result filenames")
        target = load_image(camera)
        package = (rt_render(camera, base, material, pipe, background, near=args.near, far=args.far)
                   if is_rt else ex_render(camera, base, pipe, background, near=args.near, far=args.far))
        prediction = package["render"].clamp(0, 1)
        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
            raise ValueError(f"Invalid rendered image for {name}")
        mse = float((prediction - target).square().mean().item())
        psnr = -10 * math.log10(max(mse, 1e-12))
        result["views"].append({"image": name, "timestamp": float(camera.timestamp),
                                "mse": mse, "psnr": psnr,
                                "width": int(camera.image_width), "height": int(camera.image_height)})
        weighted_mse += mse * prediction.numel()
        total_values += prediction.numel()
        images = {"gt": target, "render": prediction}
        if is_rt:
            images.update(transmission=package["final_tran"], reflection=package["final_spec"],
                          scattering=package["final_scat"], optical_opacity=package["surface_opacity"])
        for key, value in images.items():
            save_image(value, output / f"{name}_{key}.png")
        figures = "".join(f'<figure><figcaption>{html.escape(key)}</figcaption>'
                           f'<a href="{html.escape(name)}_{key}.png"><img src="{html.escape(name)}_{key}.png"></a></figure>'
                           for key in images)
        rows.append(f'<h2>{html.escape(name)}: {psnr:.3f} dB</h2><div class="row">{figures}</div>')
        del package, images, prediction, target
    if not result["views"]:
        raise ValueError("No cameras selected for rendering")
    result["mean_view_psnr"] = sum(row["psnr"] for row in result["views"]) / len(result["views"])
    result["pooled_psnr"] = -10 * math.log10(max(weighted_mse / total_values, 1e-12))
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    page = ('<!doctype html><meta charset="utf-8"><title>선택 모델 렌더링</title>'
            '<style>body{font:16px system-ui;margin:24px}.row{display:flex;flex-wrap:wrap}'
            'figure{width:45%;margin:1%}img{width:100%}pre{white-space:pre-wrap}</style>'
            f'<h1>선택 모델: {html.escape(result["selected_model"])}</h1>'
            + ('<p><strong>체크포인트 복원 검사용 자료입니다. 검증 성능으로 채택한 모델이라는 뜻이 아닙니다.</strong></p>'
               if result["test_fixture"] else '') +
            '<p>저장된 선택 모델을 불러온 결과입니다. 추가 학습이나 모델 재선택은 하지 않았습니다.</p>'
            f'<p>평균 PSNR: {result["mean_view_psnr"]:.3f} dB</p>' + "".join(rows)
            + '<details><summary>수치 및 저장된 선택 기록</summary><pre>'
            + html.escape(json.dumps(result, indent=2)) + '</pre></details>')
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", help="Comma-separated cam IDs or complete image names; default saved final-test cameras")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
