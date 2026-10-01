import argparse
import json
import subprocess
import sys
from pathlib import Path

from .data import prepare, read_json, validate_scene, write_json
from .upstream import DEFAULT_UPSTREAM


def training_flags(parser):
    parser.add_argument("--scene", required=True, help="Directory made by prepare")
    parser.add_argument("--out", required=True)
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--lpips", action="store_true", help="Use upstream VGG LPIPS loss after step 15000 and full-image LPIPS evaluation (downloads weights)")
    parser.add_argument("--env-center", type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument("--env-radius", type=float, help="Optional reflection foreground sphere in scene coordinates; default includes all Gaussians")


def compare(args):
    validate_scene(args.scene)
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    modes = ["gs", "rt"] + (["rt_no_gating"] if args.ablation else [])
    for mode in modes:
        command = [sys.executable, "-m", "rtstatic", "train", "--scene", str(Path(args.scene).resolve()),
                   "--out", str(out / mode), "--upstream", str(Path(args.upstream).resolve()), "--mode", mode]
        for key in ["iterations", "warmup", "seed", "save_every"]:
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        if args.lpips:
            command.append("--lpips")
        if args.env_radius is not None:
            command += ["--env-radius", str(args.env_radius), "--env-center", *map(str, args.env_center)]
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
    reports = {mode: read_json(out / mode / "evaluation" / "metrics.json") for mode in modes}
    if len({r["scene_fingerprint"] for r in reports.values()}) != 1:
        raise ValueError("Scenes changed between experiments; comparison is invalid")
    deltas = {}
    for name, rt in reports["rt"]["per_view"].items():
        gs = reports["gs"]["per_view"][name]
        deltas[name] = {}
        for region in ["full", "window", "opaque"]:
            deltas[name][region] = {metric: None if rt[region][metric] is None or gs[region][metric] is None
                                    else rt[region][metric] - gs[region][metric]
                                    for metric in ["psnr", "ssim", "mae"]}
        from PIL import Image
        for filename in ["prediction.png", "window_crop.png"]:
            paths = [out / mode / "evaluation" / name / filename for mode in modes]
            if all(p.exists() for p in paths):
                images = [Image.open(p).convert("RGB") for p in paths]
                board = Image.new("RGB", (sum(im.width for im in images), max(im.height for im in images)))
                x = 0
                for image in images:
                    board.paste(image, (x, 0))
                    x += image.width
                board.save(out / f"{name}_{filename}")
    summary = dict(reports=reports, delta_rt_minus_gs=deltas, image_column_order=modes,
                   interpretation="Higher PSNR/SSIM and lower MAE are better. This is one timestamp and seed; inspect window crops. Observed RGB does not establish recovery of clean transmission.")
    write_json(out / "comparison.json", summary)
    print(json.dumps(deltas, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Static N3DV reflection/transmission feasibility experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Extract one timestamp and triangulate training cameras only")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--width", type=int, default=1352)
    p.add_argument("--test-camera", default="cam00")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-points", type=int, default=50000)
    p = sub.add_parser("annotate", help="Generate a standalone browser polygon editor")
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("masks", help="Import polygons.json exported by the editor")
    p.add_argument("--scene", required=True)
    p.add_argument("--polygons", required=True)
    p = sub.add_parser("check", help="Check images, masks and split without CUDA")
    p.add_argument("--scene", required=True)
    p = sub.add_parser("train", help="Train one variant, then evaluate the held-out view")
    training_flags(p)
    p.add_argument("--mode", choices=["gs", "rt", "rt_no_gating"], default="rt")
    p.add_argument("--resume", help="Resume the same run from its checkpoint, including optimizer and RNG state")
    p.add_argument("--stop-at", type=int, help="Finish at this step while preserving the original learning-rate schedule")
    p.add_argument("--preview-every", type=int, default=0, help="Render the held-out view every N steps; zero disables previews")
    p.add_argument("--preview-on-resume", action="store_true", help="Render the restored checkpoint before resuming training")
    p = sub.add_parser("compare", help="Train GS and RT sequentially and export comparisons")
    training_flags(p)
    p.add_argument("--ablation", action="store_true", help="Also train RT without gradient gating")
    p = sub.add_parser("evaluate", help="Reload a checkpoint and reproduce held-out metrics")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scene", help="Override dataset location; content fingerprint must still match")
    p.add_argument("--out", required=True)
    p.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    p.add_argument("--lpips", action="store_true")
    args = parser.parse_args(argv)
    if getattr(args, "save_every", 1) < 1:
        parser.error("--save-every must be positive")
    if args.command == "prepare":
        result = prepare(args.data, args.out, args.frame, args.width, args.test_camera, args.seed, args.max_points)
        print(f"Prepared {len(result['cameras'])} cameras and {result['points']} points; annotate masks next.")
    elif args.command == "annotate":
        from .annotation import export_editor
        export_editor(args.scene, args.out)
        print(f"Open {args.out} in a browser. Mark visible glass, review all cameras, export polygons.json.")
    elif args.command == "masks":
        from .annotation import import_polygons
        import_polygons(args.scene, args.polygons)
        print("Saved binary masks and mask_previews. Inspect previews before training.")
    elif args.command == "check":
        manifest = validate_scene(args.scene)
        print(json.dumps(dict(status="ready", fingerprint=manifest["fingerprint"], points=manifest["points"]), indent=2))
    elif args.command == "train":
        from .engine import train
        train(args)
    elif args.command == "evaluate":
        from .engine import evaluate
        evaluate(args)
    elif args.command == "compare":
        compare(args)


if __name__ == "__main__":
    main()
