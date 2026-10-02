"""Render the selected surface/control model; never rerun candidate selection."""
import argparse
from pathlib import Path
import torch
from run_surface_hypotheses import load_baseline
from gaussian_renderer import render
from rt_pipeline.surfaces import GlassSurfaces
from rt_pipeline.report import write_preview


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--baseline-run', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Refusing to overwrite render output')
    state = torch.load(args.trial/'selected.pth')
    scene, options, opt, pipe, _, _, _ = load_baseline(args.baseline_run, args.source, args.output)
    scene.gaussians.restore(state['background'], opt)
    bg = torch.tensor([1., 1., 1.] if options.white_background else [0., 0., 0.], device='cuda')
    if state['selected_model'] == 'surface':
        surface = GlassSurfaces(state['surface_seeds'])
        surface.load_state_dict(state['surface']); surface.eval()
        render_function = lambda camera:surface(camera, scene.gaussians, pipe, bg, options.near, options.far)
    else:
        render_function = lambda camera:render(camera, scene.gaussians, pipe, bg, near=options.near, far=options.far)
    cameras = [c for c in scene.test_cameras[1.0] if c.image_name.split('_')[0] in state['trial_config']['test_cameras']]
    write_preview(args.output/'previews', 'selected_reload_test', cameras, render_function,
                  {'stage':'selected_reload', 'split':'test', 'selected_model':state['selected_model']})


if __name__ == '__main__':
    main()
