"""Fixed-pose COLMAP initialization from synchronized N3DV images.

Only train images enter feature extraction/matching/triangulation. Held-out
poses are added afterwards with no observations. Does not modify RT-Splatting.
LLFF convention: C2W OpenCV = [LLFF column 1, column 0, -column 2, center].
References: COLMAP FAQ (known poses), 4DGaussians/scripts/llff2colmap.py.
Run in a separate preprocessing environment with numpy, pillow, plyfile, plotly.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement


def run(*args):
    command = list(map(str, args))
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, check=True, env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    work, final = data / 'initialization', data / 'sparse/0'
    if work.exists() or final.exists():
        raise FileExistsError('Refusing to overwrite an existing initialization or sparse model')
    split = json.loads((data / 'approved_split.json').read_text())
    names = sorted(p.stem for p in (data / 'images').glob('*.png'))
    assert split['approved'] and split['frame_index'] == 0
    assert names == split['camera_order']
    assert names[::8] == split['test']
    assert [n for i, n in enumerate(names) if i % 8] == split['train']
    poses = np.load(data / 'poses_bounds.npy', allow_pickle=False)
    assert poses.shape == (len(names), 17) and np.isfinite(poses).all()
    poses = poses[:, :15].reshape(-1, 3, 5)
    assert np.allclose(poses[:, :, 4], poses[0, :, 4])
    h, w, focal = poses[0, :, 4]
    h, w = int(h), int(w)
    for name in names:
        with Image.open(data / 'images' / (name + '.png')) as im:
            assert im.size == (w, h)

    # Load the unchanged official parser without importing the training package.
    spec = importlib.util.spec_from_file_location('official_colmap_loader', Path(__file__).resolve().parents[1] / 'scene/colmap_loader.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    calibration = {}
    for name, p in zip(names, poses):
        c2w = np.stack([p[:, 1], p[:, 0], -p[:, 2]], axis=1)
        assert np.allclose(c2w.T @ c2w, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(c2w), 1, atol=1e-6)
        r = c2w.T
        t = -r @ p[:, 3]
        q = loader.rotmat2qvec(r)
        assert np.allclose(loader.qvec2rotmat(q), r, atol=1e-7)
        calibration[name] = (q, t, r, p[:, 3])

    work.mkdir()
    train_images = work / 'train_images'
    train_images.mkdir()
    for name in split['train']:
        (train_images / (name + '.png')).symlink_to(data / 'images' / (name + '.png'))
    db = work / 'database.db'
    run('colmap', 'feature_extractor', '--database_path', db, '--image_path', train_images,
        '--ImageReader.camera_model', 'PINHOLE', '--ImageReader.single_camera', '1',
        '--ImageReader.camera_params', f'{focal},{focal},{w/2},{h/2}',
        '--SiftExtraction.use_gpu', '0', '--SiftExtraction.num_threads', '8')
    with sqlite3.connect(db) as conn:
        rows = conn.execute('SELECT image_id, name, camera_id FROM images ORDER BY image_id').fetchall()
        assert {Path(row[1]).stem for row in rows} == set(split['train'])
        cameras = conn.execute('SELECT camera_id, model, width, height, params FROM cameras').fetchall()
        assert len(cameras) == 1
        cid, model, dw, dh, params = cameras[0]
        assert model == 1 and (dw, dh) == (w, h)
        assert np.allclose(np.frombuffer(params, dtype=np.float64), [focal, focal, w/2, h/2])
    known, triangulated = work / 'known_train_poses', work / 'triangulated'
    known.mkdir(); triangulated.mkdir()
    camera_line = f'{cid} PINHOLE {w} {h} {focal:.17g} {focal:.17g} {w/2:.17g} {h/2:.17g}\n'
    (known / 'cameras.txt').write_text(camera_line)

    def pose_line(iid, filename):
        q, t, _, _ = calibration[Path(filename).stem]
        return f'{iid} ' + ' '.join(f'{v:.17g}' for v in [*q, *t]) + f' {cid} {filename}\n\n'

    (known / 'images.txt').write_text(''.join(pose_line(i, n) for i, n, _ in rows))
    (known / 'points3D.txt').write_text('')
    run('colmap', 'exhaustive_matcher', '--database_path', db,
        '--SiftMatching.use_gpu', '0', '--SiftMatching.num_threads', '8')
    run('colmap', 'point_triangulator', '--database_path', db, '--image_path', train_images,
        '--input_path', known, '--output_path', triangulated,
        '--Mapper.fix_existing_images', '1', '--Mapper.ba_refine_focal_length', '0',
        '--Mapper.ba_refine_principal_point', '0', '--Mapper.ba_refine_extra_params', '0',
        '--Mapper.num_threads', '8')
    train_cams = loader.read_extrinsics_binary(triangulated / 'images.bin')
    assert len(train_cams) == len(split['train'])
    for cam in train_cams.values():
        _, t, r, _ = calibration[Path(cam.name).stem]
        assert np.allclose(loader.qvec2rotmat(cam.qvec), r, atol=1e-7)
        assert np.allclose(cam.tvec, t, atol=1e-7)
    intr = loader.read_intrinsics_binary(triangulated / 'cameras.bin')[cid]
    assert np.allclose(intr.params, [focal, focal, w/2, h/2], atol=1e-7)

    final.mkdir(parents=True)
    run('colmap', 'model_converter', '--input_path', triangulated, '--output_path', final, '--output_type', 'TXT')
    next_id = max(train_cams) + 1
    with (final / 'images.txt').open('a') as f:
        for offset, name in enumerate(split['test']):
            f.write(pose_line(next_id + offset, name + '.png'))
    run('colmap', 'model_converter', '--input_path', final, '--output_path', final, '--output_type', 'BIN')
    all_cams = loader.read_extrinsics_binary(final / 'images.bin')
    assert sorted(Path(c.name).stem for c in all_cams.values()) == names
    for c in all_cams.values():
        if Path(c.name).stem in split['test']:
            assert len(c.point3D_ids) == 0

    points = {}
    train_ids = set(train_cams)
    for line in (final / 'points3D.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        assert set(map(int, fields[8::2])) <= train_ids
        points[int(fields[0])] = (np.array(fields[1:4], float), np.array(fields[4:7], np.uint8), float(fields[7]), len(fields[8::2]))
    assert points, 'No points reconstructed; inspect calibration/matches before training'
    xyz = np.array([p[0] for p in points.values()])
    rgb = np.array([p[1] for p in points.values()])
    vertex = np.zeros(len(points), dtype=[(n, 'f4') for n in ['x','y','z','nx','ny','nz']] + [(n,'u1') for n in ['red','green','blue']])
    for idx, name in enumerate(['x', 'y', 'z']): vertex[name] = xyz[:, idx]
    for idx, name in enumerate(['red', 'green', 'blue']): vertex[name] = rgb[:, idx]
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(final / 'points3D.ply')

    report = work / 'report'; report.mkdir()
    metrics = {'frame_index': 0, 'resolution': [w, h], 'point_count': len(points),
               'train': split['train'], 'test': split['test'], 'fixed_poses': True,
               'fixed_intrinsics': True, 'test_observations': 0, 'views': {},
               'point_error_mean_px': float(np.mean([p[2] for p in points.values()])),
               'mean_track_length': float(np.mean([p[3] for p in points.values()]))}
    residuals = []
    cards = []
    for cam in sorted(all_cams.values(), key=lambda c: c.name):
        name = Path(cam.name).stem
        r = loader.qvec2rotmat(cam.qvec)
        pc = xyz @ r.T + cam.tvec
        uv = pc[:, :2] / pc[:, 2:3] * focal + [w/2, h/2]
        visible = (pc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        observed, positive, errors = 0, 0, []
        for xy, pid in zip(cam.xys, cam.point3D_ids):
            if pid < 0: continue
            cp = r @ points[int(pid)][0] + cam.tvec
            pred = cp[:2] / cp[2] * focal + [w/2, h/2]
            errors.append(float(np.linalg.norm(pred - xy)))
            observed += 1; positive += int(cp[2] > 0)
        residuals.extend(errors)
        metrics['views'][name] = {'split': 'test' if name in split['test'] else 'train',
            'observed_points': observed, 'positive_depth_observations': positive,
            'projected_in_frame_points': int(visible.sum()),
            'mean_reprojection_error_px': float(np.mean(errors)) if errors else None}
        assert positive == observed
        with Image.open(data / 'images' / cam.name) as photo:
            photo = photo.convert('RGB'); photo.thumbnail((1352,1014))
            photo.save(report / f'{name}_photo.jpg', quality=92)
            overlay = photo.copy(); draw = ImageDraw.Draw(overlay)
            low, high = np.percentile(pc[visible, 2], [5,95]) if visible.any() else (0,1)
            for xy, depth in zip(uv[visible], pc[visible,2]):
                x,y = xy * photo.width / w
                a = float(np.clip((depth-low)/max(high-low,1e-8), 0, 1))
                draw.ellipse((x-1,y-1,x+1,y+1), fill=(int(255*a),int(230*(1-a)),255))
            overlay.save(report / f'{name}_projection.jpg', quality=94)
        cards.append(f'<section><h2>{name} ({metrics["views"][name]["split"]})</h2><p>Matched observations: {observed}; projected points: {int(visible.sum())}. Cyan: near, magenta: far (per-view depth scale).</p><div class="pair"><img src="{name}_photo.jpg"><img src="{name}_projection.jpg"></div></section>')
    metrics['observation_error_median_px'] = float(np.median(residuals))
    metrics['observation_error_p95_px'] = float(np.percentile(residuals,95))
    (report / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    import plotly.graph_objects as go
    cloud = go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',name='Train-only sparse points',marker=dict(size=1.5,color=rgb/255,opacity=0.8))
    fig = go.Figure(cloud)
    centers = np.array([v[3] for v in calibration.values()])
    length = np.linalg.norm(centers-centers.mean(0),axis=1).max() * .2
    for name, (_, _, r, center) in calibration.items():
        color = '#e64b35' if name in split['test'] else '#0072b2'
        tip = center + r.T[:,2] * length
        fig.add_trace(go.Scatter3d(x=[center[0],tip[0]],y=[center[1],tip[1]],z=[center[2],tip[2]],mode='lines+markers+text',text=[name,''],name=name,line=dict(color=color,width=5),marker=dict(size=4,color=color)))
    fig.update_layout(title='N3DV original world coordinates: blue=train; red=test; lines=forward',scene=dict(aspectmode='data'),height=800)
    fig.write_html(report / 'cameras_points.html', include_plotlyjs=True)
    html = '<!doctype html><meta charset="utf-8"><title>Coffee Martini: initialization check</title><style>body{font:16px system-ui;margin:24px;background:#fafafa;color:#222}.pair{display:flex;gap:8px}.pair img{width:49%;object-fit:contain}section{margin:32px 0}iframe{width:100%;height:820px;border:0}pre{white-space:pre-wrap}</style><h1>Coffee Martini frame 0 — camera / sparse point inspection</h1><p>Fixed N3DV calibration. Train 15 views only; test cam00/cam09/cam19 added without observations. No RT training or SAM2 masks. Projections show sparse points, not a rendered prediction; test projections are not test accuracy. Glass/reflection can violate triangulation assumptions. A point on a window pixel alone does not establish its true depth behind glass.</p><pre>' + json.dumps({k:v for k,v in metrics.items() if k != 'views'}, indent=2) + '</pre><a href="cameras_points.html">Open interactive 3D camera / point viewer</a><iframe src="cameras_points.html"></iframe>' + ''.join(cards)
    (report / 'index.html').write_text(html, encoding='utf-8')
    (work / 'complete.json').write_text(json.dumps({'status':'complete','points':len(points)}, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
