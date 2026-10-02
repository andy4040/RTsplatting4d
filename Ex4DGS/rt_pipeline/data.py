import json
from pathlib import Path

from PIL import Image
import numpy as np

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from scene.dataset_readers import CameraInfo2, SceneInfo, getNerfppNorm, fetchPly
from utils.graphics_utils import focal2fov


def read_approved_n3dv(args):
    """Preserve approved 15/3 camera split and supplied calibration, without masks.

    Existing frame-zero folder is accepted directly. Optional frames.json has
    [{"camera":"cam01", "frame":0, "path":"frames/cam01/000000.png"}, ...].
    All cameras must have exactly the same frame IDs; timestamps are frame IDs.
    """
    root = Path(args.source_path)
    split = json.loads((root/'approved_split.json').read_text())
    assert split['approved']
    assert set(split['train']).isdisjoint(split['test'])
    assert set(split['camera_order']) == set(split['train']) | set(split['test'])
    extrinsics = read_extrinsics_binary(root/'sparse/0/images.bin')
    intrinsics = read_intrinsics_binary(root/'sparse/0/cameras.bin')
    by_name = {Path(c.name).stem:c for c in extrinsics.values()}
    assert sorted(by_name) == split['camera_order']
    manifest = root/'frames.json'
    entries = json.loads(manifest.read_text()) if manifest.exists() else [
        {'camera':n,'frame':0,'path':f'images/{n}.png'} for n in split['camera_order']]
    if args.end_timestamp >= 0:
        entries = [e for e in entries if e['frame'] <= args.end_timestamp]
    entries = [e for e in entries if e['frame'] >= args.start_timestamp]
    pairs = {(e['camera'],e['frame']) for e in entries}
    times = sorted({e['frame'] for e in entries})
    assert times and len(pairs) == len(entries)
    assert pairs == {(n,t) for n in split['camera_order'] for t in times}
    train, test = [], []
    for e in sorted(entries, key=lambda e:(e['frame'],e['camera'])):
        c = by_name[e['camera']]; intr = intrinsics[c.camera_id]
        assert intr.model in ['PINHOLE','SIMPLE_PINHOLE']
        fx, fy = intr.params[:2] if intr.model == 'PINHOLE' else [intr.params[0]]*2
        path = (root/e['path']).resolve()
        assert path.is_relative_to(root.resolve()), 'Image escapes dataset root'
        with Image.open(path) as im:
            w,h = im.size
        # Uniform resizing preserves FoV; reject changed aspect ratios.
        assert np.isclose(w/h, intr.width/intr.height, atol=1e-6)
        cam = CameraInfo2(uid=split['camera_order'].index(e['camera']),
            R=qvec2rotmat(c.qvec).T, T=c.tvec,
            FovX=focal2fov(fx,intr.width), FovY=focal2fov(fy,intr.height),
            image_path=str(path), image_name=f'{e["camera"]}_f{e["frame"]:06d}',
            width=w,height=h,near=args.near,far=args.far,timestamp=e['frame'],
            pose=None,hpdirecitons=None,cxr=0.,cyr=0.)
        (train if e['camera'] in split['train'] else test).append(cam)
    args.duration = max(times)+1
    ply = root/'sparse/0/points3D.ply'
    print(f'Approved split: {len(train)} train / {len(test)} test images; frames={times}')
    return SceneInfo(fetchPly(ply),train,test,getNerfppNorm(train),str(ply))
