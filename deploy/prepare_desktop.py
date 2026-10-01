"""Read user's desktop ZIP; export only frame zero without altering the source."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile
import zipfile

import cv2
from PIL import Image

project = Path(__file__).resolve().parents[1]
source = Path.home() / 'OneDrive' / '바탕 화면' / 'coffee_martini.zip'
output = project / 'runs' / 'desktop_input'
output.mkdir(parents=True, exist_ok=False)
dataset = output / 'coffee_martini'
dataset.mkdir()
with zipfile.ZipFile(source) as archive:
    members = sorted(x for x in archive.namelist() if x.endswith('.mp4'))
    for member in members:
        name = Path(member).stem
        temp = output / (name + '.mp4')
        with archive.open(member) as stream, temp.open('wb') as dest:
            shutil.copyfileobj(stream, dest)
        cap = cv2.VideoCapture(str(temp))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(member)
        folder = dataset / name / 'images'
        folder.mkdir(parents=True)
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(folder / '0000.png')
        temp.unlink()
        print(name, frame.shape, flush=True)
    pose_member = next(x for x in archive.namelist() if x.endswith('poses_bounds.npy'))
    (dataset / 'poses_bounds.npy').write_bytes(archive.read(pose_member))
digest = hashlib.sha256()
with source.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
record = dict(source=str(source), source_bytes=source.stat().st_size, source_sha256=digest.hexdigest(), frame=0,
              cameras=[Path(x).stem for x in members], source_modified=False)
(dataset / 'source.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
with tarfile.open(output / 'coffee_frame0.tar.gz', 'w:gz', compresslevel=1) as tar:
    tar.add(dataset, arcname='coffee_martini')
print(json.dumps(record, indent=2), flush=True)
