"""Package existing approved frame PNGs and read calibration from original ZIP."""
from pathlib import Path
import hashlib
import tarfile
import zipfile
import io

root = Path(__file__).resolve().parents[1]
data = root / 'RTsplatting4d/data/coffee_martini_f000'
source = Path('C:/Users/jsshi/OneDrive/바탕 화면/coffee_martini.zip')
target = root / 'deployment/coffee_martini_f000.tar'
assert not target.exists(), target
with zipfile.ZipFile(source) as z:
    entries = [n for n in z.namelist() if n.endswith('/poses_bounds.npy')]
    assert len(entries) == 1, entries
    poses = z.read(entries[0])
with tarfile.open(target, 'w') as t:
    images = sorted((data / 'images').glob('*.png'))
    assert len(images) == 18
    for path in images:
        t.add(path, arcname='coffee_martini_f000/images/' + path.name)
    t.add(data / 'approved_split.json', arcname='coffee_martini_f000/approved_split.json')
    entry = tarfile.TarInfo('coffee_martini_f000/poses_bounds.npy')
    entry.size = len(poses)
    t.addfile(entry, io.BytesIO(poses))
print(target, target.stat().st_size, hashlib.sha256(target.read_bytes()).hexdigest())
