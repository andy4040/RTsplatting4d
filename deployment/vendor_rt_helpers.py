"""Copy pinned RT helpers verbatim; extract only its standalone encoding class."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / 'RTsplatting4d'
if not (source / 'scene/gaussian_model.py').exists():
    source = root  # Published repository keeps official RT at its root.
destination = root / 'Ex4DGS/rt_port/vendor'
destination.mkdir(parents=True, exist_ok=True)
(destination.parent / '__init__.py').write_text('"""Shared-geometry Ex4DGS / RT-Splatting port."""\n')
(destination / '__init__.py').write_text('"""Pinned, unmodified upstream helper implementations; see provenance.json."""\n')
records = []
for name in ('image_utils.py', 'point_utils.py', 'sph_utils.py', 'color_utils.py', 'loss_utils.py'):
    data = (source / 'utils' / name).read_bytes()
    (destination / name).write_bytes(data)
    records.append({'source': 'utils/' + name, 'destination': name,
                    'sha256': hashlib.sha256(data).hexdigest(), 'verbatim': True})
path = source / 'scene/gaussian_model.py'
code = path.read_text(encoding='utf-8')
node = next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name == 'SphMipEncoding')
fragment = '\n'.join(code.splitlines()[node.lineno-1:node.end_lineno]) + '\n'
(destination / 'encoding.py').write_text('import torch\nfrom torch import nn\nimport nvdiffrast.torch\n\n' + fragment, encoding='utf-8')
records.append({'source': 'scene/gaussian_model.py:SphMipEncoding', 'destination': 'encoding.py',
                'class_sha256': hashlib.sha256(fragment.encode()).hexdigest(), 'class_verbatim': True})
for name in ('LICENSE', 'LICENSE.md'):
    if (source / name).exists():
        (destination / name).write_bytes((source / name).read_bytes())
(destination / 'provenance.json').write_text(json.dumps({
    'repository': 'https://github.com/sjj118/RT-Splatting',
    'commit': '3f45b3cac4be04db9f3092234666b695991b268a', 'files': records}, indent=2) + '\n')
