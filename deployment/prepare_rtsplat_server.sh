#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends colmap gcc-11 g++-11 libegl1 libgl1-mesa-dev libegl1-mesa-dev
python3 - <<'PY'
import json, pathlib, urllib.request, tarfile
base='https://developer.download.nvidia.com/compute/cuda/redist/'
manifest=json.load(urllib.request.urlopen(base+'redistrib_11.8.0.json'))
target=pathlib.Path('/workspace/cuda-11.8'); target.mkdir(exist_ok=True)
for name in ['cuda_nvcc','cuda_cudart','cuda_cccl']:
    info=manifest[name]['linux-x86_64']
    path=pathlib.Path('/workspace')/(name+'.tar.xz')
    urllib.request.urlretrieve(base+info['relative_path'],path)
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
    with tarfile.open(path) as t:
        for m in t.getmembers():
            m.name='/'.join(m.name.split('/')[1:])
            if m.name: t.extract(m,target)
    path.unlink()
PY
export CUDA_HOME=/workspace/cuda-11.8
export PATH="$CUDA_HOME/bin:/workspace/venvs/rtsplat/bin:$PATH"
export CC=gcc-11 CXX=g++-11 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
uv pip install --python /workspace/venvs/rtsplat/bin/python --no-cache torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python /workspace/venvs/rtsplat/bin/python --no-cache -r /workspace/RTsplatting4d/requirements.txt wheel pip
uv pip install --python /workspace/venvs/rtsplat/bin/python --no-cache nvidia-cusparse-cu11 nvidia-cublas-cu11 nvidia-cusolver-cu11 nvidia-curand-cu11
python - <<'PY'
from pathlib import Path
root=Path('/workspace/venvs/rtsplat/lib/python3.10/site-packages/nvidia')
cuda=Path('/workspace/cuda-11.8')
for package in root.iterdir():
    for folder in ['include', 'lib']:
        if (package/folder).is_dir():
            for source in (package/folder).iterdir():
                dest=cuda/folder/source.name
                if not dest.exists(): dest.symlink_to(source)
PY
cd /workspace/RTsplatting4d
uv pip install --python /workspace/venvs/rtsplat/bin/python --no-cache --no-build-isolation ./submodules/simple-knn ./submodules/diff-surfel-anych
uv pip install --python /workspace/venvs/rtsplat/bin/python --no-cache --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
python -c 'import torch, simple_knn._C, diff_surfel_anych._C, nvdiffrast.torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); print(torch.ones(2,device="cuda").sum().item())'
uv pip freeze --python /workspace/venvs/rtsplat/bin/python > /workspace/installed-rtsplat.txt
touch /workspace/rtsplat-setup-complete
