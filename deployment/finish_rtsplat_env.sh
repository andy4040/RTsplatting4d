#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/workspace/cuda-11.8
export PATH="$CUDA_HOME/bin:/workspace/venvs/rtsplat/bin:$PATH"
export CC=gcc-11 CXX=g++-11 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
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
