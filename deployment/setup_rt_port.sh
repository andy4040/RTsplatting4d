#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/workspace/cuda-11.8
export PATH="$CUDA_HOME/bin:/workspace/venvs/ex4dgs/bin:$PATH"
export CC=gcc-11 CXX=g++-11 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
uv pip install --python /workspace/venvs/ex4dgs/bin/python --no-cache 'matplotlib<3.10'
uv pip install --python /workspace/venvs/ex4dgs/bin/python --no-cache --no-build-isolation /workspace/RTsplatting4d/submodules/diff-surfel-anych
uv pip install --python /workspace/venvs/ex4dgs/bin/python --no-cache --no-build-isolation 'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'
python -c 'import torch, diff_surfel_anych, nvdiffrast.torch; print(torch.__version__, torch.cuda.get_device_name())'
uv pip freeze --python /workspace/venvs/ex4dgs/bin/python > /workspace/rt-port-environment.txt
