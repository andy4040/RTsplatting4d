#!/bin/bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=6
export NVCC_PREPEND_FLAGS='--pre-include=cstdint,cfloat'
cd /workspace/RTsplatting4d
uv pip install --python /venv/main/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /venv/main/bin/python 'numpy==1.26.4' 'opencv-python-headless==4.11.0.86' Pillow scipy matplotlib ninja plyfile kornia tqdm packaging 'setuptools<81' pytest
/venv/main/bin/python bootstrap.py
uv pip install --python /venv/main/bin/python --no-build-isolation ./third_party/RT-Splatting/submodules/simple-knn ./third_party/RT-Splatting/submodules/diff-surfel-anych
uv pip install --python /venv/main/bin/python --no-build-isolation 'git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3'
/venv/main/bin/python doctor.py
/venv/main/bin/python -m pip freeze > /workspace/RTsplatting4d/deploy/environment-lock.txt
