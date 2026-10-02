#!/usr/bin/env bash
set -euo pipefail
if [ ! -x /workspace/venvs/ex4dgs/bin/python ]; then uv venv /workspace/venvs/ex4dgs --python 3.10; fi
export CUDA_HOME=/workspace/cuda-11.8
export PATH="$CUDA_HOME/bin:/workspace/venvs/ex4dgs/bin:$PATH"
export CC=gcc-11 CXX=g++-11 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
uv pip install --python /workspace/venvs/ex4dgs/bin/python --no-cache torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python /workspace/venvs/ex4dgs/bin/python --no-cache 'numpy<2' 'kornia==0.7.2' joblib tqdm plyfile natsort opencv-python scipy scikit-image pillow ninja setuptools==69.5.1 wheel pip pytest
cd /workspace/Ex4DGS
uv pip install --python /workspace/venvs/ex4dgs/bin/python --no-cache --no-build-isolation ./submodules/diff_gaussian_rasterization_df ./submodules/simple-knn
python -c 'import torch; import diff_gaussian_rasterization_df; from scene.c_gaussian_model import CGaussianModel; print(torch.__version__, torch.cuda.get_device_name())'
uv pip freeze --python /workspace/venvs/ex4dgs/bin/python > /workspace/ex4dgs-environment.txt
touch /workspace/ex4dgs-setup-complete
