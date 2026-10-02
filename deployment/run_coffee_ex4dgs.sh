#!/usr/bin/env bash
set -euo pipefail
source /workspace/venvs/ex4dgs/bin/activate
export CUDA_HOME=/workspace/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CC=gcc-11 CXX=g++-11 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
cd /workspace/Ex4DGS
exec python -u run_residual_rt.py --config configs/coffee_residual_rt_f000.json --source /workspace/RTsplatting4d/data/coffee_martini_f000 --output /workspace/Ex4DGS/runs/coffee_f000_residual_rt
