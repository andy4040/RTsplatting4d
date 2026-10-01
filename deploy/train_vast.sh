#!/bin/bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=4
export MAX_JOBS=6
export NVCC_PREPEND_FLAGS='--pre-include=cstdint,cfloat'
export PYTHONUNBUFFERED=1
cd /workspace/RTsplatting4d
/venv/main/bin/python -m rtstatic train --scene data/coffee_f000 --mode rt --iterations 100000 --stop-at 30000 --warmup 1000 --seed 0 --save-every 1000 --preview-every 1000 --preview-on-resume --lpips --out runs/coffee_rt_100k --resume runs/coffee_rt_100k/checkpoint.pt
