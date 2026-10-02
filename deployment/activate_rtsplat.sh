# Source this file before running the unchanged official code.
source /workspace/venvs/rtsplat/bin/activate
export CUDA_HOME=/workspace/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export CC=gcc-11 CXX=g++-11 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
