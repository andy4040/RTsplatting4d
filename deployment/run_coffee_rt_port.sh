#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Ex4DGS
python -u run_rt_ex4dgs.py \
  --baseline-run runs/coffee_f000_residual_rt \
  --source /workspace/RTsplatting4d/data/coffee_martini_f000 \
  --config configs/coffee_rt_port.json \
  --output runs/coffee_f000_residual_rt/rt_port_trial_v1 "$@"
