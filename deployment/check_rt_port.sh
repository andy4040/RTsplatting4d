#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Ex4DGS
python tools/check_rt_port_resume.py --source /workspace/RTsplatting4d/data/coffee_martini_f000 --output-root checks/rt_resume_02 --repeat-control
