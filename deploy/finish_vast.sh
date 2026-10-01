#!/bin/bash
set -euo pipefail
cd /workspace/RTsplatting4d
export MPLBACKEND=Agg
exec /venv/main/bin/python -u deploy/finish_vast.py
