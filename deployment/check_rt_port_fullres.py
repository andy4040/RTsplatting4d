"""Prepare and run a short real-checkpoint check, without publishing quality claims."""
import json
from pathlib import Path
import subprocess
import sys

root = Path('/workspace/Ex4DGS')
cfg = json.loads((root/'configs/coffee_rt_port.json').read_text())
cfg['iterations_per_branch'] = 3
cfg['evaluate_every'] = 1
path = root/'checks/rt_fullres_config.json'
path.write_text(json.dumps(cfg, indent=2))
output = root/'checks/rt_fullres_01'
subprocess.run([sys.executable, '-u', 'run_rt_ex4dgs.py',
    '--baseline-run', 'runs/coffee_f000_residual_rt',
    '--source', '/workspace/RTsplatting4d/data/coffee_martini_f000',
    '--config', str(path), '--output', str(output)], cwd=root, check=True)
subprocess.run([sys.executable, 'render_rt_selected.py', '--checkpoint', str(output/'selected.pth'),
    '--source', '/workspace/RTsplatting4d/data/coffee_martini_f000',
    '--output', str(root/'checks/rt_fullres_selected_reload')], cwd=root, check=True)
