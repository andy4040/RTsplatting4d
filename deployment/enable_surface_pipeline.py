"""Enable the queued handoff only after recorded end-to-end and CUDA checks pass."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/workspace/Ex4DGS')
ready = root/'surface_pipeline_ready.json'
if ready.exists():
    raise FileExistsError('A surface handoff has already been enabled')
checks = {}
for name, path in [
    ('smoke', root/'checks/surface_smoke_final/complete.json'),
    ('full_resolution', root/'checks/surface_full_resolution/result.json'),
    ('acceptance', root/'checks/surface_acceptance/result.json'),
]:
    checks[name] = json.loads(path.read_text())
assert checks['smoke']['status'] == 'complete'
assert checks['full_resolution']['status'] == 'passed'
assert checks['acceptance']['status'] == 'passed'
subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_residuals.py', 'tests/test_surface_decisions.py', '-q'], cwd=root, check=True)
subprocess.run([sys.executable, 'tools/check_surfaces_cuda.py'], cwd=root, check=True)
old_render = root/'checks/surface_reload_01/previews/selected_reload_test/cam19_f000000_render.jpg'
new_render = root/'checks/smoke_recovery_01/previews/surface_selected_final_test/cam19_f000000_render.jpg'
assert old_render.read_bytes() == new_render.read_bytes()
sources = ['run_residual_rt.py', 'run_surface_hypotheses.py', 'render_surface_hypotheses.py',
    'rt_pipeline/surfaces.py', 'rt_pipeline/surface_candidates.py', 'rt_pipeline/surface_decisions.py',
    'rt_pipeline/report.py', 'configs/coffee_surface_hypotheses.json']
manifest = {'cuda_checks_passed':True, 'smoke_passed':True, 'unit_tests_passed':12,
    'full_resolution':checks['full_resolution'], 'acceptance_path':checks['acceptance'],
    'saved_render_reload_identical':True, 'smoke_is_quality_evidence':False,
    'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
    'next':'Wait for baseline_final.pth iteration 30000, then run the independent surface trial.'}
temp = ready.with_suffix('.tmp')
temp.write_text(json.dumps(manifest, indent=2)); temp.replace(ready)
print(json.dumps(manifest, indent=2))
