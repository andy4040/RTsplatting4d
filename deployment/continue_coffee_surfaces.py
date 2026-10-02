"""Preserve the live baseline; replace only its obsolete post-baseline adapter."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

ROOT = Path('/workspace/Ex4DGS')
RUN = ROOT/'runs/coffee_f000_residual_rt'
CHECKPOINT = RUN/'baseline_final.pth'
READY = ROOT/'surface_pipeline_ready.json'


def main():
    print('Waiting for a fully readable 30000-step baseline; training remains running.', flush=True)
    while True:
        if CHECKPOINT.exists():
            try:
                state, iteration = torch.load(CHECKPOINT, map_location='cpu')
                assert iteration == 30000
                assert len(list((RUN/'residuals').glob('*_030000.npz'))) == 15
                del state
                break
            except (OSError, EOFError, RuntimeError, AssertionError):
                pass
        status = subprocess.run(['supervisorctl', 'status', 'coffee_ex4dgs'], capture_output=True, text=True).stdout
        if 'RUNNING' not in status and not CHECKPOINT.exists():
            raise RuntimeError('Baseline stopped before producing its final checkpoint: '+status)
        time.sleep(2)
    # No stop is permitted before the complete baseline checkpoint and residuals exist.
    result = subprocess.run(['supervisorctl', 'stop', 'coffee_ex4dgs'], capture_output=True, text=True)
    print(result.stdout, result.stderr, flush=True)
    status = subprocess.run(['supervisorctl', 'status', 'coffee_ex4dgs'], capture_output=True, text=True).stdout
    assert 'RUNNING' not in status, status
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    (RUN/'surface_handoff.json').write_text(json.dumps({
        'baseline_iterations': iteration, 'baseline_sha256': digest,
        'baseline_training_interrupted': False,
        'superseded_stage': 'dominant-background-point optical adapter',
        'next': 'run_surface_hypotheses.py',
    }, indent=2))
    (RUN/'progress.json').write_text(json.dumps({'stage':'surface_validation', 'status':'waiting_for_verified_implementation'}))
    print('Baseline preserved. Waiting for validated surface implementation.', flush=True)
    while not READY.exists():
        time.sleep(5)
    readiness = json.loads(READY.read_text())
    assert readiness['cuda_checks_passed'] and readiness['smoke_passed']
    os.chdir(ROOT)
    os.execv('/workspace/venvs/ex4dgs/bin/python', [
        '/workspace/venvs/ex4dgs/bin/python', '-u', 'run_surface_hypotheses.py',
        '--baseline-run', str(RUN), '--source', '/workspace/RTsplatting4d/data/coffee_martini_f000',
        '--config', 'configs/coffee_surface_hypotheses.json', '--output', str(RUN/'surface_trial'),
    ])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (RUN/'surface_handoff_error.json').write_text(json.dumps({'error':repr(exc)}))
        raise
