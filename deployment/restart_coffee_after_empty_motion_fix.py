"""Preserve the failed run before retrying; never discard existing artifacts."""
from pathlib import Path
import json
import subprocess

root=Path('/workspace/Ex4DGS/runs').resolve()
source=root/'coffee_f000_residual_rt'
archive=root/'coffee_f000_residual_rt_failed_6000'
assert source.resolve().parent==root and not source.is_symlink()
assert not archive.exists()
status=subprocess.run(['supervisorctl','status','coffee_ex4dgs'],capture_output=True,text=True).stdout
assert 'RUNNING' not in status and 'STARTING' not in status, status
progress=json.loads((source/'progress.json').read_text())
progress.update(status='failed',error='No dynamic Gaussians: prune_invisible indexed an empty temporal tensor',
    recovery='Artifacts preserved; no intermediate model checkpoint existed. Fresh run after regression test; rolling checkpoints added.')
(source/'progress.json').write_text(json.dumps(progress,indent=2))
source.rename(archive)
subprocess.run(['supervisorctl','start','coffee_ex4dgs'],check=True)
print('Preserved failed results at',archive)
