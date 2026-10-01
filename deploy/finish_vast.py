"""Create visual outputs and a downloadable backup after the final test completes."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

root=Path('/workspace/RTsplatting4d')
run=root/'runs/coffee_rt_100k'
while not (run/'evaluation/metrics.json').exists():
    for preview in sorted((run/'previews').glob('step_*')):
        if (preview/'training_summary.json').exists() and not (preview/'report/index.html').exists():
            subprocess.run(['/venv/main/bin/python',str(root/'deploy/report_results.py'),'--run',str(preview)],check=True)
            step=json.loads((preview/'training_summary.json').read_text())['steps']
            latest=run/'preview_latest.json'
            temp=latest.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(step=step,report=str(preview/'report/index.html'))))
            temp.replace(latest)
    time.sleep(10)
summary=json.loads((run/'training_summary.json').read_text())
config=json.loads((run/'config.json').read_text())
if summary['steps']!=config.get('stop_at',config['iterations']):
    raise RuntimeError('The requested training iterations have not completed')
subprocess.run(['/venv/main/bin/python',str(root/'deploy/report_results.py'),'--run',str(run)],check=True)
bundle=Path('/workspace/coffee_rt_results.tar.gz')
subprocess.run(['tar','-czf',str(bundle),'--exclude=__pycache__','--exclude=*.pyc','-C',str(root),
    'runs/coffee_rt_100k','data/coffee_f000','rtstatic','deploy','README.md','RUN_COFFEE.md','upstream.json'],check=True)
digest=hashlib.sha256()
with bundle.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):
        digest.update(chunk)
result=dict(bundle=str(bundle),bytes=bundle.stat().st_size,sha256=digest.hexdigest(),steps=summary['steps'])
(root/'runs/final_artifacts.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
