import json
import subprocess
from pathlib import Path
root=Path('/workspace/RTsplatting4d/runs/coffee_rt_100k')
rows=[json.loads(x) for x in (root/'losses.jsonl').read_text().splitlines() if x.strip()]
last=rows[-1]
config=json.loads((root/'config.json').read_text())
total=config.get('stop_at',config['iterations'])
if len(rows)>10:
    earlier=rows[-10]
    rate=(last['step']-earlier['step'])/(last['elapsed_seconds']-earlier['elapsed_seconds'])
else:
    rate=last['step']/max(last['elapsed_seconds'],1)
report=dict(step=last['step'],total=total,loss=last['loss'],gaussians=last['points'],
            elapsed_minutes=round(last['elapsed_seconds']/60,1),iterations_per_second=round(rate,2),
            eta_minutes=round((total-last['step'])/rate/60,1),test_ready=(root/'evaluation/metrics.json').exists())
latest=root/'preview_latest.json'
report['preview_step']=json.loads(latest.read_text())['step'] if latest.exists() else None
report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader'],text=True).strip()
print(json.dumps(report),flush=True)
