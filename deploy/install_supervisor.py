"""Install this task's setup/training supervisor jobs on the authorized Vast host."""
from pathlib import Path
import subprocess

root = Path('/workspace/RTsplatting4d')
for name, script in [('rt-coffee-setup', 'setup_vast.sh'), ('rt-coffee-train', 'train_vast.sh'), ('rt-coffee-report', 'finish_vast.sh')]:
    wrapper = Path('/opt/supervisor-scripts') / (name + '.sh')
    wrapper.write_text('''#!/bin/bash
set -o pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/RTsplatting4d
bash deploy/''' + script + ' 2>&1 | tee -a /workspace/' + name + '.log\n')
    wrapper.chmod(0o755)
    conf = Path('/etc/supervisor/conf.d') / (name + '.conf')
    conf.write_text(f'''[program:{name}]
environment=PROC_NAME="{name}"
command={wrapper}
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=30
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
''')
subprocess.run(['supervisorctl', 'reread'], check=True)
subprocess.run(['supervisorctl', 'update'], check=True)
