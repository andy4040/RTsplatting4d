"""Monitor the authorized run and retrieve verified final artifacts to this repo."""
import hashlib
import os
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "runs/coffee_collection_status.json"
HOST = os.environ.get("VAST_HOST")
PORT = os.environ.get("VAST_PORT", "22")
if not HOST:
    raise SystemExit("Set VAST_HOST=user@host and VAST_PORT for your training server")
SSH = ["ssh", "-p", PORT, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST]
REMOTE = "/workspace/RTsplatting4d"


def update(**values):
    values["checked_unix"] = time.time()
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temp.replace(STATUS)
    print(json.dumps(values), flush=True)


def main():
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    previous_preview=0
    preview_root=ROOT / 'runs/coffee_previews'
    preview_root.mkdir(exist_ok=True)
    while True:
        try:
            result = subprocess.run(SSH + [
                f"if [ -f {REMOTE}/runs/final_artifacts.json ]; then cat {REMOTE}/runs/final_artifacts.json; "
                f"else /venv/main/bin/python {REMOTE}/deploy/status.py; supervisorctl status rt-coffee-train; fi"
            ], capture_output=True, text=True, timeout=35)
            output = result.stdout
            if '"bundle"' in output:
                artifact = json.loads(output[output.index("{"):])
                if artifact["bundle"] != "/workspace/coffee_rt_results.tar.gz" or artifact["steps"] != 30000:
                    raise ValueError("Unexpected final artifact")
                break
            line = next((line for line in output.splitlines() if line.startswith('{"step"')), None)
            if not line:
                update(state="connection_retry", message=(result.stderr + output)[-1500:])
            else:
                status = json.loads(line)
                preview_step=status.get('preview_step')
                if preview_step and previous_preview < preview_step <= 30000:
                    folder=f'step_{int(preview_step):06d}'
                    if not (preview_root / folder / 'report/index.html').exists():
                        subprocess.run(['scp','-P',PORT,'-o','BatchMode=yes','-r',
                            f'{HOST}:{REMOTE}/runs/coffee_rt_100k/previews/{folder}',
                            str(preview_root)],check=True)
                    previous_preview=preview_step
                    subprocess.run([sys.executable, str(ROOT/'deploy/preview_gallery.py')], check=True)
                if previous_preview:
                    status['local_preview']=str(preview_root/f'step_{previous_preview:06d}'/'report/index.html')
                process_ok = "RUNNING" in output or status["test_ready"]
                update(state="training" if process_ok else "needs_attention", **status)
        except (subprocess.TimeoutExpired, OSError, ValueError) as error:
            update(state="connection_retry", message=str(error))
        time.sleep(45)
    destination = ROOT / "runs/coffee_completed"
    destination.mkdir(exist_ok=True)
    archive = destination / "coffee_rt_results.tar.gz"
    update(state="downloading", artifact=artifact)
    subprocess.run(["scp", "-P", PORT, "-o", "BatchMode=yes",
                    HOST + ":" + artifact["bundle"], str(archive)], check=True)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if archive.stat().st_size != artifact["bytes"] or digest.hexdigest() != artifact["sha256"]:
        raise ValueError("Downloaded archive checksum mismatch")
    update(state="extracting", artifact=artifact)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    report = destination / "runs/coffee_rt_100k/report/index.html"
    if not report.is_file():
        raise FileNotFoundError(report)
    subprocess.run([sys.executable, str(ROOT/'deploy/preview_gallery.py')], check=True)
    update(state="complete", artifact=artifact, report=str(report), results=str(destination))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        update(state="failed", message=str(error))
        raise
