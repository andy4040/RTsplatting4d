"""Fetch pinned official source; optionally install into the active environment."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--install", action="store_true", help="Install dependencies and build CUDA extensions in the active environment")
    args = p.parse_args()
    lock = json.loads((ROOT / "upstream.json").read_text())
    dest = ROOT / "third_party" / "RT-Splatting"
    if not dest.exists():
        subprocess.run(["git", "clone", lock["url"], str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", lock["commit"]], check=True)
    from rtstatic.upstream import verify_checkout
    verify_checkout(dest)
    if args.install:
        if sys.version_info[:2] != (3, 10):
            raise SystemExit("Use a separate Python 3.10 environment to match upstream dependencies.")
        if shutil.which("nvcc") is None:
            raise SystemExit("CUDA toolkit (nvcc) is required to compile the native extensions.")
        pip = [sys.executable, "-m", "pip", "install"]
        subprocess.run(pip + ["-r", str(ROOT / "requirements-cpu.txt"), "-r", str(dest / "requirements.txt")], check=True)
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA-enabled PyTorch and a visible GPU are required'"], check=True)
        for module in ["simple-knn", "diff-surfel-anych"]:
            subprocess.run(pip + ["--no-build-isolation", str(dest / "submodules" / module)], check=True)
        subprocess.run(pip + ["--no-build-isolation", "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"], check=True)
        subprocess.run([sys.executable, str(ROOT / "doctor.py")], check=True)
    print(f"Official source ready: {dest}\nCommit: {lock['commit']}")


if __name__ == "__main__":
    main()
