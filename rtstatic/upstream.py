"""Pinned upstream loading; CUDA dependencies are only imported for training."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = ROOT / "third_party" / "RT-Splatting"


def verify_checkout(path):
    path = Path(path).resolve()
    expected = json.loads((ROOT / "upstream.json").read_text(encoding="utf-8"))["commit"]
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Upstream mismatch: expected {expected}, got {actual}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise RuntimeError("Official checkout has modifications; use a clean pinned checkout")
    return actual


def activate(path):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requires an NVIDIA GPU and CUDA-enabled PyTorch; CPU training is not supported")
    commit = verify_checkout(path)
    sys.path.insert(0, str(Path(path).resolve()))
    for name in ["simple_knn._C", "diff_surfel_anych", "nvdiffrast.torch"]:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise RuntimeError(f"Missing native dependency {name}; run bootstrap.py --install in a CUDA environment") from exc
    return commit
