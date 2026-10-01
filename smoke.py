"""Run real CUDA forward/backward and checkpoint round trips on a tiny fixture."""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from rtstatic.fixture import make_prepared_fixture
from rtstatic.data import write_json
from rtstatic.upstream import DEFAULT_UPSTREAM, activate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    args = parser.parse_args()
    activate(args.upstream)
    root = Path(args.out).resolve()
    if root.exists():
        raise FileExistsError(root)
    scene = make_prepared_fixture(root)
    results = {}
    for mode in ["gs", "rt", "rt_no_gating"]:
        subprocess.run([sys.executable, "-m", "rtstatic", "train", "--scene", str(scene), "--mode", mode,
            "--iterations", "3", "--warmup", "1", "--out", str(root / mode), "--upstream", args.upstream], check=True)
        subprocess.run([sys.executable, "-m", "rtstatic", "evaluate", "--checkpoint", str(root / mode / "checkpoint.pt"),
            "--out", str(root / f"{mode}_reloaded"), "--upstream", args.upstream], check=True)
        first = np.load(root / mode / "evaluation" / "cam00" / "prediction.npy")
        second = np.load(root / f"{mode}_reloaded" / "cam00" / "prediction.npy")
        error = float(np.max(np.abs(first - second)))
        if not np.isfinite(first).all() or error > 1e-5:
            raise AssertionError(f"{mode}: checkpoint reload changed render by {error}")
        results[mode] = dict(checkpoint_render_max_error=error, steps=3)
    write_json(root / "smoke_result.json", dict(status="passed", modes=results,
               scope="synthetic native CUDA execution and checkpoint parity; not N3DV quality"))
    print("CUDA smoke passed for all three modes. No N3DV quality claim.")


if __name__ == "__main__":
    main()
