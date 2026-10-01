"""Environment preflight. Does not claim a CUDA forward/backward smoke test."""
import importlib
import json
import shutil
import sys


def main():
    report = dict(python=sys.version, nvcc=shutil.which("nvcc"), modules={})
    for module in ["numpy", "cv2", "PIL", "torch", "simple_knn._C", "diff_surfel_anych", "nvdiffrast.torch"]:
        try:
            obj = importlib.import_module(module)
            report["modules"][module] = dict(ok=True, version=getattr(obj, "__version__", None))
        except Exception as exc:
            report["modules"][module] = dict(ok=False, error=str(exc))
    if report["modules"]["torch"]["ok"]:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name()
    ready = all(v["ok"] for v in report["modules"].values()) and report.get("cuda_available", False)
    report["ready_for_gpu_smoke_test"] = ready
    print(json.dumps(report, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
