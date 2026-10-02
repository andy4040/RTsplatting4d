"""GPU integration check: identical paired trial uninterrupted versus resumed.

Run with the Ex4DGS CUDA environment after building the official RT rasterizers.
The existing tiny smoke baseline supplies its native low resolution and point
count; the RT material/lighting architecture is never reduced for this check.
The harness creates a fresh output directory and never removes previous runs.
Passing means exact resume bookkeeping and numerical equivalence on the tested
validation images, not bit-identical CUDA optimization. Final parameter mismatch
against the unchanged strict tolerance is reported separately and explicitly.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint(path):
    # These checkpoints are written by the local runner, not untrusted inputs.
    return torch.load(path, map_location="cpu")


def exact_tree(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise AssertionError(f"Exact tensor mismatch: {path}")
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(f"Key mismatch: {path}")
        for key in left:
            exact_tree(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f"Length mismatch: {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            exact_tree(a, b, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"Value mismatch: {path}: {left!r} != {right!r}")


def close_tree(left, right, *, atol, rtol, path="state", stats=None):
    if stats is None:
        stats = {"tensor_count": 0, "max_abs_difference": 0., "largest_difference_path": None}
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol, equal_nan=True,
                                   msg=lambda msg: f"{path}: {msg}")
        stats["tensor_count"] += 1
        if left.is_floating_point() and left.numel():
            finite = torch.isfinite(left) & torch.isfinite(right)
            delta = float((left[finite] - right[finite]).abs().max()) if finite.any() else 0.
            if delta > stats["max_abs_difference"]:
                stats["max_abs_difference"] = delta
                stats["largest_difference_path"] = path
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(f"Key mismatch: {path}")
        for key in left:
            close_tree(left[key], right[key], atol=atol, rtol=rtol, path=f"{path}.{key}", stats=stats)
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f"Length mismatch: {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            close_tree(a, b, atol=atol, rtol=rtol, path=f"{path}[{index}]", stats=stats)
    elif isinstance(left, float):
        if not math.isclose(left, right, abs_tol=atol, rel_tol=rtol):
            raise AssertionError(f"Scalar mismatch: {path}: {left} vs {right}")
    elif left != right:
        raise AssertionError(f"Value mismatch: {path}: {left!r} != {right!r}")
    return stats


def optimizer_steps(optimizer):
    result = {}
    for index, group in enumerate(optimizer["param_groups"]):
        name = group.get("name", str(index))
        values = []
        for parameter in group["params"]:
            state = optimizer["state"].get(parameter, {})
            step = state.get("step")
            values.append(float(step) if step is not None else None)
        result[name] = values
    return result


def metric_records(history):
    """Compare unrounded MSEs; do not make a PSNR display rounding assertion."""
    result = []
    for observation in history:
        record = {"iteration": observation["iteration"], "views": []}
        for view in observation["views"]:
            keys = ("image", "split", "roi_sha256", "full_control_mse", "full_rt_mse",
                    "outside_pixels", "outside_control_mse", "outside_rt_mse")
            measured = {key: view[key] for key in keys}
            measured["regions"] = [
                {key: region[key] for key in ("id", "pixels", "control_mse", "rt_mse")}
                for region in view["regions"]]
            record["views"].append(measured)
        result.append(record)
    return result


def trace(path):
    return [{key: row[key] for key in ("iteration", "image", "background")}
            for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())]


def material_difference(left, right, *, atol, rtol):
    """Diagnostic statistics only; these never replace the strict tolerance."""
    result = {}
    for key, x in left.items():
        y = right[key]
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            continue
        if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise AssertionError(f"Invalid material tensor: {key}")
        difference = (x - y).abs()
        result[key] = {
            "elements": x.numel(),
            "mismatched_elements": int((difference > atol + rtol * y.abs()).sum()),
            "max_abs_difference": float(difference.max()) if x.numel() else 0.,
            "rms_difference": float(difference.square().mean().sqrt()) if x.numel() else 0.,
        }
    return result


def mse_differences(left_history, right_history):
    result = []
    for left, right in zip(left_history, right_history):
        for a, b in zip(left["views"], right["views"]):
            keys = ("full_control_mse", "full_rt_mse", "outside_control_mse", "outside_rt_mse")
            result.append({"iteration": left["iteration"], "image": a["image"],
                           **{key: abs(a[key] - b[key]) for key in keys}})
    return result


def command(args, baseline, source, output, config, *extra):
    return [sys.executable, str(ROOT / "run_rt_ex4dgs.py"), "--baseline-run", str(baseline),
            "--source", str(source), "--output", str(output), "--config", str(config), *extra]


def run_command(argv, log_path, record):
    start = time.monotonic()
    entry = {"command": argv, "log": str(log_path)}
    record["commands"].append(entry)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False)
    entry.update(returncode=completed.returncode, elapsed_seconds=time.monotonic() - start)
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-35:]
        raise RuntimeError(f"Runner failed ({completed.returncode}); {log_path}\n" + "\n".join(tail))


def run(args):
    output = args.output_root.resolve()
    baseline, source = args.baseline_run.resolve(), Path(args.source).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    if not (baseline / "baseline_final.pth").is_file():
        raise FileNotFoundError(baseline / "baseline_final.pth")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config.update(iterations_per_branch=3, evaluate_every=1)
    config["validation_roi"]["patch_size"] = 16
    config["purpose"] = "resume integration test only; not quality evidence"
    output.mkdir(parents=True)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {
        "status": "running", "architecture_changed": False, "steps": 3, "stop_after": 1,
        "baseline_run": str(baseline), "source": str(source), "commands": [],
        "tolerances": {"tensor_atol": args.tensor_atol, "tensor_rtol": args.tensor_rtol,
                       "mse_atol": args.mse_atol, "mse_rtol": args.mse_rtol},
    }
    report_path = output / "check.json"
    try:
        # Preview folders live under the baseline run and use output.name, so
        # include this fresh harness name to avoid reusing another check's UI.
        uninterrupted = output / f"{output.name}_uninterrupted"
        resumed = output / f"{output.name}_resumed"
        report["trial_outputs"] = {"uninterrupted": str(uninterrupted), "resumed": str(resumed)}
        run_command(command(args, baseline, source, uninterrupted, config_path),
                    output / "uninterrupted.log", report)
        run_command(command(args, baseline, source, resumed, config_path, "--stop-after", "1"),
                    output / "stopped.log", report)
        stopped = load_checkpoint(resumed / "latest.pth")
        if stopped["iteration"] != 1:
            raise AssertionError("Diagnostic interruption did not stop at the saved first step")
        if (resumed / "decision.json").exists() or (resumed / "complete.json").exists():
            raise AssertionError("Diagnostic stop must not freeze a final decision or complete the trial")
        del stopped
        run_command(command(args, baseline, source, resumed, config_path, "--resume"),
                    output / "resumed.log", report)
        a, b = load_checkpoint(uninterrupted / "latest.pth"), load_checkpoint(resumed / "latest.pth")
        if a["iteration"] != 3 or b["iteration"] != 3:
            raise AssertionError("Both trials must complete exactly three additional steps")
        # Check resume invariants and rendered measurements before tensor
        # closeness: CUDA texture atomics can perturb a few near-zero Adam
        # updates and must not hide otherwise useful diagnostic evidence.
        steps_a = {"control": optimizer_steps(a["control"][16]),
                   "treatment": optimizer_steps(a["treatment"][16]),
                   "material": optimizer_steps(a["material_optimizer"])}
        steps_b = {"control": optimizer_steps(b["control"][16]),
                   "treatment": optimizer_steps(b["treatment"][16]),
                   "material": optimizer_steps(b["material_optimizer"])}
        exact_tree(steps_a, steps_b, "optimizer_steps")
        report["optimizer_steps"] = steps_a
        exact_tree(a["schedule"], b["schedule"], "remaining_camera_schedule")
        exact_tree(a["sampler_rng"], b["sampler_rng"], "sampler_rng")
        exact_tree(a["rng"], b["rng"], "python_numpy_torch_cuda_rng")
        exact_tree(trace(uninterrupted / "loss.jsonl"), trace(resumed / "loss.jsonl"), "training_trace")
        close_tree(metric_records(a["history"]), metric_records(b["history"]),
                   atol=args.mse_atol, rtol=args.mse_rtol, path="validation_mse")
        decision_a = json.loads((uninterrupted / "decision.json").read_text())
        decision_b = json.loads((resumed / "decision.json").read_text())
        for key in ("selected_model", "accepted", "passes", "reason", "window_iterations"):
            exact_tree(decision_a[key], decision_b[key], f"decision.{key}")
        for folder in (uninterrupted, resumed):
            if not (folder / "complete.json").exists() or not (folder / "test_metrics.json").exists():
                raise AssertionError(f"Final selection/test evaluation did not complete: {folder}")
        report.update(selected_model=decision_a["selected_model"],
                      exact_rng_match=True, exact_camera_background_match=True,
                      validation_mse_match=True, decision_match=True)
        report["resume_mse_abs_differences"] = mse_differences(a["history"], b["history"])
        report["resume_material_difference"] = material_difference(
            a["material"], b["material"], atol=args.tensor_atol, rtol=args.tensor_rtol)
        if args.repeat_control:
            # A second uninterrupted process tests whether the same discrepancies
            # occur without any resume. It is diagnostic, never a reason to
            # widen the requested tensor or functional metric tolerances.
            repeated = output / f"{output.name}_uninterrupted_repeat"
            report["trial_outputs"]["uninterrupted_repeat"] = str(repeated)
            run_command(command(args, baseline, source, repeated, config_path),
                        output / "uninterrupted_repeat.log", report)
            repeated_state = load_checkpoint(repeated / "latest.pth")
            exact_tree(a["rng"], repeated_state["rng"], "repeat_rng")
            exact_tree(trace(uninterrupted / "loss.jsonl"), trace(repeated / "loss.jsonl"), "repeat_training_trace")
            repeat_material = material_difference(a["material"], repeated_state["material"],
                                                   atol=args.tensor_atol, rtol=args.tensor_rtol)
            report["same_seed_uninterrupted_repeat"] = {
                "material_difference": repeat_material,
                "mse_abs_differences": mse_differences(a["history"], repeated_state["history"]),
                "exceeds_tensor_tolerance": any(v["mismatched_elements"] for v in repeat_material.values()),
                "exact_rng_and_training_trace": True,
            }
            del repeated_state
        strict_failures = {}
        for key in ("control", "treatment", "material", "material_optimizer"):
            try:
                report[key] = close_tree(a[key], b[key], atol=args.tensor_atol,
                                         rtol=args.tensor_rtol, path=key)
            except AssertionError as exc:
                strict_failures[key] = str(exc)
        report["strict_tensor_failures"] = strict_failures
        report["parameter_match"] = not strict_failures
        report["resume_assertion_scope"] = (
            "Exact RNG, sampler/background trace and optimizer step counts; raw validation MSE "
            "within predeclared tolerances and unchanged final selection. This does not assert "
            "bit-identical CUDA training or exact final parameter equality. Any stricter parameter "
            "mismatch remains in strict_tensor_failures and material difference statistics."
        )
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(report_path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, default=ROOT / "checks/smoke_recovery_01")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/coffee_rt_port.json")
    parser.add_argument("--tensor-atol", type=float, default=2e-5)
    parser.add_argument("--tensor-rtol", type=float, default=2e-4)
    parser.add_argument("--mse-atol", type=float, default=1e-6)
    parser.add_argument("--mse-rtol", type=float, default=1e-4)
    parser.add_argument("--repeat-control", action="store_true",
                        help="Run a second same-seed uninterrupted trial to diagnose CUDA noise; never changes tolerances")
    args = parser.parse_args()
    for key in ("tensor_atol", "tensor_rtol", "mse_atol", "mse_rtol"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"{key} must be finite and nonnegative")
    run(args)


if __name__ == "__main__":
    main()
