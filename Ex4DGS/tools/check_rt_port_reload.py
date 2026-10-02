"""Check both standalone inference branches using explicit TEST FIXTURES.

Fixtures are exported into a NEW output directory from an existing trial's
latest.pth. They do not represent validation acceptance, and this tool never
changes the original selected.pth, decision, or test metrics. Each branch is
rendered in a fresh process and checked against its already recorded test MSE.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch


ABSOLUTE_MSE_TOLERANCE = 1e-8
RELATIVE_MSE_TOLERANCE = 1e-5


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trial, source, output = args.trial.resolve(), args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Reload check output must be a new folder")
    if not source.is_dir():
        raise FileNotFoundError(source)
    required_files = ["latest.pth", "selected.pth", "decision.json", "test_metrics.json"]
    for name in required_files:
        if not (trial / name).is_file():
            raise FileNotFoundError(trial / name)
    expected_metrics = json.loads((trial / "test_metrics.json").read_text(encoding="utf-8"))
    if expected_metrics.get("split") != "test":
        raise ValueError("Reload parity must use the trial's existing final test measurements")
    reference = {record["image"]: record for record in expected_metrics["views"]}
    if not reference or len(reference) != len(expected_metrics["views"]):
        raise ValueError("Missing or duplicate images in existing test measurements")
    immutable = {name: digest(trial / name) for name in required_files}
    output.mkdir(parents=True)
    fixture_dir = output / "test_fixtures"
    fixture_dir.mkdir()
    report = {
        "status": "running", "purpose": "standalone checkpoint reload parity only",
        "fixtures_are_validation_selections": False,
        "trial": str(trial), "source": str(source),
        "mse_absolute_tolerance": ABSOLUTE_MSE_TOLERANCE,
        "mse_relative_tolerance": RELATIVE_MSE_TOLERANCE,
        "original_file_sha256": immutable, "branches": {},
    }
    try:
        # Keep fixture export on CPU so the child renderer starts with free GPU
        # memory, including for the two complete real 30k model optimizer states.
        saved = torch.load(trial / "latest.pth", map_location="cpu")
        if saved.get("format_version") != 1:
            raise ValueError("Unsupported trial checkpoint format")
        if saved["iteration"] != expected_metrics["iteration"]:
            raise ValueError("Saved model iteration and existing test measurements differ")
        if set(saved["protocol"]["test"]) != set(reference):
            raise ValueError("Stored test camera protocol and measured views differ")
        for branch, base_key in (("ex4dgs", "control"), ("rt_ex4dgs", "treatment")):
            fixture = {
                "format_version": 1,
                "test_fixture": True,
                "fixture_purpose": "checkpoint reload parity, not validation selection",
                "selected_model": branch,
                "base": saved[base_key],
                "material": saved["material"] if branch == "rt_ex4dgs" else None,
                "config": saved["config"], "baseline_config": saved["baseline_config"],
                "protocol": saved["protocol"], "iteration": saved["iteration"],
                "decision": {"selected_model": branch,
                             "reason": "checkpoint reload test fixture, not validation selection"},
            }
            torch.save(fixture, fixture_dir / f"{branch}_TEST_FIXTURE.pth")
        del fixture, saved
        gc.collect()
        renderer = Path(__file__).resolve().parents[1] / "render_rt_selected.py"
        for branch, mse_key in (("ex4dgs", "full_control_mse"), ("rt_ex4dgs", "full_rt_mse")):
            render_dir = output / f"{branch}_render"
            log_path = output / f"{branch}_reload.log"
            command = [sys.executable, str(renderer), "--checkpoint",
                       str(fixture_dir / f"{branch}_TEST_FIXTURE.pth"), "--source", str(source),
                       "--output", str(render_dir)]
            with log_path.open("wb") as log:
                completed = subprocess.run(command, cwd=renderer.parent, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"{branch} reload renderer failed; see {log_path}")
            actual = json.loads((render_dir / "metrics.json").read_text(encoding="utf-8"))
            if actual["selected_model"] != branch or actual["selection_recomputed"] is not False:
                raise AssertionError("Inference branch or no-selection contract violated")
            actual_views = {record["image"]: record for record in actual["views"]}
            if set(actual_views) != set(reference):
                raise AssertionError("Reloaded camera set differs from existing test measurements")
            results = []
            for name, reference_view in reference.items():
                expected_mse, actual_mse = float(reference_view[mse_key]), float(actual_views[name]["mse"])
                difference = abs(actual_mse - expected_mse)
                passed = (math.isfinite(actual_mse) and math.isfinite(expected_mse)
                          and difference <= ABSOLUTE_MSE_TOLERANCE
                          + RELATIVE_MSE_TOLERANCE * abs(expected_mse))
                results.append({"image": name, "expected_mse": expected_mse,
                                "reloaded_mse": actual_mse, "absolute_difference": difference,
                                "passed": passed})
            report["branches"][branch] = {"status": "passed" if all(v["passed"] for v in results) else "failed",
                                           "test_fixture": True, "views": results,
                                           "render_report": str(render_dir / "index.html")}
            if not all(v["passed"] for v in results):
                raise AssertionError(f"{branch} standalone reload MSE disagrees with existing test metrics")
        after = {name: digest(trial / name) for name in required_files}
        report["original_files_unchanged"] = immutable == after
        if not report["original_files_unchanged"]:
            raise AssertionError("Original trial artifacts changed during reload verification")
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
