"""Validation-only, whole-model selection for the Ex4DGS RT experiment.

The runner must freeze validation ROI masks *before* training either branch.
This module checks their identity (when hashes are supplied), view names, region
IDs and pixel counts across observations. It never derives masks from scores.
Regions must be disjoint and their MSEs measured in normalized [0, 1] image
values using the same image space for both models (usually the dataset's RGB
values; no additional colour-space transform is implied here).

Candidate PSNR gain is computed from pixel-pooled MSE, not averaged tile dB.
Global gain is the arithmetic mean of the per-view whole-image PSNR gains.
Selection concerns the final *whole model*, not individual region compositing.
"""

import math
from numbers import Integral, Real


DEFAULT_CONFIG = {
    "decision_observations": 3,
    "required_passes": 2,
    "min_validation_pixels": 128,
    "min_validation_views": 2,
    "min_gain_db": 0.5,
    "min_per_view_gain_db": 0.0,
    "min_global_gain_db": 0.0,
    "max_outside_mse_increase": 0.02,
    "require_roi_hash": False,
}
_EPS = 1e-12


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def psnr_gain(control_mse, rt_mse):
    """Return RT minus control PSNR in dB; zero MSE uses a 1e-12 floor."""
    control_mse = _finite_nonnegative(control_mse, "control_mse")
    rt_mse = _finite_nonnegative(rt_mse, "rt_mse")
    return 10.0 * (math.log10(max(control_mse, _EPS)) - math.log10(max(rt_mse, _EPS)))


def _psnr(mse):
    return -10.0 * math.log10(max(mse, _EPS))


def _config(config):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown selection configuration: {sorted(unknown)}")
        cfg.update(config)
    for name in ("decision_observations", "required_passes",
                 "min_validation_pixels", "min_validation_views"):
        cfg[name] = _integer(cfg[name], name, 1)
    if cfg["required_passes"] > cfg["decision_observations"]:
        raise ValueError("required_passes exceeds decision_observations")
    for name in ("min_gain_db", "min_per_view_gain_db", "min_global_gain_db",
                 "max_outside_mse_increase"):
        cfg[name] = _finite_nonnegative(cfg[name], name)
    if not isinstance(cfg["require_roi_hash"], bool):
        raise ValueError("require_roi_hash must be bool")
    return cfg


def _pooled(rows):
    pixels = sum(row["pixels"] for row in rows)
    if pixels == 0:
        return {"pixels": 0, "control_mse": None, "rt_mse": None,
                "control_psnr_db": None, "rt_psnr_db": None, "gain_db": None}
    control = math.fsum(row["pixels"] * row["control_mse"] for row in rows) / pixels
    rt = math.fsum(row["pixels"] * row["rt_mse"] for row in rows) / pixels
    return {"pixels": pixels, "control_mse": control, "rt_mse": rt,
            "control_psnr_db": _psnr(control), "rt_psnr_db": _psnr(rt),
            "gain_db": psnr_gain(control, rt)}


def _observation(observation, cfg):
    iteration = _integer(observation["iteration"], "iteration")
    identities, views, candidate_rows, outside_rows, regions_by_id = {}, [], [], [], {}
    failures = []
    for raw in observation["views"]:
        if raw.get("split") != "validation":
            raise ValueError("Model selection accepts only split='validation'; never train/test")
        image = raw["image"]
        if not isinstance(image, str) or not image or image in identities:
            raise ValueError("Each validation image must have a unique nonempty name")
        roi_hash = raw.get("roi_sha256")
        if roi_hash is not None and (not isinstance(roi_hash, str) or len(roi_hash) != 64
                                     or any(c not in "0123456789abcdefABCDEF" for c in roi_hash)):
            raise ValueError("roi_sha256 must be a 64-character hexadecimal SHA256")
        if cfg["require_roi_hash"] and roi_hash is None:
            raise ValueError("roi_sha256 is required for fixed-ROI selection")
        full_control = _finite_nonnegative(raw["full_control_mse"], "full_control_mse")
        full_rt = _finite_nonnegative(raw["full_rt_mse"], "full_rt_mse")
        region_rows, region_identity = [], {}
        for region in raw["regions"]:
            region_id = region["id"]
            if not isinstance(region_id, str) or not region_id or region_id in region_identity:
                raise ValueError("Region IDs must be unique nonempty strings within each view")
            row = {"id": region_id,
                   "pixels": _integer(region["pixels"], "region pixels"),
                   "control_mse": _finite_nonnegative(region["control_mse"], "region control_mse"),
                   "rt_mse": _finite_nonnegative(region["rt_mse"], "region rt_mse")}
            region_identity[region_id] = row["pixels"]
            region_rows.append(row)
            regions_by_id.setdefault(region_id, []).append({"image": image, **row})
        candidate = _pooled(region_rows)
        eligible = candidate["pixels"] >= cfg["min_validation_pixels"]
        if eligible:
            candidate_rows.extend(region_rows)
        outside = {
            "pixels": _integer(raw["outside_pixels"], "outside_pixels"),
            "control_mse": _finite_nonnegative(raw["outside_control_mse"], "outside_control_mse"),
            "rt_mse": _finite_nonnegative(raw["outside_rt_mse"], "outside_rt_mse"),
        }
        if outside["pixels"]:
            outside_rows.append(outside)
            outside_increase = ((outside["rt_mse"] - outside["control_mse"])
                                / max(outside["control_mse"], _EPS))
        else:
            outside_increase = None
        views.append({"image": image, "candidate": candidate, "eligible": eligible,
                      "full_control_psnr_db": _psnr(full_control),
                      "full_rt_psnr_db": _psnr(full_rt),
                      "global_gain_db": psnr_gain(full_control, full_rt),
                      "outside_relative_mse_increase": outside_increase,
                      "regions": [{"id": r["id"], **_pooled([r])} for r in region_rows]})
        identities[image] = (roi_hash, region_identity, outside["pixels"])
    eligible_views = [view for view in views if view["eligible"]]
    pooled = _pooled(candidate_rows)
    global_gain = math.fsum(view["global_gain_db"] for view in views) / len(views) if views else None
    outside_pooled = _pooled(outside_rows)
    outside_increase = ((outside_pooled["rt_mse"] - outside_pooled["control_mse"])
                        / max(outside_pooled["control_mse"], _EPS)) if outside_rows else None
    if len(eligible_views) < cfg["min_validation_views"]:
        failures.append("insufficient_validation_views_or_candidate_pixels")
    if pooled["gain_db"] is None or pooled["gain_db"] < cfg["min_gain_db"]:
        failures.append("insufficient_pooled_candidate_gain")
    if any(view["candidate"]["gain_db"] < cfg["min_per_view_gain_db"] for view in eligible_views):
        failures.append("candidate_regression_in_validation_view")
    if global_gain is None or global_gain < cfg["min_global_gain_db"]:
        failures.append("insufficient_global_mean_psnr_gain")
    if outside_increase is not None and outside_increase > cfg["max_outside_mse_increase"] + _EPS:
        failures.append("pooled_outside_mse_regression")
    if any(view["outside_relative_mse_increase"] is not None
           and view["outside_relative_mse_increase"] > cfg["max_outside_mse_increase"] + _EPS
           for view in views):
        failures.append("outside_mse_regression_in_validation_view")
    regions = []
    for region_id, rows in sorted(regions_by_id.items()):
        regions.append({"id": region_id, **_pooled(rows),
                        "views": [{"image": row["image"], **_pooled([row])} for row in rows]})
    return {
        "iteration": iteration, "passed": not failures, "failure_reasons": failures,
        "eligible_validation_views": len(eligible_views), "candidate": pooled,
        "global_mean_psnr_gain_db": global_gain,
        "outside_relative_mse_increase": outside_increase,
        "views": views, "per_region": regions,
        "roi_hashes_verified": bool(identities) and all(item[0] is not None for item in identities.values()),
    }, identities


def decide(history, config=None):
    """Choose the final whole model using a fixed trailing validation window.

    All observations must use the same pre-frozen validation ROIs. In production
    set require_roi_hash=True and hash each view's label map before training.
    At least the full decision_observations window is needed. required_passes
    observations AND the final observation must pass; an earlier best checkpoint
    cannot rescue a regressed final model. No test metrics are accepted anywhere
    in history, including observations outside the trailing window.
    """
    cfg = _config(config)
    evaluations, reference_identity = [], None
    for raw in history:
        evaluation, identity = _observation(raw, cfg)
        if evaluations and evaluation["iteration"] <= evaluations[-1]["iteration"]:
            raise ValueError("Evaluation iterations must be strictly increasing")
        if reference_identity is None:
            reference_identity = identity
        elif identity != reference_identity:
            raise ValueError("Validation views or ROI identities changed after freezing")
        evaluations.append(evaluation)
    window = evaluations[-cfg["decision_observations"]:]
    passes = sum(item["passed"] for item in window)
    sufficient = len(window) == cfg["decision_observations"]
    final_pass = bool(window) and window[-1]["passed"]
    accepted = sufficient and passes >= cfg["required_passes"] and final_pass
    if accepted:
        reason = "rt_passes_regional_global_and_outside_validation_criteria"
    elif not sufficient:
        reason = "insufficient_validation_history"
    elif not final_pass:
        reason = "final_observation_failed: " + ", ".join(window[-1]["failure_reasons"])
    else:
        reason = "insufficient_repeated_validation_passes"
    return {
        "selected_model": "rt_ex4dgs" if accepted else "ex4dgs",
        "accepted": accepted, "reason": reason, "criteria": cfg,
        "selection_scope": "whole_model", "passes": passes,
        "required_passes": cfg["required_passes"],
        "window_iterations": [item["iteration"] for item in window],
        "observations": window, "latest": window[-1] if window else None,
        "per_region": window[-1]["per_region"] if window else [],
    }


coredecide = decide
