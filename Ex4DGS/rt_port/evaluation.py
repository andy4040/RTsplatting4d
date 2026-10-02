"""Frozen validation ROIs and raw-float regional metrics; never training masks.

The baseline residual selects evaluation regions once, before either branch is
trained. This is an error proxy, not glass segmentation or a recovered glass
depth. Components in different cameras are explicitly different region IDs.
Validation RGB is used only to define fixed evaluation ROIs and score results;
do not send these labels to a training loss or candidate detector.
"""

import hashlib
import math
from numbers import Integral, Real
import struct

import numpy as np
from scipy import ndimage

from rt_pipeline.hard_regions import expand_patches, patch_mse


_CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def _rgb(image, *, prediction, channel_axis):
    # No torch dependency on CPU; torch tensors are accepted through this protocol.
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3 or channel_axis not in (0, 2, -1):
        raise ValueError("RGB images must be CHW (channel_axis=0) or HWC (channel_axis=-1)")
    if array.shape[channel_axis] != 3 or not all(array.shape):
        raise ValueError("RGB images require three channels and nonempty spatial dimensions")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Use normalized floating-point RGB images, not JPEG bytes or uint8")
    array = np.moveaxis(array, channel_axis, -1).astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("RGB values must be finite before clamping")
    if prediction:
        return np.clip(array, 0., 1.)
    if array.min() < 0 or array.max() > 1:
        raise ValueError("Ground truth RGB must be normalized to [0, 1]")
    return array


def _labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not all(labels.shape):
        raise ValueError("ROI labels must be a nonempty HxW array")
    if labels.dtype == np.bool_:
        labels, _ = ndimage.label(labels, structure=_CONNECTIVITY)
    elif not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("ROI labels must be boolean or nonnegative integer IDs")
    if labels.min() < 0 or labels.max() > np.iinfo(np.int32).max:
        raise ValueError("ROI label IDs must fit nonnegative int32")
    return np.ascontiguousarray(labels, dtype="<i4")


def roi_sha256(labels):
    """Hash exact labels plus dimensions; stable across integer dtype/endianness."""
    labels = _labels(labels)
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", *labels.shape))
    digest.update(labels.tobytes(order="C"))
    return digest.hexdigest()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _name(image):
    if not isinstance(image, str) or not image:
        raise ValueError("image must be a nonempty validation image name")


def _db(mse):
    return -10. * math.log10(max(float(mse), 1e-12))


def freeze_validation_roi(baseline, target, *, image, patch_size=32,
                          error_ratio=2., mse_floor=1e-4, channel_axis=0):
    """Return read-only int32 labels and metadata from the common baseline.

    A patch is selected when its mean RGB squared error is strictly larger than
    max(error_ratio * whole-image MSE, mse_floor). Partial border patches use
    their actual areas. Adjacent selected patches form four-connected regions.
    There is no fallback to select pixels when no patch clears the threshold.
    Inputs are CHW by default; explicitly use channel_axis=-1 for HWC arrays.
    """
    _name(image)
    if isinstance(patch_size, bool) or not isinstance(patch_size, Integral) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    error_ratio = _number(error_ratio, "error_ratio")
    mse_floor = _number(mse_floor, "mse_floor")
    baseline = _rgb(baseline, prediction=True, channel_axis=channel_axis)
    target = _rgb(target, prediction=False, channel_axis=channel_axis)
    if baseline.shape != target.shape:
        raise ValueError("Baseline and target RGB shapes must match")
    patches, global_mse = patch_mse(np.moveaxis(baseline, -1, 0),
                                    np.moveaxis(target, -1, 0), int(patch_size))
    threshold = max(global_mse * error_ratio, mse_floor)
    grid_labels, count = ndimage.label(patches > threshold, structure=_CONNECTIVITY)
    labels = _labels(expand_patches(grid_labels, *target.shape[:2], int(patch_size)))
    metadata = {
        "schema": "rt_port.validation_roi.v1", "image": image, "split": "validation",
        "source": "common_baseline_before_branch_training",
        "meaning": "fixed baseline error proxy, not glass truth or cross-camera correspondence",
        "training_use": False, "roi_sha256": roi_sha256(labels),
        "height": int(labels.shape[0]), "width": int(labels.shape[1]),
        "patch_size": int(patch_size), "error_ratio": error_ratio, "mse_floor": mse_floor,
        "threshold_mse": float(threshold), "baseline_full_mse": float(global_mse),
        "baseline_full_psnr_db": _db(global_mse), "regions": int(count),
        "candidate_pixels": int(np.count_nonzero(labels)),
        "region_pixels": {f"{image}:{label}": int(np.count_nonzero(labels == label))
                          for label in range(1, int(count) + 1)},
    }
    labels.setflags(write=False)
    return labels, metadata


def measure_view(control, rt, target, labels, *, image, split="validation",
                 expected_roi_sha256=None, channel_axis=0):
    """Return one selection.py view record measured before image encoding.

    Pass the hash saved before branch training as expected_roi_sha256. The exact
    same frozen labels are used for both predictions; they are never updated
    from either model's scores. Predictions are clamped to [0,1], target is not.
    """
    if split != "validation":
        raise ValueError("Selection metrics must use split='validation', never train/test")
    return _measure_view(control, rt, target, labels, image=image, split=split,
                         expected_roi_sha256=expected_roi_sha256, channel_axis=channel_axis)


def measure_test_view(control, rt, target, labels, *, image,
                      expected_roi_sha256=None, channel_axis=0):
    """Measure a held-out test view after model selection is irrevocably frozen.

    Test scoring shares precisely the validation metric calculation, but every
    returned record is marked split='test' so selection.decide rejects it. For a
    full-image-only report use an all-zero HxW label map; do not mine new test
    regions or use these results to adjust candidates, thresholds or selection.
    The caller is responsible for freezing the decision before rendering test.
    """
    return _measure_view(control, rt, target, labels, image=image, split="test",
                         expected_roi_sha256=expected_roi_sha256, channel_axis=channel_axis)


def _measure_view(control, rt, target, labels, *, image, split,
                  expected_roi_sha256, channel_axis):
    _name(image)
    control = _rgb(control, prediction=True, channel_axis=channel_axis)
    rt = _rgb(rt, prediction=True, channel_axis=channel_axis)
    target = _rgb(target, prediction=False, channel_axis=channel_axis)
    if control.shape != target.shape or rt.shape != target.shape:
        raise ValueError("Control, RT and target RGB shapes must match")
    labels = _labels(labels)
    if labels.shape != target.shape[:2]:
        raise ValueError("Frozen ROI dimensions must match the evaluated images")
    digest = roi_sha256(labels)
    if expected_roi_sha256 is not None and digest != expected_roi_sha256:
        raise ValueError("Frozen evaluation ROI hash changed")
    control_error = np.square(control - target).mean(axis=-1)
    rt_error = np.square(rt - target).mean(axis=-1)
    regions = []
    for label in np.unique(labels):
        if label == 0:
            continue
        selected = labels == label
        cmse, rmse = float(control_error[selected].mean()), float(rt_error[selected].mean())
        regions.append({"id": f"{image}:{int(label)}", "pixels": int(selected.sum()),
                        "control_mse": cmse, "rt_mse": rmse,
                        "control_psnr_db": _db(cmse), "rt_psnr_db": _db(rmse)})
    outside = labels == 0
    outside_pixels = int(outside.sum())
    full_control, full_rt = float(control_error.mean()), float(rt_error.mean())
    return {
        "image": image, "split": split, "roi_sha256": digest,
        "full_control_mse": full_control, "full_rt_mse": full_rt,
        "full_control_psnr_db": _db(full_control), "full_rt_psnr_db": _db(full_rt),
        "regions": regions, "outside_pixels": outside_pixels,
        "outside_control_mse": float(control_error[outside].mean()) if outside_pixels else 0.,
        "outside_rt_mse": float(rt_error[outside].mean()) if outside_pixels else 0.,
        "metric": "normalized RGB, prediction clamped [0,1], float64 MSE before image encoding",
    }
