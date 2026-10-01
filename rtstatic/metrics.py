"""Metrics on actual selected pixels, never on a black-padded full image."""
import math

import cv2
import numpy as np


def region_metrics(prediction, target, mask):
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape != (*mask.shape, 3):
        raise ValueError("Expected matching H x W x 3 images and H x W mask")
    if not (np.isfinite(prediction).all() and np.isfinite(target).all()):
        raise ValueError("Non-finite render")
    count = int(mask.sum())
    if not count:
        return dict(pixels=0, mse=None, psnr=None, mae=None, ssim=None, ssim_pixels=0)
    error = prediction[mask] - target[mask]
    mse = float(np.mean(error ** 2))
    # A selected SSIM center is valid only if its full 11x11 support is in the ROI.
    kernel = np.ones((11, 11), np.uint8)
    interior = cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    ssim = None
    if interior.any():
        def blur(x):
            return cv2.GaussianBlur(x, (11, 11), 1.5)
        ma, mb = blur(prediction), blur(target)
        va = blur(prediction * prediction) - ma * ma
        vb = blur(target * target) - mb * mb
        cov = blur(prediction * target) - ma * mb
        ssim_map = ((2 * ma * mb + .01 ** 2) * (2 * cov + .03 ** 2)) / ((ma * ma + mb * mb + .01 ** 2) * (va + vb + .03 ** 2))
        ssim = float(ssim_map[interior].mean())
    return dict(pixels=count, mse=mse, psnr=-10 * math.log10(max(mse, 1e-12)),
                mae=float(np.abs(error).mean()), ssim=ssim, ssim_pixels=int(interior.sum()))


def all_metrics(prediction, target, mask):
    return {name: region_metrics(prediction, target, selected) for name, selected in
            [("full", np.ones(mask.shape, bool)), ("window", mask), ("opaque", ~mask)]}
