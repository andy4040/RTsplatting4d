"""Analytic textured plane for execution tests, never a glass-quality benchmark."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .data import calibration, write_json


def make_raw_fixture(root, width=320):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    height, focal = width * 3 // 4, float(width)
    rng = np.random.default_rng(7)
    texture = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), .6)
    for _ in range(width):
        xy = tuple(rng.integers([0, 0], [width, height]).tolist())
        color = tuple(rng.integers(0, 256, 3).tolist())
        cv2.circle(texture, xy, int(rng.integers(2, 7)), color, -1)
    rows = []
    names = ["cam00", "cam02", "cam05"]
    for name, center in zip(names, [-.3, 0., .3]):
        pose = np.column_stack([[0, 1, 0], [1, 0, 0], [0, 0, -1], [center, 0, 0], [height, width, focal]])
        rows.append(np.r_[pose.ravel(), [1., 5.]])
        folder = root / name / "images"
        folder.mkdir(parents=True)
        image = cv2.warpAffine(texture, np.float32([[1, 0, -focal * center / 3], [0, 1, 0]]), (width, height))
        Image.fromarray(image).save(folder / "0000.png")
    np.save(root / "poses_bounds.npy", np.asarray(rows))
    return texture


def make_prepared_fixture(root, width=96):
    root = Path(root)
    texture = make_raw_fixture(root / "raw", width)
    scene = root / "scene"
    scene.mkdir()
    (scene / "images").mkdir()
    (scene / "masks").mkdir()
    cameras = calibration(root / "raw", width)
    for c in cameras:
        image = Image.open(root / "raw" / c["name"] / "images" / "0000.png")
        image.save(scene / "images" / f"{c['name']}.png")
        mask = np.zeros((c["height"], c["width"]), dtype=np.uint8)
        mask[mask.shape[0] // 4:3 * mask.shape[0] // 4, mask.shape[1] // 4:3 * mask.shape[1] // 4] = 255
        Image.fromarray(mask).save(scene / "masks" / f"{c['name']}.png")
    y, x = np.mgrid[5:texture.shape[0] - 5:4, 5:width - 5:4]
    xyz = np.column_stack([(x.ravel() - width / 2) * 3 / width,
                           (y.ravel() - texture.shape[0] / 2) * 3 / width, np.full(x.size, 3)])
    colors = texture[y.ravel(), x.ravel()] / 255.
    np.savez_compressed(scene / "points.npz", xyz=xyz.astype(np.float32), colors=colors.astype(np.float32))
    train_centers = np.asarray([c["center"] for c in cameras if c["split"] == "train"])
    extent = float(np.linalg.norm(train_centers - train_centers.mean(0), axis=1).max() * 1.1)
    write_json(scene / "scene.json", dict(version=1, cameras=cameras, source="synthetic execution fixture",
        frame=0, width=width, test_camera="cam00", seed=7, points=len(xyz), extent=extent,
        initialization="analytic synthetic plane (not a reconstruction result)",
        initialization_cameras=[c["name"] for c in cameras if c["split"] == "train"]))
    return scene
