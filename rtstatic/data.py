"""CPU-only preparation; OpenCV camera convention, zero-based video frames."""
import hashlib
import itertools
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def calibration(root, width, test_camera="cam00"):
    root = Path(root)
    raw = np.load(root / "poses_bounds.npy", allow_pickle=False)
    if raw.ndim != 2 or raw.shape[1] != 17 or not np.isfinite(raw).all():
        raise ValueError("poses_bounds.npy must be finite N x 17 LLFF calibration")
    names = {p.stem for p in root.glob("cam*.mp4")}
    names |= {p.name for p in root.glob("cam*") if p.is_dir()}
    names = sorted(n for n in names if re.fullmatch(r"cam\d+", n))
    if len(names) != len(raw) or len({int(n[3:]) for n in names}) != len(names):
        raise ValueError("Calibration rows must match sorted existing camera streams (gaps are allowed)")
    if len(names) < 3 or test_camera not in names or width < 16:
        raise ValueError("Need at least two training cameras, the test camera, and width >= 16")
    cameras = []
    for name, row in zip(names, raw):
        pose = row[:15].reshape(3, 5)
        h, w, f = pose[:, 4]
        if min(h, w, f) <= 0:
            raise ValueError(f"{name}: invalid intrinsics")
        # N3DV / LLFF columns [down, right, back] -> OpenCV [right, down, forward].
        c2w = np.column_stack([pose[:, 1], pose[:, 0], -pose[:, 2]])
        if not np.allclose(c2w.T @ c2w, np.eye(3), atol=2e-3) or np.linalg.det(c2w) < 0.99:
            raise ValueError(f"{name}: invalid camera rotation")
        rotation = c2w.T
        translation = -rotation @ pose[:, 3]
        height = round(h * width / w)
        k = [[f * width / w, 0, width / 2], [0, f * height / h, height / 2], [0, 0, 1]]
        cameras.append(dict(name=name, R=rotation.tolist(), T=translation.tolist(), K=k,
                            width=width, height=height, center=pose[:, 3].tolist(),
                            split="test" if name == test_camera else "train"))
    return cameras


def read_frame(root, name, frame):
    if frame < 0:
        raise ValueError("Frame index is zero-based and must be nonnegative")
    folder = Path(root) / name / "images"
    files = [p for p in folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.stem.isdigit()]
    if files:
        if not any(int(p.stem) == 0 for p in files):
            raise ValueError(f"{folder}: no frame 0; ambiguous numbering")
        selected = [p for p in files if int(p.stem) == frame]
        if len(selected) != 1:
            raise ValueError(f"{folder}: expected exactly one image for frame {frame}")
        with Image.open(selected[0]) as image:
            return np.asarray(image.convert("RGB"))
    cap = cv2.VideoCapture(str(Path(root) / f"{name}.mp4"))
    try:
        # Sequential grab avoids inaccurate keyframe seeking in compressed video.
        for _ in range(frame + 1):
            if not cap.grab():
                raise ValueError(f"Cannot decode {name} at frame {frame}")
        ok, image = cap.retrieve()
        if not ok:
            raise ValueError(f"Cannot retrieve {name} at frame {frame}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def load_mask(path, size):
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"))
        if not np.isin(values, [0, 1, 255]).all():
            raise ValueError(f"{path}: mask must be binary (0 background, 1/255 transparent)")
        image = Image.fromarray((values > 0).astype(np.uint8) * 255)
        return np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0


def triangulate(cameras, images, max_points=50000, seed=0, min_points=100):
    """Mutual SIFT matches, calibrated reprojection, cheirality and parallax filtering."""
    train = [c for c in cameras if c["split"] == "train"]
    centers = np.asarray([c["center"] for c in train])
    extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    if extent < 1e-8:
        raise ValueError("Degenerate training camera baseline")
    sift = cv2.SIFT_create(nfeatures=8000)
    features = {c["name"]: sift.detectAndCompute(cv2.cvtColor(images[c["name"]], cv2.COLOR_RGB2GRAY), None) for c in train}
    matcher = cv2.BFMatcher()
    xyz, rgb = [], []
    for a, b in itertools.combinations(train, 2):
        ka, da = features[a["name"]]
        kb, db = features[b["name"]]
        if da is None or db is None or min(len(da), len(db)) < 2:
            continue
        def good_matches(left, right):
            return [pair[0] for pair in matcher.knnMatch(left, right, k=2)
                    if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
        reverse = {m.queryIdx: m.trainIdx for m in good_matches(db, da)}
        matches = [m for m in good_matches(da, db) if reverse.get(m.trainIdx) == m.queryIdx]
        if len(matches) < 8:
            continue
        ua = np.asarray([ka[m.queryIdx].pt for m in matches], dtype=np.float64)
        ub = np.asarray([kb[m.trainIdx].pt for m in matches], dtype=np.float64)
        def projection(c):
            return np.asarray(c["K"]) @ np.column_stack([c["R"], c["T"]])
        homogeneous = cv2.triangulatePoints(projection(a), projection(b), ua.T, ub.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            points = (homogeneous[:3] / homogeneous[3:]).T
            good = np.isfinite(points).all(1)
            for c, uv in [(a, ua), (b, ub)]:
                local = points @ np.asarray(c["R"]).T + c["T"]
                pixels = local @ np.asarray(c["K"]).T
                good &= (local[:, 2] > 0.01) & (local[:, 2] < 90)
                good &= np.linalg.norm(pixels[:, :2] / pixels[:, 2:] - uv, axis=1) < 1.5
            ra, rb = points - a["center"], points - b["center"]
            cosine = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1))
            good &= cosine < np.cos(np.deg2rad(1))
        pixels = np.rint(ua[good]).astype(int)
        if len(pixels):
            pixels[:, 0] = pixels[:, 0].clip(0, a["width"] - 1)
            pixels[:, 1] = pixels[:, 1].clip(0, a["height"] - 1)
            xyz.extend(points[good])
            rgb.extend(images[a["name"]][pixels[:, 1], pixels[:, 0]] / 255.)
    if not xyz:
        raise ValueError("No triangulated points. Check camera poses and image correspondences.")
    xyz, rgb = np.asarray(xyz), np.asarray(rgb)
    _, indices = np.unique(np.round(xyz / (extent * 0.001)), axis=0, return_index=True)
    np.random.default_rng(seed).shuffle(indices)
    indices = indices[:max_points]
    if len(indices) < min_points:
        raise ValueError(f"Only {len(indices)} valid points; need {min_points}. Increase width/check calibration.")
    return xyz[indices].astype(np.float32), rgb[indices].astype(np.float32), extent


def prepare(root, out, frame=0, width=1352, test_camera="cam00", seed=0, max_points=50000):
    root, out = Path(root).resolve(), Path(out).resolve()
    if max_points < 100 or frame < 0:
        raise ValueError("max_points must be >= 100 and frame must be nonnegative")
    cameras = calibration(root, width, test_camera)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose a new output directory")
    out.mkdir(parents=True)
    (out / "images").mkdir()
    (out / "masks").mkdir()
    images = {}
    for c in cameras:
        image = read_frame(root, c["name"], frame)
        image = cv2.resize(image, (c["width"], c["height"]), interpolation=cv2.INTER_AREA)
        Image.fromarray(image).save(out / "images" / f"{c['name']}.png")
        # Held-out RGB is exported for evaluation, never passed to triangulation.
        if c["split"] == "train":
            images[c["name"]] = image
    manifest = dict(version=1, source=str(root), frame=frame, width=width, seed=seed,
                    test_camera=test_camera, cameras=cameras, initialization="pending")
    write_json(out / "scene.json", manifest)
    xyz, colors, extent = triangulate(cameras, images, max_points, seed)
    np.savez_compressed(out / "points.npz", xyz=xyz, colors=colors)
    manifest.update(initialization="training-only calibrated SIFT triangulation", points=len(xyz), extent=extent,
                    initialization_cameras=[c["name"] for c in cameras if c["split"] == "train"])
    write_json(out / "scene.json", manifest)
    return manifest


def validate_scene(root, require_masks=True):
    root = Path(root)
    manifest = read_json(root / "scene.json")
    cameras = manifest["cameras"]
    names = [c["name"] for c in cameras]
    train = [c["name"] for c in cameras if c["split"] == "train"]
    test = [c["name"] for c in cameras if c["split"] == "test"]
    if len(names) != len(set(names)) or len(train) < 2 or test != [manifest["test_camera"]]:
        raise ValueError("Invalid camera split")
    if sorted(manifest.get("initialization_cameras", [])) != sorted(train):
        raise ValueError("Initialization must use exactly the training cameras")
    if not (root / "points.npz").exists():
        raise ValueError("Preparation incomplete: missing points.npz")
    with np.load(root / "points.npz", allow_pickle=False) as cloud:
        xyz, colors = cloud["xyz"], cloud["colors"]
        if xyz.ndim != 2 or xyz.shape[1] != 3 or colors.shape != xyz.shape or len(xyz) < 4:
            raise ValueError("Invalid initialization point/color dimensions")
        if not np.isfinite(xyz).all() or not np.isfinite(colors).all() or (colors < 0).any() or (colors > 1).any():
            raise ValueError("Initialization must contain finite points and colors in [0,1]")
    files = [root / "scene.json", root / "points.npz"]
    nonempty_train, nonempty_test = False, False
    for c in cameras:
        path = root / "images" / f"{c['name']}.png"
        with Image.open(path) as image:
            if image.size != (c["width"], c["height"]):
                raise ValueError(f"Image dimensions changed: {path}")
        files.append(path)
        if require_masks:
            path = root / "masks" / f"{c['name']}.png"
            mask = load_mask(path, (c["width"], c["height"]))
            nonempty_train |= bool(mask.any()) and c["split"] == "train"
            nonempty_test |= bool(mask.any()) and c["split"] == "test"
            files.append(path)
    if require_masks and not (nonempty_train and nonempty_test):
        raise ValueError("Window experiment needs a nonempty mask in training and held-out views")
    manifest["fingerprint"] = hashlib.sha256("".join(sha256(p) for p in files).encode()).hexdigest()
    return manifest
