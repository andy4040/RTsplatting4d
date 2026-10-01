import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rtstatic.annotation import export_editor, import_polygons
from rtstatic.data import calibration, load_mask, prepare, read_frame, read_json, validate_scene, write_json
from rtstatic.fixture import make_prepared_fixture, make_raw_fixture
from rtstatic.metrics import all_metrics, region_metrics


def test_calibration_gapped_cameras_and_projection(tmp_path):
    root = tmp_path / "raw"
    make_raw_fixture(root)
    cameras = calibration(root, 160)
    assert [c["name"] for c in cameras] == ["cam00", "cam02", "cam05"]
    for c, center in zip(cameras, [-.3, 0., .3]):
        np.testing.assert_allclose(c["R"], np.eye(3))
        np.testing.assert_allclose(c["T"], [-center, 0, 0])
        local = np.asarray(c["R"]) @ np.array([0, 0, 3]) + c["T"]
        pixel = np.asarray(c["K"]) @ local
        np.testing.assert_allclose(pixel[:2] / pixel[2], [80 - 160 * center / 3, 60])


def test_prepare_triangulates_plane_without_heldout_rgb(tmp_path):
    raw = tmp_path / "raw"
    make_raw_fixture(raw)
    first = tmp_path / "first"
    manifest = prepare(raw, first, width=320)
    assert manifest["initialization_cameras"] == ["cam02", "cam05"]
    points = np.load(first / "points.npz")
    assert len(points["xyz"]) >= 100
    np.testing.assert_allclose(np.median(points["xyz"][:, 2]), 3, atol=.03)
    # Changing all test RGB cannot influence initial geometry or colors.
    Image.new("RGB", (320, 240), "red").save(raw / "cam00" / "images" / "0000.png")
    second = tmp_path / "second"
    prepare(raw, second, width=320)
    changed = np.load(second / "points.npz")
    np.testing.assert_array_equal(points["xyz"], changed["xyz"])
    np.testing.assert_array_equal(points["colors"], changed["colors"])
    with pytest.raises(FileExistsError):
        prepare(raw, first, width=320)


def test_frame_numbering_and_calibration_mismatch(tmp_path):
    raw = tmp_path / "raw"
    make_raw_fixture(raw)
    image = raw / "cam02" / "images" / "0000.png"
    image.rename(image.with_name("0001.png"))
    with pytest.raises(ValueError, match="no frame 0"):
        read_frame(raw, "cam02", 1)
    with pytest.raises(ValueError, match="nonnegative"):
        read_frame(raw, "cam00", -1)
    (raw / "cam09").mkdir()
    with pytest.raises(ValueError, match="rows"):
        calibration(raw, 160)


def test_true_masked_psnr_is_not_inflated_by_background():
    target = np.zeros((32, 32, 3))
    pred = target.copy()
    mask = np.zeros((32, 32), bool)
    mask[:8, :8] = True
    pred[mask] = .1
    scores = all_metrics(pred, target, mask)
    assert scores["window"]["psnr"] == pytest.approx(20)
    assert scores["full"]["psnr"] == pytest.approx(20 + 10 * np.log10(16))
    assert scores["window"]["ssim"] is None  # no complete 11x11 window
    assert scores["opaque"]["mse"] == 0
    assert region_metrics(pred, target, np.zeros(mask.shape, bool))["psnr"] is None


def test_ssim_ignores_error_outside_roi():
    target = np.full((64, 64, 3), .5)
    pred = target.copy()
    mask = np.zeros((64, 64), bool)
    mask[16:48, 16:48] = True
    pred[~mask] = 1
    scores = region_metrics(pred, target, mask)
    assert scores["ssim"] == pytest.approx(1)
    assert scores["ssim_pixels"] == 22 * 22


def test_mask_import_binary_and_scene_fingerprint(tmp_path):
    scene = make_prepared_fixture(tmp_path / "fixture")
    before = validate_scene(scene)
    editor = tmp_path / "editor.html"
    export_editor(scene, editor)
    assert "__IMAGES__" not in editor.read_text(encoding="utf-8")
    assert "data:image/png;base64," in editor.read_text(encoding="utf-8")
    names = [c["name"] for c in before["cameras"]]
    polygon = [[.1, .1], [.6, .1], [.6, .6], [.1, .6]]
    annotations = dict(frame=0, polygons={name: [polygon] for name in names}, reviewed=names)
    source = tmp_path / "polygons.json"
    write_json(source, annotations)
    import_polygons(scene, source)
    after = validate_scene(scene)
    assert before["fingerprint"] != after["fingerprint"]
    assert set(np.unique(np.array(Image.open(scene / "masks" / "cam00.png")))) == {0, 255}
    annotations["reviewed"] = []
    write_json(source, annotations)
    with pytest.raises(ValueError, match="Review"):
        import_polygons(scene, source)
    Image.new("L", (96, 72), 127).save(scene / "masks" / "cam00.png")
    with pytest.raises(ValueError, match="binary"):
        load_mask(scene / "masks" / "cam00.png", (96, 72))


def test_missing_masks_and_split_leakage_fail(tmp_path):
    scene = make_prepared_fixture(tmp_path / "fixture")
    path = scene / "scene.json"
    metadata = read_json(path)
    metadata["initialization_cameras"].append("cam00")
    write_json(path, metadata)
    with pytest.raises(ValueError, match="Initialization"):
        validate_scene(scene)
    metadata["initialization_cameras"].remove("cam00")
    write_json(path, metadata)
    (scene / "masks" / "cam00.png").unlink()
    with pytest.raises(FileNotFoundError):
        validate_scene(scene)


def test_gradient_gating_preserves_forward_and_scales_backward():
    torch = pytest.importorskip("torch")
    from rtstatic.losses import gate_transmission
    transmitted = torch.tensor([.2, .4, .6], requires_grad=True)
    complexity = torch.tensor([0., .5, 1.], requires_grad=True)
    result = gate_transmission(transmitted, complexity)
    torch.testing.assert_close(result, transmitted)
    result.sum().backward()
    torch.testing.assert_close(transmitted.grad, torch.tensor([1., .5, 0.]))
    assert complexity.grad is None


def test_checkpoint_keeps_materials_and_neural_modules(tmp_path):
    torch = pytest.importorskip("torch")
    from rtstatic.engine import save_model
    model = SimpleNamespace(active_sh_degree=2)
    for name in ["_xyz", "_opacity", "_occupancy", "_transmissivity", "_roughness", "_reflectance", "_language_feature"]:
        setattr(model, name, torch.nn.Parameter(torch.rand(4, 3)))
    model.light_mlp = torch.nn.Linear(3, 4)
    model.dir_encoding = torch.nn.Linear(4, 2)
    path = tmp_path / "checkpoint.pt"
    save_model(model, path, {"mode": "rt"}, SimpleNamespace(sh_degree=3), 10)
    state = torch.load(path, weights_only=True)
    assert set(state["parameters"]) == {name for name in vars(model) if name.startswith("_")}
    torch.testing.assert_close(state["parameters"]["_opacity"], model._opacity)
    torch.testing.assert_close(state["light_mlp"]["weight"], model.light_mlp.weight)
    torch.testing.assert_close(state["dir_encoding"]["weight"], model.dir_encoding.weight)
    assert state["active_sh_degree"] == 2


def test_resume_checkpoint_restores_optimizer_buffers_and_rng(tmp_path):
    import random
    torch = pytest.importorskip("torch")
    from rtstatic.engine import save_model
    model = SimpleNamespace(active_sh_degree=0, spatial_lr_scale=2.5)
    model._xyz = torch.nn.Parameter(torch.rand(4, 3))
    model.light_mlp = torch.nn.Linear(3, 4)
    model.dir_encoding = torch.nn.Linear(4, 2)
    model.optimizer = torch.optim.Adam([model._xyz], lr=np.float64(.01))
    for name in ["max_radii2D", "xyz_gradient_accum", "denom", "last_update"]:
        setattr(model, name, torch.rand(4, 1))
    model._xyz.square().sum().backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "checkpoint.pt"
    save_model(model, path, {"mode": "rt"}, SimpleNamespace(sh_degree=3), 5000,
               {"last_reset": 3000, "elapsed_seconds": 100.})
    expected_random, expected_tensor = random.random(), torch.rand(5)
    state = torch.load(path, map_location="cpu", weights_only=True)
    training = state["training_state"]
    random.setstate(training["python_rng"])
    torch.set_rng_state(training["torch_rng"])
    assert random.random() == expected_random
    torch.testing.assert_close(torch.rand(5), expected_tensor)
    assert training["last_reset"] == 3000
    assert training["spatial_lr_scale"] == 2.5
    for name, value in training["buffers"].items():
        torch.testing.assert_close(value, getattr(model, name))
    restored = torch.nn.Parameter(state["parameters"]["_xyz"].clone())
    optimizer = torch.optim.Adam([restored], lr=.5)
    optimizer.load_state_dict(training["optimizer"])
    model._xyz.square().sum().backward()
    restored.square().sum().backward()
    model.optimizer.step()
    optimizer.step()
    torch.testing.assert_close(restored, model._xyz)
    assert not path.with_suffix(".tmp").exists()


def test_read_early_numpy_optimizer_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from rtstatic.engine import read_checkpoint
    path = tmp_path / "early.pt"
    torch.save({"learning_rate": np.float64(.001), "weight": torch.ones(2)}, path)
    state = read_checkpoint(path)
    assert state["learning_rate"] == pytest.approx(.001)
    torch.testing.assert_close(state["weight"], torch.ones(2))


def test_resume_log_preserves_discarded_unsaved_steps(tmp_path):
    from rtstatic.engine import trim_resume_log
    rows=[json.dumps(dict(step=step,loss=.1)) for step in [14900,15000,15100,15200]]
    (tmp_path/'losses.jsonl').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    assert trim_resume_log(tmp_path,15000)==2
    assert [json.loads(line)['step'] for line in (tmp_path/'losses.jsonl').read_text().splitlines()]==[14900,15000]
    archives=list(tmp_path.glob('discarded_after_*.jsonl'))
    assert len(archives)==1
    assert [json.loads(line)['step'] for line in archives[0].read_text().splitlines()]==[15100,15200]


@pytest.mark.parametrize("mode", ["gs", "rt", "rt_no_gating"])
@pytest.mark.parametrize("transparent", [False, True])
def test_actual_upstream_losses_have_finite_gradients_for_empty_regions(monkeypatch, mode, transparent):
    torch = pytest.importorskip("torch")
    from rtstatic.upstream import DEFAULT_UPSTREAM
    if not DEFAULT_UPSTREAM.exists():
        pytest.skip("Run bootstrap.py to fetch upstream CPU loss functions")
    monkeypatch.syspath_prepend(str(DEFAULT_UPSTREAM))
    from arguments import OptimizationParams
    from rtstatic.losses import loss_for_view
    import argparse
    parser = argparse.ArgumentParser()
    group = OptimizationParams(parser)
    opt = group.extract(parser.parse_args([]))
    opt.lambda_lpips = 0
    def leaf(channels):
        return torch.rand(channels, 16, 16, requires_grad=True)
    package = {name: leaf(3) for name in ["final_rendering", "render_tran", "render_scat",
        "final_tran", "final_scat", "final_spec", "surface_normal", "surface_depth_normal"]}
    package.update(surface_opacity=leaf(1), transmissivity=leaf(1), foreground=torch.ones(1, 16, 16),
                   visibility_filter=torch.tensor([True, False, True]))
    model = SimpleNamespace(get_occupancy=torch.rand(3, 1, requires_grad=True))
    loss, components = loss_for_view(package, torch.rand(3, 16, 16),
        torch.full((1, 16, 16), transparent), model, opt, 2, mode, 1)
    assert torch.isfinite(loss)
    loss.backward()
    for value in package.values():
        if value.grad is not None:
            assert torch.isfinite(value.grad).all()
    if mode != "gs":
        assert ("consistency" in components) == transparent
        assert ("opaque_transmission" in components) != transparent
