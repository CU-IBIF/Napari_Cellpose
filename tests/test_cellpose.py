import numpy as np
import pytest

from ibif_napari import _cellpose


class FakeModel:
    def __init__(self):
        self.calls = []

    def eval(self, x, **kwargs):
        self.calls.append((x.shape, kwargs))
        z_axis = kwargs["z_axis"]
        ch = kwargs["channel_axis"]
        shape = [s for i, s in enumerate(x.shape) if i != ch]
        masks = np.zeros(shape, dtype=np.int32)
        masks[..., :2, :2] = 1
        return masks, None, None


@pytest.fixture(params=[3, 4])
def fake(monkeypatch, request):
    model = FakeModel()
    monkeypatch.setattr(_cellpose, "cellpose_major", lambda: request.param)
    monkeypatch.setattr(_cellpose, "get_model", lambda name, gpu: model)
    model.major = request.param
    return model


def test_3d_with_nuclei_stacks_channels(fake):
    img = np.zeros((4, 16, 16))
    masks = _cellpose.segment(img, img, do_3D=True, anisotropy=3.0)
    assert masks.shape == img.shape
    assert masks.dtype == np.uint16
    shape, kw = fake.calls[0]
    assert shape == (4, 2, 16, 16)
    assert kw["channel_axis"] == 1 and kw["z_axis"] == 0
    assert kw["do_3D"] is True and kw["anisotropy"] == 3.0
    assert kw["stitch_threshold"] == 0.0
    if fake.major == 3:
        assert kw["channels"] == [1, 2]
    else:
        assert "channels" not in kw


def test_stitch_mode(fake):
    img = np.zeros((4, 16, 16))
    _cellpose.segment(img, do_3D=False, stitch_threshold=0.25)
    _, kw = fake.calls[0]
    assert kw["do_3D"] is False and kw["stitch_threshold"] == 0.25
    assert kw["channel_axis"] is None


def test_2d_image(fake):
    masks = _cellpose.segment(np.zeros((16, 16)), anisotropy=2.0)
    assert masks.shape == (16, 16)
    _, kw = fake.calls[0]
    assert kw["z_axis"] is None and kw["do_3D"] is False and kw["anisotropy"] is None


def test_bad_shapes(fake):
    with pytest.raises(ValueError):
        _cellpose.segment(np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValueError):
        _cellpose.segment(np.zeros((2, 4, 4)), np.zeros((3, 4, 4)))


@pytest.fixture
def fake_cp4(monkeypatch, tmp_path):
    """A stand-in `cellpose.models` (Cellpose 4) whose weights come from a local folder."""
    import sys
    import types

    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "cpsam_v2").write_bytes(b"w" * 3_000_000)
    created = []

    class CellposeModel:
        def __init__(self, gpu=False, pretrained_model="cpsam_v2", use_bfloat16=True):
            created.append(dict(gpu=gpu, pretrained_model=pretrained_model, bf16=use_bfloat16))

    mod = types.SimpleNamespace(
        MODEL_DIR=tmp_path / "models",
        MODEL_NAMES=["cpsam_v2", "cpsam"],
        _MODEL_URL=remote.as_uri() + "/",
        CellposeModel=CellposeModel,
    )
    pkg = types.ModuleType("cellpose")
    pkg.models = mod
    monkeypatch.setitem(sys.modules, "cellpose", pkg)
    monkeypatch.setitem(sys.modules, "cellpose.models", mod)
    monkeypatch.setattr(_cellpose, "cellpose_major", lambda: 4)
    monkeypatch.setattr(_cellpose, "_models", {})
    mod.created = created
    return mod


def test_load_model_downloads_then_loads_on_cpu_in_float32(fake_cp4):
    assert _cellpose.builtin_models() == ["cpsam_v2", "cpsam"]
    gen = _cellpose.load_model("cpsam_v2", gpu=False)
    messages = []
    try:
        while True:
            messages.append(next(gen))
    except StopIteration as stop:
        model = stop.value
    assert (fake_cp4.MODEL_DIR / "cpsam_v2").stat().st_size == 3_000_000
    assert not list(fake_cp4.MODEL_DIR.glob("*.part"))
    assert any("Downloading" in m or "Connecting" in m for m in messages)
    assert messages[-1].startswith("Loading Cellpose model")
    assert fake_cp4.created == [dict(gpu=False, pretrained_model="cpsam_v2", bf16=False)]

    # Already downloaded and loaded: nothing to report, same model object.
    assert list(_cellpose.load_model("cpsam_v2")) == []
    assert _cellpose.get_model("cpsam_v2") is model


def test_stalled_download_raises_instead_of_hanging(fake_cp4, monkeypatch):
    import socket
    import threading

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    conns = []
    threading.Thread(target=lambda: conns.append(server.accept()), daemon=True).start()
    fake_cp4._MODEL_URL = f"http://127.0.0.1:{server.getsockname()[1]}/"
    monkeypatch.setattr(_cellpose, "DOWNLOAD_TIMEOUT_S", 1)
    try:
        with pytest.raises(RuntimeError, match="could not download"):
            list(_cellpose.ensure_weights("cpsam"))
    finally:
        server.close()
    assert not (fake_cp4.MODEL_DIR / "cpsam").exists()
    assert not list(fake_cp4.MODEL_DIR.glob("*.part"))
    assert not _cellpose._download_lock.locked()


def test_cpdino_models_hidden_without_dinov3(fake_cp4, monkeypatch):
    import importlib.util

    fake_cp4.MODEL_NAMES = ["cpsam_v2", "cpdino", "cpdino-vitb", "cpsam"]
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "dinov3" else real(n, *a))
    assert _cellpose.builtin_models() == ["cpsam_v2", "cpsam"]
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: object() if n == "dinov3" else real(n, *a))
    assert _cellpose.builtin_models() == ["cpsam_v2", "cpdino", "cpdino-vitb", "cpsam"]
