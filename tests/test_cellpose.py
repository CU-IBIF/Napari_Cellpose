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
