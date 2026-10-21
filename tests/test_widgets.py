import numpy as np
import pytest

pytest.importorskip("napari")

from ibif_napari import _cellpose  # noqa: E402
from ibif_napari._cellpose_widget import SCOPE_ALL, CellposeWidget  # noqa: E402
from ibif_napari._dims_widget import DimensionsWidget  # noqa: E402
from ibif_napari._reader import layer_data  # noqa: E402


@pytest.fixture
def viewer(qapp, monkeypatch):
    # The panels only touch the viewer model, so no OpenGL canvas is needed.
    from napari.components import ViewerModel

    monkeypatch.setenv(_cellpose.PRELOAD_ENV, "0")
    v = ViewerModel()
    data = np.random.randint(0, 100, (2, 3, 4, 2, 8, 8), dtype=np.uint16)  # T P Z C Y X
    sizes = {"T": 2, "P": 3, "Z": 4, "C": 2, "Y": 8, "X": 8}
    (ld,) = layer_data(data, sizes, ["DAPI", "GFP"], [450, 520], (2.0, 0.5, 0.5), 10.0, "s", "s")
    v._add_layer_from_data(*ld)
    return v


def test_dimensions_panel(viewer, qtbot):
    w = DimensionsWidget(viewer)
    qtbot.addWidget(w)
    assert list(viewer.dims.axis_labels) == ["time", "position", "z", "y", "x"]
    assert len(w._rows) == 3  # time, position, z sliders

    slider, spin, readout = w._rows[0]
    spin.setValue(1)
    assert viewer.dims.current_step[0] == 1
    assert readout.text() == "10.0 s"

    viewer.dims.set_current_step(1, 2)
    assert w._rows[1][0].value() == 2

    w._set_ndisplay(3)
    assert viewer.dims.ndisplay == 3
    assert len(w._rows) == 2  # z is now displayed
    w._set_ndisplay(2)

    viewer.layers["s GFP"].visible = False
    w.max_projection()
    assert "s DAPI MIP" in viewer.layers
    assert "s GFP MIP" not in viewer.layers
    assert viewer.layers["s DAPI MIP"].data.shape == (8, 8)

    w.extract_current()
    extracted = [l for l in viewer.layers if l.name.startswith("s DAPI T1")]
    assert extracted and extracted[0].data.shape == (4, 8, 8)


def fake_segment(vol, nuc=None, **params):
    assert vol.shape == (4, 8, 8)
    assert params["do_3D"] is True
    out = np.zeros(vol.shape, dtype=np.uint16)
    out[:, :2, :2] = 1
    return out


def test_cellpose_panel(viewer, qtbot, monkeypatch):
    monkeypatch.setattr(_cellpose, "segment", fake_segment)
    w = CellposeWidget(viewer)
    qtbot.addWidget(w)
    w.image.setCurrentText("s GFP")
    w.nuclei.setCurrentText("s DAPI")
    assert w.anisotropy.value() == pytest.approx(4.0)

    viewer.dims.set_current_step(0, 1)
    viewer.dims.set_current_step(1, 0)
    w.run()
    qtbot.waitUntil(lambda: any("cellpose" in l.name for l in viewer.layers), timeout=10000)
    labels = [l for l in viewer.layers if "cellpose" in l.name][0]
    assert labels.name == "s GFP cellpose T1 P0"
    assert labels.data.shape == (4, 8, 8)
    assert list(labels.scale) == [2.0, 0.5, 0.5]

    qtbot.waitUntil(lambda: w._worker is None, timeout=10000)
    w.scope.setCurrentText(SCOPE_ALL)
    w.run()
    qtbot.waitUntil(lambda: "s GFP cellpose all" in viewer.layers, timeout=10000)
    assert viewer.layers["s GFP cellpose all"].data.shape == (2, 3, 4, 8, 8)
    assert viewer.layers["s GFP cellpose all"].data.max() == 1
