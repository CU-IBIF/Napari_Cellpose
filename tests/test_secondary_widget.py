import numpy as np
import pytest

pytest.importorskip("napari")

from napari.components import ViewerModel  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402

from ibif_napari._reader import layer_data  # noqa: E402
from ibif_napari._secondary_widget import CELLS, EXPAND, RIM, RING, SecondaryWidget, aligned_image  # noqa: E402


def make_viewer():
    """T=2, Z=3 stack with a membrane channel and a nuclear channel, and nuclei labels
    for both time points cropped to a region (as Cellpose region results are)."""
    v = ViewerModel()
    t, z, y, x = 2, 3, 40, 60
    membrane = np.zeros((t, z, y, x), np.float32)
    membrane[:, :, 10:31, 5:46] = 0.3
    membrane[:, :, 10, 5:46] = membrane[:, :, 30, 5:46] = 1.0
    membrane[:, :, 10:31, 5] = membrane[:, :, 10:31, 45] = membrane[:, :, 10:31, 24] = 1.0
    data = np.stack([membrane, membrane * 0], axis=2)  # T Z C Y X
    (ld,) = layer_data(data, {"T": t, "Z": z, "C": 2, "Y": y, "X": x}, ["Membrane", "DAPI"], [610, 450],
                       (2.0, 0.5, 0.5), None, "s", "s")
    v._add_layer_from_data(*ld)
    nuclei = np.zeros((t, z, 30, 50), np.uint16)  # region: Y 5..35, X 0..50
    yy, xx = np.indices((30, 50))
    for label, cx in ((1, 15), (2, 33)):
        nuclei[:, :, ((yy - 15) ** 2 + (xx - cx) ** 2) <= 16] = label
    v.add_labels(nuclei, name="nuclei", scale=(1, 2, 0.5, 0.5), translate=(0, 0, 2.5, 0),
                 metadata={"ibif_axes": "TZYX", "ibif_voxel_um": {"Z": 2, "Y": 0.5, "X": 0.5}})
    return v


def run(w, qtbot):
    w.run()
    qtbot.waitUntil(lambda: w._worker is None, timeout=30000)


def test_aligned_image_follows_the_crop(qapp):
    v = make_viewer()
    img = aligned_image(v, v.layers["s Membrane"], v.layers["nuclei"], (1,))
    assert img.shape == (3, 30, 50)
    assert np.array_equal(img, v.layers["s Membrane"].data[1, :, 5:35, 0:50])
    other = v.add_image(np.zeros((3, 40, 60)), scale=(1, 1, 1))
    with pytest.raises(ValueError, match="different pixel sizes"):
        aligned_image(v, other, v.layers["nuclei"], (0,))


def test_panel_expand_ring_rim(qtbot):
    v = make_viewer()
    w = SecondaryWidget(v)
    qtbot.addWidget(w)
    assert w.primary.currentText() == "nuclei"
    assert w.distance.suffix() == " µm"
    assert w.distance.isVisibleTo(w) and not w.channels.isVisibleTo(w)

    w.method.setCurrentText(EXPAND)
    w.distance.setValue(1.5)  # 3 px
    run(w, qtbot)
    out = v.layers["nuclei expanded 1.5 µm"]
    assert out.data.shape == (2, 3, 30, 50) and list(out.translate) == [0, 0, 2.5, 0]
    assert out.data[1, 1, 15, 8] == 1 and out.data[1, 1, 15, 7] == 0  # edge at x=11, +3 px

    w.method.setCurrentText(RING)
    w.gap.setValue(0.5)
    w.width.setValue(1.0)
    run(w, qtbot)
    ring = v.layers["nuclei ring 0.5+1 µm"].data
    assert not ring[v.layers["nuclei"].data > 0].any() and ring[0, 0, 15, 8] == 1

    w.method.setCurrentText(RIM)
    run(w, qtbot)
    rim = v.layers["nuclei rim 1 µm"].data
    assert rim[0, 0, 15, 11] == 1 and rim[0, 0, 15, 15] == 0
    assert "objects" in w.status.text()


def test_panel_whole_cells_from_nuclei(qtbot):
    v = make_viewer()
    w = SecondaryWidget(v)
    qtbot.addWidget(w)
    w.method.setCurrentText(CELLS)
    assert w.channels.isVisibleTo(w) and not w.distance.isVisibleTo(w)
    w.run()
    assert "Tick the channel" in w.status.text()

    item = w.channels.findItems("s Membrane", Qt.MatchExactly)[0]
    item.setCheckState(Qt.Checked)
    w.factor.setValue(0.5)
    w.smooth.setValue(0)
    run(w, qtbot)
    cells = v.layers["nuclei cells"].data
    assert cells.shape == (2, 3, 30, 50) and set(np.unique(cells)) == {0, 1, 2}
    # Region row 15 = image row 20: the membrane at image x=24 splits the two cells.
    assert cells[0, 1, 15, 22] == 1 and cells[0, 1, 15, 27] == 2
    assert cells[1, 2, 7, 8] == 1  # a far corner of cell 1 (image row 12), at the 2nd time point
    cyto = v.layers["nuclei cytoplasm"].data
    assert not cyto[v.layers["nuclei"].data > 0].any() and cyto[0, 0, 15, 8] == 1
    assert "Foreground threshold" in w.status.text()
    assert v.layers["nuclei cells"].metadata["ibif_seeds"] == "nuclei"
