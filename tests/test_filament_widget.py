import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("skan")

from _synthetic import SPACING, draw  # noqa: E402

from ibif_napari._filament_widget import FilamentWidget  # noqa: E402

SP = np.asarray(SPACING)


@pytest.fixture
def viewer(qapp):
    from napari.components import ViewerModel

    v = ViewerModel()
    # T Z Y X: a Y-shaped filament at t=1 only, and a second, separate filament
    img = np.zeros((2, 12, 100, 160), np.float32)
    img[1] = draw((12, 100, 160), [((2, 3, 2), (2, 3, 14)), ((2, 3, 8), (2.4, 7, 12)),
                                   ((2, 8, 2), (2, 8, 8))])
    v.add_image(100 * img + 100, name="neuron", scale=(1, *SPACING))
    v.dims.set_current_step(0, 1)
    return v


def wait_idle(qtbot, w):
    qtbot.waitUntil(lambda: w._worker is None and not w._pending, timeout=60000)


def test_automatic_tracing(viewer, qtbot, tmp_path):
    w = FilamentWidget(viewer)
    qtbot.addWidget(w)
    w.d_min.setValue(0.2)
    w.d_max.setValue(0.8)
    w.prune.setValue(0.5)
    w.run()
    wait_idle(qtbot, w)
    assert "2 filament(s)" in w.status.text(), w.status.text()
    skel = viewer.layers["neuron filaments T1"]
    assert skel.data.shape == (12, 100, 160) and set(np.unique(skel.data)) == {0, 1, 2}
    assert tuple(skel.scale) == pytest.approx(SPACING)
    points = viewer.layers["neuron filament points T1"]
    assert set(points.features["type"]) == {"root", "branch", "terminal"}
    assert w.table.rowCount() == 2
    lengths = sorted(w._session.result.filaments["length_um"])
    assert lengths == pytest.approx([6, 17.7], rel=0.08)
    # Re-tracing with another threshold reuses the filtered image and the layers
    enhanced = w._session.enhanced[1]
    w.threshold.setValue(0.9)
    w.run()
    wait_idle(qtbot, w)
    assert w._session.enhanced[1] is enhanced
    assert len([l for l in viewer.layers if l.name.startswith("neuron filaments")]) == 1

    written = w.export(str(tmp_path / "n"))
    assert len(written) == 5 and (tmp_path / "n_segments.csv").exists()


def test_click_to_trace(viewer, qtbot):
    w = FilamentWidget(viewer)
    qtbot.addWidget(w)
    w.d_min.setValue(0.2)
    w.d_max.setValue(0.8)
    w.click_btn.setChecked(True)
    clicks = viewer.layers["neuron trace clicks T1"]
    assert clicks.mode == "add"
    # Along the trunk: start, middle, end (pixels; clicked on the right plane)
    for zyx in ((2, 3, 2.2), (2, 3, 8), (2, 3, 13.8)):
        clicks.add(np.asarray(zyx) / SP)
        wait_idle(qtbot, w)
    f = w._session.result.filaments
    assert len(f) == 1
    assert f["length_um"].iloc[0] == pytest.approx(11.6, rel=0.06)
    assert f["mean_diameter_um"].iloc[0] > 0
    # A side branch from the trunk: the first click snaps onto the traced filament
    w.new_filament()
    for zyx in ((2, 3.1, 8.05), (2.4, 7, 12)):
        clicks.add(np.asarray(zyx) / SP)
        wait_idle(qtbot, w)
    f = w._session.result.filaments
    assert len(f) == 1 and f["branch_points"].iloc[0] == 1
    assert f["length_um"].iloc[0] == pytest.approx(11.6 + 5.7, rel=0.08)
    w.undo_last()
    wait_idle(qtbot, w)
    f = w._session.result.filaments
    assert f["branch_points"].iloc[0] == 0
    assert f["length_um"].iloc[0] == pytest.approx(11.6, rel=0.08)


def test_region_limits_tracing(viewer, qtbot):
    w = FilamentWidget(viewer)
    qtbot.addWidget(w)
    w.d_min.setValue(0.2)
    w.d_max.setValue(0.8)
    w.add_region_layer()
    region = viewer.layers["neuron region"]
    region.data[:, 60:95, 10:90] = 1  # around the second filament only
    w.region.setCurrentText("neuron region")
    w.run()
    wait_idle(qtbot, w)
    skel = viewer.layers["neuron filaments T1 region"]
    assert skel.data.shape == (12, 35, 80)
    assert tuple(skel.translate) == pytest.approx((0, 6.0, 1.0))
    r = w._session.result
    assert len(r.filaments) == 1
    # Positions are in the image's um coordinates, not the crop's
    assert r.filaments["root_y_um"].iloc[0] == pytest.approx(8, abs=0.3)
