import time

import numpy as np
import pytest

pytest.importorskip("napari")

from ibif_napari import _cellpose  # noqa: E402
from ibif_napari import _cellpose_widget as cellpose_widget  # noqa: E402
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


class FakeProcess:
    """In-thread stand-in for the Cellpose process (same generator API)."""

    def __init__(self, segment):
        self.segment = segment

    def load(self, name, gpu=False):
        yield f"Loading Cellpose model '{name}'"

    def call(self, op, *args, **kwargs):
        yield None
        return self.segment(*args, **kwargs)

    def cancel(self):
        pass


@pytest.fixture
def use_process(monkeypatch):
    monkeypatch.setattr(_cellpose, "ensure_weights", lambda name: iter(()))
    monkeypatch.setattr(_cellpose, "default_model", lambda: "cpsam_v2")

    def use(process):
        monkeypatch.setattr(cellpose_widget, "get_process", lambda: process)
        return process

    return use


def test_cellpose_panel(viewer, qtbot, use_process):
    use_process(FakeProcess(fake_segment))
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


def test_cellpose_panel_region(viewer, qtbot, use_process):
    calls = []

    def fake(vol, nuc=None, **params):
        calls.append(vol.shape)
        out = np.zeros(vol.shape, dtype=np.uint16)
        out[:, :2, :2] = 1  # inside the painted corner
        out[:, -2:, -2:] = 2  # outside the painted area
        return out

    use_process(FakeProcess(fake))
    w = CellposeWidget(viewer)
    qtbot.addWidget(w)
    w.image.setCurrentText("s GFP")
    w.add_region_layer()
    region = viewer.layers["s GFP region"]
    assert region.data.shape == (4, 8, 8) and region.mode == "paint"  # Z Y X
    assert list(region.scale) == [2.0, 0.5, 0.5]
    assert w.region.currentText() == "s GFP region"

    w.run()  # nothing painted yet
    assert "Nothing to segment" in w.status.text() and not calls

    w.z_first.setValue(1)
    w.z_last.setValue(2)
    # Paint an L-shape on plane 1 (as a brush stroke would): same bounding box as the
    # rectangle, bottom-right corner unpainted. It is copied to planes 1-2 only.
    yy, xx = np.nonzero(np.pad(np.ones((3, 4), bool), ((2, 3), (3, 1))))
    keep = ~((yy >= 3) & (xx >= 5))
    region.data_setitem((np.full(keep.sum(), 1), yy[keep], xx[keep]), 1)
    assert [int(region.data[z].sum()) for z in range(4)] == [0, 8, 8, 0]
    viewer.dims.set_current_step(0, 0)
    viewer.dims.set_current_step(1, 0)
    w.run()
    qtbot.waitUntil(lambda: "s GFP cellpose T0 P0 region" in viewer.layers, timeout=10000)
    assert calls == [(2, 3, 4)]  # Z 1..2, Y 2..4, X 3..6
    out = viewer.layers["s GFP cellpose T0 P0 region"]
    assert out.data.shape == (2, 3, 4)
    assert list(out.translate) == [2.0, 1.0, 1.5]  # 1 plane * 2 um, 2 px * 0.5, 3 px * 0.5
    assert set(np.unique(out.data)) == {0, 1}  # object 2 sits outside the paint
    assert "s GFP cellpose T0 P0 region" not in [w.region.itemText(i) for i in range(w.region.count())]

    qtbot.waitUntil(lambda: w._worker is None, timeout=10000)
    w.keep_inside.setChecked(False)
    w.scope.setCurrentText(SCOPE_ALL)
    w.run()
    qtbot.waitUntil(lambda: "s GFP cellpose all region" in viewer.layers, timeout=10000)
    out = viewer.layers["s GFP cellpose all region"]
    assert out.data.shape == (2, 3, 2, 3, 4)
    assert list(out.translate) == [0.0, 0.0, 2.0, 1.0, 1.5]
    assert set(np.unique(out.data)) == {0, 1, 2}


def test_cancel_stops_cellpose_at_once(viewer, qtbot, use_process, monkeypatch):
    from ibif_napari._process import CellposeProcess

    process = use_process(CellposeProcess(backend="_fake_cellpose"))
    reloads = []
    monkeypatch.setattr(CellposeWidget, "preload_model", lambda self, name=None: reloads.append(1))
    w = CellposeWidget(viewer)
    qtbot.addWidget(w)
    try:
        w.image.setCurrentText("s GFP")
        w.model.setEditText("slow")  # the stand-in sleeps for 60 s
        w.run()
        qtbot.waitUntil(lambda: w.status.text().startswith("Segmenting"), timeout=30000)
        child = process._proc
        t = time.monotonic()
        w.run()  # the Run button reads "Cancel" while running
        qtbot.waitUntil(lambda: w._worker is None, timeout=10000)
        assert time.monotonic() - t < 5
        assert not child.is_alive()
        assert w.run_btn.text() == "Run Cellpose"
        assert w.status.text().startswith("Cancelled")
        assert reloads == [1]
        assert not any("cellpose" in l.name for l in viewer.layers)

        w.model.setEditText("fast")  # and the next run works in a fresh process
        w.run()
        qtbot.waitUntil(lambda: any("cellpose" in l.name for l in viewer.layers), timeout=30000)
    finally:
        process.close()


def test_cellpose_panel_on_a_server(viewer, qtbot, use_process, monkeypatch, tmp_path):
    """End to end: the panel runs Cellpose on an ibif-napari-server (stand-in Cellpose)."""
    import threading

    from ibif_napari import _remote
    from ibif_napari._server import CellposeServer, Users, make_http_server

    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.delenv(var, raising=False)
    users = Users(tmp_path / "users.json")
    token = users.add("alice")
    app = CellposeServer(users, backend="_fake_cellpose", gpu=False,
                         model_names=lambda: ["cpsam_v2", "slow"], prepare=lambda n: iter(()))
    httpd = make_http_server(app, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(_remote, "CONFIG", tmp_path / "servers.json")
    _remote.save_servers([{"name": "Facility GPU", "url": f"http://127.0.0.1:{httpd.server_address[1]}",
                           "token": token}])
    use_process(FakeProcess(fake_segment))  # the local process, which must not be used
    w = CellposeWidget(viewer)
    qtbot.addWidget(w)
    try:
        assert [w.run_on.itemText(i) for i in range(w.run_on.count())] == ["This computer", "Facility GPU"]
        w.run_on.setCurrentText("Facility GPU")
        assert w.remote and not w.gpu.isEnabled()
        qtbot.waitUntil(lambda: w.status.text().startswith("Connected to Facility GPU as alice"), timeout=30000)
        assert [w.model.itemText(i) for i in range(w.model.count())] == ["cpsam_v2", "slow"]

        w.image.setCurrentText("s GFP")
        w.model.setCurrentText("cpsam_v2")
        viewer.dims.set_current_step(0, 1)
        viewer.dims.set_current_step(1, 0)
        w.run()
        assert not w.run_on.isEnabled()
        qtbot.waitUntil(lambda: "s GFP cellpose T1 P0" in viewer.layers, timeout=30000)
        assert viewer.layers["s GFP cellpose T1 P0"].data.shape == (4, 8, 8)
        qtbot.waitUntil(lambda: w._worker is None, timeout=10000)
        assert w.run_on.isEnabled()

        # Cancel on the server is immediate too.
        w.model.setCurrentText("slow")
        w.run()
        qtbot.waitUntil(lambda: w.status.text().startswith("Segmenting"), timeout=30000)
        t = time.monotonic()
        w.run()
        qtbot.waitUntil(lambda: w._worker is None, timeout=10000)
        assert time.monotonic() - t < 5
        assert w.status.text().startswith("Cancelled")
        qtbot.waitUntil(lambda: app.current is None, timeout=10000)

        w.run_on.setCurrentText("This computer")
        assert not w.remote and w.gpu.isEnabled()
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
