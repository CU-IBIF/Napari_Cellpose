import json
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("napari")

from napari.components import ViewerModel  # noqa: E402

from ibif_napari import _measure, _workspace  # noqa: E402
from ibif_napari._reader import napari_get_reader  # noqa: E402


def write_czi(path):
    czi = pytest.importorskip("pylibCZIrw.czi")
    pytest.importorskip("bioio_czi")
    with czi.create_czi(str(path)) as w:
        for z in range(3):
            for c in range(2):
                w.write(np.full((16, 20, 1), 100 * c + z, np.uint16), plane={"Z": z, "C": c})
        w.write_metadata(channel_names={0: "DAPI", 1: "GFP"}, scale_x=0.5e-6, scale_y=0.5e-6, scale_z=2e-6)
    return path


def session(data_dir):
    """A session like a real one: two CZI channels, a max projection, nuclei labels,
    a drawn region, shapes, points, measurement tables and a 3D view."""
    v = ViewerModel()
    src = write_czi(data_dir / "cells.czi")
    for ld in napari_get_reader(str(src))(str(src)):
        v._add_layer_from_data(*ld)
    v.layers["cells GFP"].colormap = "magenta"
    v.layers["cells GFP"].contrast_limits = (10, 150)
    v.layers["cells DAPI"].visible = False
    v.add_image(np.arange(320, dtype=np.float32).reshape(16, 20), name="GFP MIP", scale=(0.5, 0.5))
    nuclei = np.zeros((3, 16, 20), np.uint16)
    nuclei[:, 2:6, 2:6] = 1
    nuclei[:, 9:13, 10:15] = 2
    v.add_labels(nuclei, name="nuclei", scale=(2, 0.5, 0.5), opacity=0.6,
                 metadata={"ibif_axes": "ZYX", "ibif_voxel_um": {"Z": 2, "Y": 0.5, "X": 0.5}, "note": object()})
    v.add_shapes([np.array([[1, 1], [1, 5], [4, 5], [4, 1]])], shape_type="rectangle", name="roi", edge_width=2)
    v.add_points(np.array([[2.0, 3.0], [5.0, 6.0]]), name="marks", size=4)
    tables = {"nuclei": _measure.measure_layer(v, v.layers["nuclei"], [v.layers["cells GFP"]])}
    _measure.remember(v, tables)
    v.dims.ndisplay = 3
    v.dims.set_point(0, 2.0)
    return v, src


def snapshot(v):
    return {l.name: np.asarray(l.data) for l in v.layers if l._type_string in ("image", "labels")}


def compare(a, b, data=None):
    data = data if data is not None else snapshot(a)
    assert [l.name for l in a.layers] == [l.name for l in b.layers]
    for la, lb in zip(a.layers, b.layers):
        assert type(la) is type(lb), la.name
        assert np.array_equal(np.asarray(la.scale), np.asarray(lb.scale)), la.name
        assert la.visible == lb.visible and la.opacity == pytest.approx(lb.opacity), la.name
        if la._type_string in ("image", "labels"):
            assert np.array_equal(data[la.name], np.asarray(lb.data)), la.name
        if la._type_string == "image":
            assert la.colormap.name == lb.colormap.name and tuple(la.contrast_limits) == pytest.approx(tuple(lb.contrast_limits))
    assert np.array_equal(a.layers["roi"].data[0], b.layers["roi"].data[0])
    assert np.array_equal(a.layers["marks"].data, b.layers["marks"].data)
    assert b.dims.ndisplay == 3 and b.dims.point[0] == pytest.approx(2.0)


def test_round_trip_references_the_microscope_file(tmp_path):
    (tmp_path / "data").mkdir()
    v, src = session(tmp_path / "data")
    folder, notes = _workspace.save_workspace(v, tmp_path / "data" / "experiment")
    assert folder.name == "experiment.ibif" and notes == []
    state = json.loads((folder / "workspace.json").read_text())
    gfp = next(e for e in state["layers"] if e["name"] == "cells GFP")
    assert gfp["source"] == str(src) and gfp["source_rel"] == "../cells.czi" and "file" not in gfp
    assert "note" not in next(e for e in state["layers"] if e["name"] == "nuclei")["metadata"]  # not JSON
    assert sorted(p.name for p in (folder / "layers").iterdir()) == ["02 GFP MIP.tif", "03 nuclei.tif"]

    w = ViewerModel()
    assert _workspace.load_workspace(w, folder) == []
    compare(v, w)
    assert w.layers["cells GFP"].metadata["ibif_source"] == str(src)
    restored = _measure.tables_of(w)["nuclei"]
    pd.testing.assert_frame_equal(restored, _measure.tables_of(v)["nuclei"], check_dtype=False)

    # Moved somewhere else together with the data: found through the relative path.
    moved = tmp_path / "elsewhere"
    shutil.move(tmp_path / "data", moved)
    w2 = ViewerModel()
    assert _workspace.load_workspace(w2, moved / "experiment.ibif") == []
    assert w2.layers["cells GFP"].metadata["ibif_source"] == str(moved / "cells.czi")


def test_copies_for_sharing_and_missing_files(tmp_path):
    (tmp_path / "data").mkdir()
    v, src = session(tmp_path / "data")
    shared, _ = _workspace.save_workspace(v, tmp_path / "shared", include_images=True)
    plain, _ = _workspace.save_workspace(v, tmp_path / "plain.ibif")
    expected = snapshot(v)
    src.unlink()  # the colleague does not have the microscope file

    w = ViewerModel()
    assert _workspace.load_workspace(w, shared / "workspace.json") == []
    compare(v, w, expected)
    w2 = ViewerModel()
    notes = _workspace.load_workspace(w2, plain)
    assert len(notes) == 2 and all("data not found" in n for n in notes)
    assert "cells GFP" not in w2.layers and "nuclei" in w2.layers


def test_large_images_are_copied_plane_by_plane(tmp_path, monkeypatch):
    import dask.array as da

    monkeypatch.setattr(_workspace, "LARGE", 10)  # treat everything as large
    data = da.from_array(np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5), chunks=(1, 1, 4, 5))
    path = _workspace.write_tiff(tmp_path / "big.tif", data)
    back = _workspace.read_tiff(path)
    assert isinstance(back, np.memmap) and np.array_equal(back, np.asarray(data))


def test_save_layers_as_files(tmp_path):
    (tmp_path / "data").mkdir()
    v, _ = session(tmp_path / "data")
    paths = _workspace.save_layers([v.layers["nuclei"], v.layers["roi"], v.layers["marks"]], tmp_path / "out")
    assert sorted(p.name for p in paths) == ["marks.csv", "nuclei.tif", "roi.csv"]
    import tifffile

    assert np.array_equal(tifffile.imread(tmp_path / "out" / "nuclei.tif"), v.layers["nuclei"].data)
    roi = pd.read_csv(tmp_path / "out" / "roi.csv")
    assert list(roi.columns) == ["index", "shape-type", "vertex-index", "axis-0", "axis-1"] and len(roi) == 4
    assert pd.read_csv(tmp_path / "out" / "marks.csv").shape == (2, 2)


def test_workspace_panel_and_drag_and_drop(tmp_path, qtbot, monkeypatch):
    from ibif_napari._measure_widget import MeasureWidget
    from ibif_napari._workspace_widget import WorkspaceWidget

    (tmp_path / "data").mkdir()
    v, _ = session(tmp_path / "data")
    w = WorkspaceWidget(v)
    qtbot.addWidget(w)
    w.save(str(tmp_path / "panel"))
    qtbot.waitUntil(lambda: w._worker is None, timeout=30000)
    assert w.status.text() == f"Saved workspace {tmp_path / 'panel.ibif'}."

    v.layers.selection = {v.layers["nuclei"]}
    w.save_selected(str(tmp_path / "files"))
    qtbot.waitUntil(lambda: w._worker is None, timeout=30000)
    assert (tmp_path / "files" / "nuclei.tif").exists()

    other = ViewerModel()
    panel = MeasureWidget(other)  # shows the tables of a reopened workspace
    qtbot.addWidget(panel)
    w2 = WorkspaceWidget(other)
    qtbot.addWidget(w2)
    assert w2.open(str(tmp_path / "panel.ibif"), replace=True) == []
    assert "nuclei" in other.layers and panel.model.rowCount() == 2

    # Dropping a .ibif folder on napari adds it to the current viewer.
    reader = _workspace.napari_get_reader(str(tmp_path / "panel.ibif"))
    assert reader is not None and _workspace.napari_get_reader(str(tmp_path / "files")) is None
    third = ViewerModel()
    import napari

    monkeypatch.setattr(napari, "current_viewer", lambda: third)
    assert reader(str(tmp_path / "panel.ibif")) == [(None,)]
    assert "nuclei" in third.layers


def test_lif_image_reopens_without_asking(tmp_path):
    pytest.importorskip("bioio_lif")
    from _lif_writer import write_lif

    from ibif_napari._lif import read_lif

    src = tmp_path / "project.lif"
    a = np.arange(1 * 2 * 1 * 8 * 8, dtype=np.uint16).reshape(1, 2, 1, 8, 8)
    write_lif(src, [("Series001", a, ["Green"], 0.5, 1.0, 1), ("Series002", a + 7, ["Red"], 0.5, 1.0, 1)])
    v = ViewerModel()
    for ld in read_lif(src, choose=lambda title, scenes: ([1], False)):  # only the second image
        v._add_layer_from_data(*ld)
    assert [l.name for l in v.layers] == ["project Red"]
    folder, _ = _workspace.save_workspace(v, tmp_path / "w")
    w = ViewerModel()
    assert _workspace.load_workspace(w, folder) == []
    assert [l.name for l in w.layers] == ["project Red"]
    assert np.array_equal(np.asarray(w.layers["project Red"].data), np.asarray(v.layers["project Red"].data))


def test_tiff_reopens_with_its_dimension_order(tmp_path):
    """A plain multipage TIFF opened with a chosen order (XY-Z-C-T) reopens the same way,
    without asking again."""
    import tifffile

    from ibif_napari._tiff import Layout, read_tiff

    czyx = np.arange(2 * 3 * 8 * 8, dtype=np.uint16).reshape(2, 3, 8, 8)  # C Z Y X
    src = tmp_path / "plain.tif"
    tifffile.imwrite(src, czyx.reshape(6, 8, 8), metadata=None)  # planes: all Z of C1, then C2
    v = ViewerModel()
    for ld in read_tiff(src, layout=Layout.parse("XY-Z-C", {"X": 8, "Y": 8, "Z": 3, "C": 2})):
        v._add_layer_from_data(*ld)
    assert [l.data.shape for l in v.layers] == [(3, 8, 8), (3, 8, 8)]
    folder, _ = _workspace.save_workspace(v, tmp_path / "w")
    w = ViewerModel()
    assert _workspace.load_workspace(w, folder) == []
    assert [l.name for l in w.layers] == [l.name for l in v.layers]
    assert np.array_equal(np.asarray(w.layers[1].data), czyx[1])
