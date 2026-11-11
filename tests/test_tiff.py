"""TIFFs in any order of dimensions: metadata-described files open directly, plain
multipage files with a chosen dimension order; large files are read plane by plane."""
import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from ibif_napari import _tiff  # noqa: E402

T, C, Z, Y, X = 2, 3, 4, 5, 6


def truth():
    """A TCZYX array whose every value encodes its position (t, c, z, y, x)."""
    t, c, z, y, x = np.indices((T, C, Z, Y, X))
    return (t * 10000 + c * 1000 + z * 100 + y * 10 + x).astype(np.uint16)


def stored(order):
    """The truth stored in `order` (fastest first), as a flat sequence of values."""
    slow_first = order[::-1]
    return np.ascontiguousarray(truth().transpose(["TCZYX".index(a) for a in slow_first]))


def layers_by_channel(out):
    """{channel index: (data, kwargs)} from the layer data, data as (T, Z, Y, X)."""
    (data, kwargs, kind), = out
    axis = kwargs["channel_axis"]
    return {c: np.take(np.asarray(data), c, axis=axis) for c in range(data.shape[axis])}, kwargs


@pytest.mark.parametrize("order", ["XYZCT", "XYCZT", "XYTCZ", "XYZTC", "CXYZT", "XYCTZ"])
def test_plain_multipage_in_any_order(tmp_path, order):
    path = tmp_path / "plain.tif"
    data = stored(order)
    # A plain multipage TIFF: just planes, no description of how they are arranged.
    # (CXYZT: channels interleaved pixel by pixel, so a plane is X*C values wide.)
    pages = data.reshape(-1, Y, X) if order.startswith("XY") else data.reshape(T * Z, Y, X * C)
    tifffile.imwrite(path, pages, metadata=None)
    info = _tiff.inspect(path)[0]
    assert not info.known
    layout = _tiff.Layout.parse("-".join(order), {"T": T, "C": C, "Z": Z, "Y": Y, "X": X})
    out = _tiff.read_tiff(path, layout=layout)
    channels, kwargs = layers_by_channel(out)
    assert kwargs["metadata"][0]["ibif_axes"] == "TZYX"
    for c, got in channels.items():
        assert np.array_equal(got, truth()[:, c])
    assert kwargs["metadata"][0]["ibif_tiff_layout"] == layout.to_dict()


def test_dialog_asked_only_when_the_file_does_not_say(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, stored("XYZCT").reshape(-1, Y, X), metadata=None)
    asked = []

    def ask(info):
        asked.append(info)
        assert info.n_planes == T * C * Z and info.layout.order == "XYZ"
        return _tiff.Layout("XYZCT", {"X": X, "Y": Y, "Z": Z, "C": C, "T": T})

    out = _tiff.read_tiff(path, ask=ask)
    assert len(asked) == 1
    assert np.array_equal(layers_by_channel(out)[0][2], truth()[:, 2])
    assert _tiff.read_tiff(path, ask=lambda info: None) == [(None,)]  # cancelled
    # An ImageJ hyperstack says how it is arranged: opened without asking.
    hyper = tmp_path / "hyper.tif"
    tifffile.imwrite(hyper, truth().transpose(0, 2, 1, 3, 4), imagej=True, resolution=(2.0, 2.0),
                     metadata={"axes": "TZCYX", "spacing": 0.7, "unit": "um", "finterval": 5})
    out = _tiff.read_tiff(hyper, ask=lambda info: pytest.fail("asked"))
    channels, kwargs = layers_by_channel(out)
    assert all(np.array_equal(channels[c], truth()[:, c]) for c in range(C))
    assert kwargs["scale"] == pytest.approx([1, 0.7, 0.5, 0.5])
    assert kwargs["metadata"][0]["ibif_time_interval_s"] == pytest.approx(5)
    # ... unless asked to choose (metadata that is wrong): the reader "choose the order".
    out = _tiff.read_tiff(hyper, ask_always=True, ask=lambda info: info.layout)
    assert np.array_equal(layers_by_channel(out)[0][1], truth()[:, 1])


def test_ome_and_tifffile_metadata(tmp_path):
    path = tmp_path / "x.ome.tif"
    tifffile.imwrite(path, truth(), ome=True, metadata={"axes": "TCZYX", "PhysicalSizeX": 0.3,
                                                         "PhysicalSizeY": 0.3, "PhysicalSizeZ": 1.5,
                                                         "Channel": {"Name": ["DAPI", "GFP", "RFP"]}})
    out = _tiff.read_tiff(path, ask=lambda info: pytest.fail("asked"))
    channels, kwargs = layers_by_channel(out)
    assert all(np.array_equal(channels[c], truth()[:, c]) for c in range(C))
    assert [n.split()[-1] for n in kwargs["name"]] == ["DAPI", "GFP", "RFP"]
    assert kwargs["scale"] == pytest.approx([1, 1.5, 0.3, 0.3])
    shaped = tmp_path / "shaped.tif"
    tifffile.imwrite(shaped, truth()[0], metadata={"axes": "CZYX"})
    channels, kwargs = layers_by_channel(_tiff.read_tiff(shaped, ask=lambda info: pytest.fail("asked")))
    assert np.array_equal(channels[2], truth()[0, 2])


def test_large_files_are_read_plane_by_plane(tmp_path, monkeypatch):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, stored("XYCZT").reshape(-1, Y, X), metadata=None)
    monkeypatch.setattr(_tiff, "LAZY_BYTES", 10)
    layout = _tiff.Layout.parse("XY-C-Z-T", {"T": T, "C": C, "Z": Z, "Y": Y, "X": X})
    (data, kwargs, _), = _tiff.read_tiff(path, layout=layout)
    import dask.array as da

    assert isinstance(data, da.Array)
    assert np.array_equal(np.asarray(data[1, 2]), truth()[1, 2])


def test_orders_and_sizes():
    assert _tiff.clean_order("xy-z-c-t") == "XYZCT"
    assert _tiff.clean_order("C-Z-XY-T") == "CZXYT"
    for bad in ("XYZZ", "XZ", "XYQ"):
        with pytest.raises(ValueError):
            _tiff.clean_order(bad)
    assert _tiff.fill_sizes("XYZC", {"X": 6, "Y": 5, "C": 3}, 6 * 5 * 3 * 4)["Z"] == 4


def test_reader_and_dialog(qtbot, tmp_path):
    from ibif_napari._tiff_dialog import DimensionOrderDialog

    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, stored("XYZCT").reshape(-1, Y, X), metadata=None)
    assert _tiff.napari_get_reader(str(path)) is _tiff.read_tiff
    assert _tiff.napari_get_reader("a.png") is None
    info = _tiff.inspect(path)[0]
    dialog = DimensionOrderDialog(info)
    qtbot.addWidget(dialog)
    assert dialog.chosen().order == "XYZ" and dialog.chosen().total == info.total
    dialog.set_order("XY-Z-C-T", Z=4, C=3)  # T is worked out from the rest
    assert dialog.chosen().sizes == {"X": X, "Y": Y, "Z": Z, "C": C, "T": T}
    assert "✓" in dialog.check.text()
    dialog.set_order("XYZCT", Z=5, C=3, T=2)
    assert "✓" not in dialog.check.text()
