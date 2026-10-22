import types
from datetime import timedelta
from xml.etree import ElementTree as ET

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from ibif_napari._czi import read_czi
from ibif_napari._reader import napari_get_reader


def test_get_reader_accepts_czi():
    assert napari_get_reader("a.CZI") is read_czi


@pytest.fixture
def write_czi(tmp_path):
    czi = pytest.importorskip("pylibCZIrw.czi")
    pytest.importorskip("bioio_czi")

    def write(name, tiles, channels, t=1, z=1):
        """tiles: list of x offsets; each tile is 32x40 and filled with t/z/c/tile codes."""
        path = tmp_path / name
        with czi.create_czi(str(path)) as w:
            for ti in range(t):
                for zi in range(z):
                    for c in range(len(channels)):
                        for k, x in enumerate(tiles):
                            value = 1000 * c + 100 * ti + 10 * zi + k
                            plane = np.full((32, 40, 1), value, np.uint16)
                            w.write(plane, plane={"T": ti, "Z": zi, "C": c}, location=(x, 0))
            w.write_metadata(
                channel_names=dict(enumerate(channels)),
                scale_x=0.2e-6,
                scale_y=0.2e-6,
                scale_z=1e-6,
            )
        return path

    return write


def test_real_czi_time_z_channels(write_czi):
    path = write_czi("tz.czi", [0], ["DAPI", "AF488"], t=3, z=4)
    (layer,) = napari_get_reader(str(path))(str(path))
    data, kw, kind = layer
    assert kind == "image"
    assert kw["name"] == ["tz DAPI", "tz AF488"]
    assert kw["channel_axis"] == 1
    assert kw["metadata"][0]["ibif_axes"] == "TZYX"
    assert kw["scale"] == pytest.approx([1.0, 1.0, 0.2, 0.2])
    assert data.shape == (3, 2, 4, 32, 40)
    assert int(data[2, 1, 3, 0, 0]) == 1000 + 200 + 30


def test_real_czi_mosaic_single_plane_is_2d(write_czi):
    path = write_czi("tiles.czi", [0, 40], ["GFP"])
    (layer,) = read_czi(path)
    data, kw, _ = layer
    assert kw["metadata"][0]["ibif_axes"] == "YX"  # T and Z of length 1 dropped
    assert data.shape == (1, 32, 80)  # channel axis kept for splitting, tiles stitched
    row = np.asarray(data[0, 0])
    assert row[0] == 0 and row[79] == 1


class FakeReader:
    """Two equal scenes, T=2, C=2, Z=3, with Zeiss channel metadata."""

    def __init__(self, shapes=((2, 2, 3, 8, 8), (2, 2, 3, 8, 8))):
        self.scenes = tuple(f"Scene {i}" for i in range(len(shapes)))
        self._data = [
            da.from_array(np.full(s, i, np.uint16)) for i, s in enumerate(shapes)
        ]
        self._scene = 0
        self.channel_names = ["DAPI", "mCherry"]
        self.physical_pixel_sizes = types.SimpleNamespace(Z=0.5, Y=0.1, X=0.1)
        self.time_interval = timedelta(seconds=90)
        self.metadata = ET.fromstring(
            "<Root><Image><Dimensions><Channels>"
            "<Channel Name='DAPI'><EmissionWavelength>461</EmissionWavelength></Channel>"
            "<Channel Name='mCherry'><EmissionWavelength>610</EmissionWavelength></Channel>"
            "</Channels></Dimensions></Image></Root>"
        )

    def set_scene(self, i):
        self._scene = i

    @property
    def dims(self):
        return types.SimpleNamespace(order="TCZYX")

    @property
    def xarray_dask_data(self):
        return xr.DataArray(self._data[self._scene], dims=list("TCZYX"))


def test_equal_scenes_become_positions():
    (layer,) = read_czi("multi.czi", reader=FakeReader())
    data, kw, _ = layer
    assert data.shape == (2, 2, 2, 3, 8, 8)  # T P C Z Y X
    assert kw["channel_axis"] == 2
    assert kw["metadata"][0]["ibif_axes"] == "TPZYX"
    assert kw["metadata"][0]["ibif_time_interval_s"] == 90
    assert kw["colormap"] == ["blue", "magenta"]
    assert int(data[0, 1, 0, 0, 0, 0]) == 1  # second scene


def test_unequal_scenes_open_separately():
    layers = read_czi("multi.czi", reader=FakeReader(((1, 2, 3, 8, 8), (1, 2, 3, 6, 6))))
    assert [l[1]["name"] for l in layers] == [
        ["multi Scene 0 DAPI", "multi Scene 0 mCherry"],
        ["multi Scene 1 DAPI", "multi Scene 1 mCherry"],
    ]
    assert layers[1][0].shape == (2, 3, 6, 6)
