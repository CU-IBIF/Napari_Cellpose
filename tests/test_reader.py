import numpy as np
import pytest

from ibif_napari._reader import layer_data, napari_get_reader


def test_get_reader_only_accepts_nd2():
    assert napari_get_reader("a.nd2") is not None
    assert napari_get_reader(["a.ND2"]) is not None
    assert napari_get_reader("a.tif") is None
    assert napari_get_reader(["a.nd2", "b.nd2"]) is None


def test_multichannel_stack_is_split_into_layers():
    data = np.random.randint(0, 4000, (3, 5, 2, 16, 16), dtype=np.uint16)  # T Z C Y X
    sizes = {"T": 3, "Z": 5, "C": 2, "Y": 16, "X": 16}
    (layer,) = layer_data(data, sizes, ["DAPI", "GFP"], [450, 520], (2.0, 0.5, 0.5), 30.0, "s", "s.nd2")
    arr, kw, kind = layer
    assert kind == "image"
    assert kw["channel_axis"] == 2
    assert kw["name"] == ["s DAPI", "s GFP"]
    assert kw["colormap"] == ["blue", "green"]
    assert kw["scale"] == [1.0, 2.0, 0.5, 0.5]
    assert kw["metadata"][0]["ibif_axes"] == "TZYX"
    assert kw["metadata"][1]["ibif_channel"] == "GFP"
    assert kw["metadata"][0]["ibif_time_interval_s"] == 30.0
    lo, hi = kw["contrast_limits"][0]
    assert lo < hi


def test_missing_channel_metadata_gets_default_names():
    data = np.zeros((3, 8, 8), dtype=np.uint16)
    (layer,) = layer_data(data, {"C": 3, "Y": 8, "X": 8}, [], [], (None, 0.1, 0.1), None, "s", "s")
    kw = layer[1]
    assert kw["name"] == ["s C0", "s C1", "s C2"]
    assert kw["contrast_limits"][0] == [0.0, 1.0]


def test_single_channel_and_rgb():
    data = np.zeros((4, 8, 8), dtype=np.uint8)
    (layer,) = layer_data(data, {"Z": 4, "Y": 8, "X": 8}, ["BF"], [None], (1, 1, 1), None, "s", "s")
    assert layer[1]["name"] == "s BF"
    assert layer[1]["colormap"] == "gray"

    rgb = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    (layer,) = layer_data(rgb, {"P": 2, "Y": 8, "X": 8, "S": 3}, [], [], (1, 1, 1), None, "s", "s")
    assert layer[1]["rgb"] is True
    assert layer[1]["metadata"]["ibif_axes"] == "PYX"


def test_axes_must_match_data():
    with pytest.raises(ValueError):
        layer_data(np.zeros((2, 2)), {"Z": 2, "Y": 2, "X": 2}, [], [], (1, 1, 1), None, "s", "s")
