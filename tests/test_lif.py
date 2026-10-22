import numpy as np
import pytest

pytest.importorskip("bioio_lif")

from _lif_writer import write_lif  # noqa: E402

from ibif_napari._bioio import NO_LAYERS  # noqa: E402
from ibif_napari._lif import read_lif  # noqa: E402
from ibif_napari._reader import napari_get_reader  # noqa: E402


def tzcyx(t, z, c, y, x, offset=0):
    return ((np.arange(t * z * c * y * x).reshape(t, z, c, y, x) + offset) % 60000).astype(np.uint16)


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "project.lif"
    write_lif(
        path,
        [
            ("Series001", tzcyx(2, 3, 2, 16, 20), ["Green", "Red"], 0.2, 1.5, 30.0),
            ("Series002", tzcyx(1, 1, 1, 12, 12), ["Gray"], 0.5, 1, 1),
            ("Series003", tzcyx(2, 3, 2, 16, 20, offset=7), ["Green", "Red"], 0.2, 1.5, 30.0),
        ],
    )
    return path


def test_get_reader_accepts_lif():
    assert napari_get_reader("a.LIF") is read_lif


def test_single_image_opens_without_asking(tmp_path):
    path = tmp_path / "one.lif"
    write_lif(path, [("Series001", tzcyx(1, 4, 2, 16, 20), ["Blue", "Magenta"], 0.3, 2.0, 1)])

    def never(*_):
        raise AssertionError("should not ask")

    (layer,) = read_lif(path, choose=never)
    data, kw, _ = layer
    assert kw["name"] == ["one Blue", "one Magenta"]
    assert kw["colormap"] == ["blue", "magenta"]
    assert kw["metadata"][0]["ibif_axes"] == "ZYX"
    assert kw["scale"] == pytest.approx([2.0, 0.3, 0.3])
    assert np.array_equal(np.asarray(data), tzcyx(1, 4, 2, 16, 20)[0].transpose(1, 0, 2, 3))


def test_chooser_lists_images_and_opens_the_picked_ones(project):
    seen = {}

    def choose(title, scenes):
        seen["title"] = title
        seen["rows"] = [(s.name, s.summary) for s in scenes]
        return [1, 2], False

    layers = read_lif(project, choose=choose)
    assert seen["title"] == "project"
    assert seen["rows"] == [
        ("Series001", "T 2 · C 2 · Z 3 · 16×20"),
        ("Series002", "12×12"),
        ("Series003", "T 2 · C 2 · Z 3 · 16×20"),
    ]
    assert [l[1]["name"] for l in layers] == [
        ["project Series002 Gray"],
        ["project Series003 Green", "project Series003 Red"],
    ]
    assert layers[0][1]["colormap"] == ["gray"]
    assert layers[1][1]["metadata"][0]["ibif_axes"] == "TZYX"


def test_same_size_images_can_be_stacked_as_positions(project):
    (layer,) = read_lif(project, choose=lambda title, scenes: ([0, 2], True))
    data, kw, _ = layer
    assert data.shape == (2, 2, 2, 3, 16, 20)  # T P C Z Y X
    assert kw["metadata"][0]["ibif_axes"] == "TPZYX"
    assert int(data[0, 1, 0, 0, 0, 0]) == 7  # second image


def test_cancelled_chooser_opens_nothing(project):
    assert read_lif(project, choose=lambda title, scenes: None) == NO_LAYERS


def test_chooser_dialog(project, qtbot):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QDialogButtonBox

    from bioio_lif import Reader

    from ibif_napari._bioio import read_scene
    from ibif_napari._chooser import ImageChooser

    reader = Reader(str(project))
    scenes = [read_scene(reader, i) for i in range(3)]
    dlg = ImageChooser("project", scenes)
    qtbot.addWidget(dlg)
    assert dlg.selected() == [0]
    assert not dlg.stack.isEnabled()

    dlg.list.item(2).setCheckState(Qt.Checked)  # same size as the first
    assert dlg.stack.isEnabled()
    dlg.list.item(1).setCheckState(Qt.Checked)
    assert not dlg.stack.isEnabled()

    dlg.filter.setText("002")
    dlg._set_all(Qt.Unchecked)
    assert dlg.selected() == [0, 2]
    dlg.filter.setText("")
    dlg._set_all(Qt.Unchecked)
    assert dlg.selected() == []
    assert not dlg.buttons.button(QDialogButtonBox.Open).isEnabled()
