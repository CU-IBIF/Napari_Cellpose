import numpy as np
import pytest

pytest.importorskip("napari")

from napari.components import ViewerModel  # noqa: E402
from napari.utils._proxies import ReadOnlyWrapper  # noqa: E402
from napari.utils.interactions import (  # noqa: E402
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

from ibif_napari._draw3d import DrawingPlane  # noqa: E402


class Event:
    def __init__(self, type, position, view_direction, dims_displayed):
        self.type, self.position, self.view_direction = type, position, view_direction
        self.dims_displayed, self.is_dragging, self.modifiers, self.button = dims_displayed, type == "mouse_move", (), 1
        self.dims_point = position
        self.handled = False


def stroke(layer, viewer, points, direction):
    displayed = tuple(viewer.dims.displayed)
    mouse_press_callbacks(layer, ReadOnlyWrapper(Event("mouse_press", points[0], direction, displayed)))
    for p in points[1:]:
        mouse_move_callbacks(layer, ReadOnlyWrapper(Event("mouse_move", p, direction, displayed)))
    mouse_release_callbacks(layer, ReadOnlyWrapper(Event("mouse_release", points[-1], direction, displayed)))


@pytest.fixture
def viewer(qapp):
    v = ViewerModel()
    # T Z Y X image (2 µm planes, 0.5 µm pixels) and a 3D labels layer on the same grid
    v.add_image(np.random.rand(2, 10, 40, 40), scale=(1, 2, 0.5, 0.5), name="img")
    v.add_labels(np.zeros((10, 40, 40), np.uint8), scale=(2, 0.5, 0.5), name="region")
    v.dims.ndisplay = 3
    return v


def test_napari_alone_only_repaints_existing_labels_in_3d(qapp):
    # Why the drawing plane is needed: napari's brush in 3D skips empty voxels.
    v = ViewerModel()
    labels = v.add_labels(np.zeros((10, 40, 40), np.uint8), name="region")
    v.dims.ndisplay = 3
    labels.mode, labels.brush_size, labels.selected_label = "paint", 4, 3
    labels._draw(3, np.array([4, 20, 20]), np.array([4, 20, 20]))
    assert labels.data.max() == 0


def test_drawing_plane_paints_on_the_chosen_z(viewer):
    labels = viewer.layers["region"]
    labels.mode, labels.brush_size, labels.selected_label = "paint", 4, 3
    plane = DrawingPlane(viewer)
    plane.enable(z=8.0)  # world µm -> plane 4
    img = viewer.layers["img"]
    assert img.depiction == "plane" and tuple(img.plane.normal) == (1, 0, 0)
    assert img.plane.position[0] == pytest.approx(4.0)

    # A slanted ray from above, then a short drag: paints only on plane 4.
    direction = (0, 1.0, 0.2, 0.1)
    stroke(labels, viewer, [(0, -2.0, 9.0, 9.0), (0, -2.0, 9.0, 11.0)], direction)
    painted = np.nonzero(labels.data)
    assert set(painted[0]) == {4}
    # The ray from (-2, 9, 9) along (1, .2, .1) meets z = 8 µm after t = 10 -> y 11, x 10 µm.
    assert labels.data[4, 22, 20] == 3

    # Erasing works on the plane too; moving the plane changes the depth.
    labels.mode = "erase"
    stroke(labels, viewer, [(0, -2.0, 9.0, 9.0)], direction)
    assert labels.data[4, 22, 20] == 0
    labels.mode = "paint"
    plane.set_z(2.0)
    assert img.plane.position[0] == pytest.approx(1.0)
    stroke(labels, viewer, [(0, -2.0, 5.0, 5.0)], (0, 1.0, 0, 0))
    assert labels.data[1, 10, 10] == 3

    # Rays that miss the layer or run along the plane paint nothing.
    before = labels.data.copy()
    stroke(labels, viewer, [(0, -2.0, 500.0, 500.0)], (0, 1.0, 0, 0))
    stroke(labels, viewer, [(0, -2.0, 5.0, 5.0)], (0, 0, 1.0, 0))
    assert np.array_equal(labels.data, before)

    # One stroke is one undo step.
    labels.undo()
    assert labels.data[1, 10, 10] == 0 and labels.data[4].max() == 3

    # Off: the image is shown as before and napari's normal brush is back.
    plane.disable()
    assert img.depiction == "volume"
    assert plane._drag not in labels.mouse_drag_callbacks


def test_plane_only_applies_in_3d_and_to_its_viewer(viewer):
    labels = viewer.layers["region"]
    labels.mode, labels.brush_size, labels.selected_label = "paint", 2, 1
    plane = DrawingPlane(viewer)
    plane.enable(z=4.0)
    other = ViewerModel()
    other.add_labels(np.zeros((10, 40, 40), np.uint8), name="elsewhere")
    assert not plane.owns(other.layers["elsewhere"])
    viewer.dims.ndisplay = 2  # back to 2D: napari's usual painting at the slider position
    assert not plane.owns(labels)
    plane.disable()


def test_dimensions_panel_drawing_plane(qtbot):
    from ibif_napari._dims_widget import DimensionsWidget
    from ibif_napari._reader import layer_data

    v = ViewerModel()
    data = np.random.randint(0, 100, (2, 6, 1, 32, 32), dtype=np.uint16)  # T Z C Y X
    (ld,) = layer_data(data, {"T": 2, "Z": 6, "C": 1, "Y": 32, "X": 32}, ["GFP"], [520], (2.0, 0.5, 0.5),
                       None, "s", "s")
    v._add_layer_from_data(*ld)
    labels = v.add_labels(np.zeros((6, 32, 32), np.uint8), scale=(2, 0.5, 0.5), name="cells")
    w = DimensionsWidget(v)
    qtbot.addWidget(w)
    assert not w.plane_slider.isEnabled()

    w.draw_plane.setChecked(True)
    assert v.dims.ndisplay == 3 and w.plane_slider.isEnabled()
    assert w.plane_slider.maximum() == 5
    w.plane_slider.setValue(4)
    assert w.plane.z == pytest.approx(8.0) and w.plane_label.text() == "8.00 µm (5/6)"
    assert v.layers["s GFP"].depiction == "plane" and v.layers["s GFP"].plane.position[0] == pytest.approx(4)

    labels.mode, labels.brush_size, labels.selected_label = "paint", 3, 2
    stroke(labels, v, [(0, -1.0, 8.0, 8.0)], (0, 1.0, 0, 0))
    assert set(np.nonzero(labels.data)[0]) == {4} and labels.data[4, 16, 16] == 2

    w.draw_plane.setChecked(False)
    assert v.layers["s GFP"].depiction == "volume" and not w.plane_slider.isEnabled()


def test_cellpose_region_drawn_in_3d_fills_the_z_range(viewer):
    from ibif_napari._region import ZExtendedRegion

    region = viewer.layers["region"]
    sync = ZExtendedRegion(region, (2, 6))
    region.mode, region.brush_size, region.selected_label = "paint", 3, 1
    plane = DrawingPlane(viewer)
    plane.enable(z=8.0)  # plane 4, inside the range
    stroke(region, viewer, [(0, -2.0, 10.0, 10.0)], (0, 1.0, 0, 0))
    assert [int((region.data[z] != 0).sum() > 0) for z in range(10)] == [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    plane.set_z(16.0)  # plane 8: outside the range, so the range grows to include it
    stroke(region, viewer, [(0, -2.0, 4.0, 4.0)], (0, 1.0, 0, 0))
    assert sync.z_range == (2, 8)
    plane.disable()
