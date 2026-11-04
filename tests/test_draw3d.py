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
    assert w.plane.mode == "surface" and w.brush_mode.currentData() == "surface"  # on by default

    w.set_brush_mode("plane")
    assert v.dims.ndisplay == 3 and w.plane_slider.isEnabled()
    assert w.plane_slider.maximum() == 5
    w.plane_slider.setValue(4)
    assert w.plane.z == pytest.approx(8.0) and w.plane_label.text() == "8.00 µm (5/6)"
    assert v.layers["s GFP"].depiction == "plane" and v.layers["s GFP"].plane.position[0] == pytest.approx(4)

    labels.mode, labels.brush_size, labels.selected_label = "paint", 3, 2
    stroke(labels, v, [(0, -1.0, 8.0, 8.0)], (0, 1.0, 0, 0))
    assert set(np.nonzero(labels.data)[0]) == {4} and labels.data[4, 16, 16] == 2

    w.set_brush_mode("surface")
    assert v.layers["s GFP"].depiction == "volume" and not w.plane_slider.isEnabled()
    w.set_brush_mode("off")
    assert w.plane._drag not in labels.mouse_drag_callbacks


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


# ---- the brush on the image (default) ------------------------------------------------------


@pytest.fixture
def blob(qapp):
    """A dark Z Y X stack (2 µm planes, 0.5 µm pixels) with a bright spot on plane 6 and a
    dimmer one on plane 2, and an empty 3D labels layer on the same grid."""
    v = ViewerModel()
    img = np.zeros((10, 40, 40), np.float32)
    img[6, 16:25, 16:25] = 1.0
    img[2, 16:25, 16:25] = 0.6
    v.add_image(img, scale=(2, 0.5, 0.5), name="img", contrast_limits=(0, 1))
    v.add_labels(np.zeros((10, 40, 40), np.uint8), scale=(2, 0.5, 0.5), name="cells")
    v.dims.ndisplay = 3
    return v


def test_brush_paints_a_ball_where_the_ray_meets_the_signal(blob):
    from ibif_napari._draw3d import brush_for

    labels = blob.layers["cells"]
    labels.mode, labels.brush_size, labels.selected_label = "paint", 8, 5
    brush = brush_for(blob)
    assert brush.mode == "surface" and brush_for(blob) is brush  # one per viewer
    # Straight down through the spots at y = x = 10 µm (pixel 20): a max projection shows
    # the brightest one, on plane 6.
    stroke(labels, blob, [(-2.0, 10.0, 10.0)], (1.0, 0, 0))
    z, y, x = np.nonzero(labels.data)
    assert labels.data[6, 20, 20] == 5
    # A ball in µm: radius 4.5 px = 2.25 µm, so one 2 µm plane above and below.
    assert set(z) == {5, 6, 7} and y.min() == 16 and y.max() == 24
    assert (labels.data[5] != 0).sum() < (labels.data[6] != 0).sum()
    assert labels.n_edit_dimensions == 2  # the layer's own setting is kept

    # Translucent rendering paints the front of the signal (the spot nearer the camera).
    labels.data[:] = 0
    blob.layers["img"].rendering = "translucent"
    stroke(labels, blob, [(-2.0, 10.0, 10.0)], (1.0, 0, 0))
    assert labels.data[2, 20, 20] == 5 and labels.data[6].max() == 0

    # From below the other spot is in front.
    labels.data[:] = 0
    stroke(labels, blob, [(30.0, 10.0, 10.0)], (-1.0, 0, 0))
    assert labels.data[6, 20, 20] == 5 and labels.data[2].max() == 0

    # Dark background: nothing to paint on.
    before = labels.data.copy()
    stroke(labels, blob, [(-2.0, 2.0, 2.0)], (1.0, 0, 0))
    assert np.array_equal(labels.data, before)


def test_brush_drags_follow_the_surface_and_undo(blob):
    from ibif_napari._draw3d import brush_for

    labels = blob.layers["cells"]
    labels.mode, labels.brush_size, labels.selected_label = "paint", 2, 1
    brush_for(blob)
    # A slanted ray, dragged across the bright spot.
    stroke(labels, blob, [(-2.0, 8.5, 8.5), (-2.0, 8.5, 11.0)], (1.0, 0.0, 0.05))
    painted = np.nonzero(labels.data)
    assert set(painted[0]) <= {5, 6, 7} and labels.data[6].sum() > 3
    labels.undo()  # one stroke, one undo step
    assert labels.data.max() == 0


def test_a_long_drag_stays_on_the_object(qapp):
    # Regression: each ball used to be painted on the ball before it, so a drag climbed
    # towards the camera. A slanted drag over a solid ball stays on its surface.
    from ibif_napari._draw3d import brush_for

    v = ViewerModel()
    zz, yy, xx = np.mgrid[:20, :64, :64]
    r = np.sqrt(((zz - 10) * 2.0) ** 2 + ((yy - 32) * 0.5) ** 2 + ((xx - 32) * 0.5) ** 2)
    v.add_image((r < 10).astype(np.float32), scale=(2, 0.5, 0.5), contrast_limits=(0, 1), name="ball")
    labels = v.add_labels(np.zeros((20, 64, 64), np.uint8), scale=(2, 0.5, 0.5), name="cells")
    v.dims.ndisplay = 3
    labels.mode, labels.brush_size, labels.selected_label = "paint", 6, 1
    brush_for(v)
    direction = np.array([0.8, 0.5, -0.33])
    start = np.array([20.0, 16.0, 16.0]) - 30 * direction
    points = [tuple(start + (0, 0.0, dx)) for dx in np.arange(-6, 6.5, 0.5)]
    stroke(labels, v, points, tuple(direction))
    painted = np.stack(np.nonzero(labels.data), 1) * np.array([2, 0.5, 0.5])
    assert len(painted)
    dist = np.linalg.norm(painted - np.array([20, 16, 16]), axis=1)
    assert dist.max() < 10 + 2 + 1.5  # ball radius + brush radius + a voxel


def test_erasing_takes_the_existing_label_in_front(blob):
    from ibif_napari._draw3d import brush_for

    labels = blob.layers["cells"]
    labels.data[1, 18:23, 18:23] = 7  # a label in front of both spots
    labels.mode, labels.brush_size = "erase", 4
    brush_for(blob)
    stroke(labels, blob, [(-2.0, 10.0, 10.0)], (1.0, 0, 0))
    assert labels.data[1, 20, 20] == 0
    # Painting follows the image, not the labels in front of it ...
    labels.data[1, 18:23, 18:23] = 7
    labels.mode, labels.selected_label = "paint", 3
    stroke(labels, blob, [(-2.0, 10.0, 10.0)], (1.0, 0, 0))
    assert labels.data[6, 20, 20] == 3 and labels.data[1, 20, 20] == 7
    # ... and with no image shown, it paints over another label in the way, but never
    # climbs onto its own label.
    labels.data[:] = 0
    labels.data[1, 18:23, 18:23] = 7
    blob.layers["img"].visible = False
    stroke(labels, blob, [(-2.0, 10.0, 10.0)], (1.0, 0, 0))
    assert labels.data[1, 20, 20] == 3
    labels.data[labels.data == 7] = 0
    before = labels.data.copy()
    stroke(labels, blob, [(-2.0, 10.0, 10.0)], (1.0, 0, 0))
    assert np.array_equal(labels.data, before)


def test_brush_with_time_and_napari_default(qapp):
    from ibif_napari._draw3d import brush_for

    v = ViewerModel()
    img = np.zeros((2, 10, 40, 40), np.float32)
    img[1, 4, 10:20, 10:20] = 1.0  # only at the second time point
    v.add_image(img, scale=(1, 2, 0.5, 0.5), name="img", contrast_limits=(0, 1))
    labels = v.add_labels(np.zeros((2, 10, 40, 40), np.uint8), scale=(1, 2, 0.5, 0.5), name="cells")
    v.dims.ndisplay = 3
    labels.mode, labels.brush_size, labels.selected_label = "paint", 2, 1
    brush = brush_for(v)
    stroke(labels, v, [(0, -2.0, 7.5, 7.5)], (0, 1.0, 0, 0))
    assert labels.data.max() == 0  # the first time point is dark
    v.dims.set_point(0, 1)
    stroke(labels, v, [(1, -2.0, 7.5, 7.5)], (0, 1.0, 0, 0))
    assert labels.data[1, 4, 15, 15] == 1 and labels.data[0].max() == 0

    brush.set_mode("off")  # napari's own brush: nothing on empty voxels
    labels.data[:] = 0
    stroke(labels, v, [(1, -2.0, 7.5, 7.5)], (0, 1.0, 0, 0))
    assert labels.data.max() == 0


def test_cellpose_region_painted_in_3d_keeps_its_shape(blob):
    from napari.layers import Image

    from ibif_napari._draw3d import brush_for
    from ibif_napari._region import ZExtendedRegion, keep_inside, painted_mask, region_crop

    blob.layers.remove("cells")
    region = blob.add_labels(np.zeros((10, 40, 40), np.uint8), scale=(2, 0.5, 0.5), name="region",
                             metadata={"ibif_region": True})
    sync = ZExtendedRegion(region, (0, 9))
    region.mode, region.brush_size, region.selected_label = "paint", 8, 1
    brush_for(blob)
    stroke(region, blob, [(-2.0, 10.0, 10.0)], (1.0, 0, 0))
    assert region.metadata["ibif_region_3d"]
    planes = sorted(set(np.nonzero(region.data)[0]))
    assert planes == [5, 6, 7]  # not copied to every plane
    sync.set_range(2, 8)  # changing the Z planes setting leaves it as painted
    assert sorted(set(np.nonzero(region.data)[0])) == [5, 6, 7]

    image = blob.layers["img"]
    assert isinstance(image, Image)
    crop = region_crop(image, "ZYX", region, (2, 8))
    assert crop["Z"] == (5, 8)
    mask = painted_mask(image, region, crop)
    assert mask.ndim == 3 and mask.shape == tuple(b - a for a, b in (crop["Z"], crop["Y"], crop["X"]))
    assert mask[1].sum() > mask[0].sum()  # a ball: widest in the middle plane
    # Objects mostly outside the painted ball are dropped.
    cells = np.zeros(mask.shape, np.uint16)
    cells[1][mask[1]] = 1
    cells[0, :2, :2] = 2
    assert set(np.unique(keep_inside(cells, mask))) == {0, 1}
