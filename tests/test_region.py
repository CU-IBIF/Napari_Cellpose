import numpy as np
import pytest

pytest.importorskip("napari")

from napari.layers import Image, Labels, Shapes  # noqa: E402

from ibif_napari._region import keep_inside, painted_mask_yx, region_crop  # noqa: E402


@pytest.fixture
def image():
    # T Z Y X, 0.5 um pixels, 2 um planes, shifted by 10 um in X
    return Image(np.zeros((2, 6, 40, 50)), scale=(1, 2, 0.5, 0.5), translate=(0, 0, 0, 10))


def test_whole_image_with_z_range(image):
    assert region_crop(image, "TZYX", None, None) == {}
    assert region_crop(image, "TZYX", None, (1, 3)) == {"Z": (1, 4)}


def test_painted_2d_labels_on_same_grid(image):
    paint = np.zeros((40, 50), np.uint8)
    paint[5:15, 20:31] = 1
    region = Labels(paint, scale=(0.5, 0.5), translate=(0, 10))
    assert region_crop(image, "TZYX", region, (0, 5)) == {"Y": (5, 15), "X": (20, 31), "Z": (0, 6)}


def test_labels_on_coarser_grid_maps_through_world_coords(image):
    paint = np.zeros((20, 25), np.uint8)  # 1 um pixels, no offset
    paint[2:4, 10:13] = 1  # world y 2..3, x 10..12
    region = Labels(paint, scale=(1, 1))
    crop = region_crop(image, "TZYX", region)
    # image pixel centres between world y 1.5..3.5 -> rows 3..7; x 9.5..12.5 -> cols 0..5
    assert crop == {"Y": (3, 8), "X": (0, 6)}


def test_3d_labels_painted_on_several_planes_limit_z(image):
    paint = np.zeros((6, 40, 50), np.uint8)
    paint[2:4, 10:12, 10:12] = 1
    region = Labels(paint, scale=(2, 0.5, 0.5), translate=(0, 0, 10))
    crop = region_crop(image, "TZYX", region, (0, 5))
    assert crop == {"Z": (2, 4), "Y": (10, 12), "X": (10, 12)}
    # A narrower Z range still wins
    assert region_crop(image, "TZYX", region, (3, 5))["Z"] == (3, 4)


def test_3d_labels_painted_on_one_plane_apply_to_all_planes(image):
    paint = np.zeros((6, 40, 50), np.uint8)
    paint[4, 10:12, 10:12] = 1
    region = Labels(paint, scale=(2, 0.5, 0.5), translate=(0, 0, 10))
    assert region_crop(image, "TZYX", region)["Y"] == (10, 12)
    assert "Z" not in region_crop(image, "TZYX", region)


def test_shapes_rectangle(image):
    # rectangle in world coordinates (y 2..5 um, x 15..20 um)
    region = Shapes([np.array([[2, 15], [2, 20], [5, 20], [5, 15]])], shape_type="rectangle")
    assert region_crop(image, "TZYX", region) == {"Y": (4, 11), "X": (10, 21)}


def test_empty_or_outside_region(image):
    assert region_crop(image, "TZYX", Labels(np.zeros((40, 50), np.uint8))) is None
    far = np.zeros((10, 10), np.uint8)
    far[:] = 1
    assert region_crop(image, "TZYX", Labels(far, translate=(500, 500))) is None
    assert region_crop(image, "TZYX", Shapes([])) is None


def test_keep_inside_drops_cells_mostly_outside(image):
    paint = np.zeros((40, 50), np.uint8)
    paint[0:10, 0:10] = 1
    region = Labels(paint, scale=(0.5, 0.5), translate=(0, 10))
    crop = {"Y": (0, 20), "X": (0, 20)}
    mask = painted_mask_yx(image, region, crop)
    assert mask.shape == (20, 20) and mask[:10, :10].all() and not mask[10:, :].any()

    labels = np.zeros((3, 20, 20), np.uint16)
    labels[:, 2:6, 2:6] = 1  # inside
    labels[:, 8:12, 2:6] = 2  # half inside -> kept
    labels[:, 14:18, 14:18] = 3  # outside
    out = keep_inside(labels, mask)
    assert set(np.unique(out)) == {0, 1, 2}
    assert keep_inside(labels, None) is labels
