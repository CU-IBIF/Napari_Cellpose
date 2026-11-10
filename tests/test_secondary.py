import numpy as np
import pytest
from scipy import ndimage as ndi

from ibif_napari._secondary import (
    combine_channels,
    count_pieces,
    cytoplasm,
    expand,
    foreground,
    ring,
    seeded_cells,
    structures,
)


def disk(shape, center, radius, value=1):
    yy, xx = np.indices(shape)
    return np.where((yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2, value, 0)


def two_nuclei():
    a = disk((40, 60), (20, 15), 4, 1)
    b = disk((40, 60), (20, 33), 4, 2)
    return (a + b).astype(np.uint16)


def test_expand_keeps_ids_and_does_not_merge():
    nuclei = two_nuclei()
    out = expand(nuclei, 6)
    assert out.dtype == nuclei.dtype and set(np.unique(out)) == {0, 1, 2}
    assert (out[nuclei > 0] == nuclei[nuclei > 0]).all()  # objects keep their own pixels
    # Grown by 6 px: radius 4 -> ~10 on the free side; nucleus 1's left edge at x=11 -> ~5.
    assert out[20, 5] == 1 and out[20, 3] == 0
    # Between the nuclei (x=15 and x=33) the border is half way, at x=24.
    assert out[20, 23] == 1 and out[20, 25] == 2


def test_expand_uses_physical_spacing_in_3d():
    labels = np.zeros((11, 21, 21), np.uint8)
    labels[5, 10, 10] = 1
    out = expand(labels, 2.0, spacing=(2.0, 0.5, 0.5))  # 2 µm: 1 plane up/down, 4 px sideways
    assert out[4, 10, 10] == 1 and out[3, 10, 10] == 0
    assert out[5, 10, 14] == 1 and out[5, 10, 15] == 0


def test_ring_outside_and_inside():
    nuclei = two_nuclei()
    r = ring(nuclei, width=3, gap=1)
    assert set(np.unique(r)) == {0, 1, 2}
    assert not (r[nuclei > 0]).any()  # never on the object itself
    assert r[20, 10] == 0  # 1 px from nucleus 1 (edge at x=11): inside the gap
    assert r[20, 8] == 1  # 3 px away: in the band (gap 1, width 3)
    assert r[20, 6] == 0  # 5 px away: beyond the band

    rim = ring(nuclei, width=2, inside=True)
    assert set(np.unique(rim)) == {0, 1, 2}
    assert rim[20, 11] == 1 and rim[20, 12] == 1  # edge pixels of nucleus 1
    assert rim[20, 15] == 0  # its centre is not in a 2 px rim
    assert not rim[nuclei == 0].any()


def test_combine_and_threshold():
    a = np.zeros((10, 10)); a[:5] = 1000
    b = np.zeros((10, 10)); b[:, :5] = 1
    combined = combine_channels([a, b])
    assert combined.max() == pytest.approx(1.0) and combined[0, 0] == pytest.approx(1.0)
    assert combined[0, 9] == pytest.approx(0.5) and combined[9, 9] == 0
    mask, t = foreground(combined, "otsu")
    assert mask[0, 0] and not mask[9, 9] and 0 < t < 1
    mask, t = foreground(combined, "manual", 0.75)
    assert mask.sum() == 25 and t == 0.75
    assert foreground(combined, "none")[0].all()
    with pytest.raises(ValueError):
        combine_channels([])


def cells_image():
    """Two touching square cells with bright membranes, each with a nucleus."""
    img = np.zeros((40, 60), np.float32)
    img[10:31, 5:26] = 0.3  # cell 1 interior (cytoplasm signal)
    img[10:31, 24:46] = 0.3  # cell 2 interior
    img[10, 5:46] = img[30, 5:46] = 1.0  # membranes
    img[10:31, 5] = img[10:31, 45] = img[10:31, 24] = 1.0
    return img


def test_seeded_cells_membrane_stain():
    nuclei = two_nuclei()
    cells, t = seeded_cells(nuclei, cells_image(), kind="membrane", threshold="otsu", factor=0.5, smooth=0)
    assert set(np.unique(cells)) == {0, 1, 2}
    assert cells[20, 15] == 1 and cells[20, 33] == 2  # the nuclei
    assert cells[12, 8] == 1 and cells[28, 42] == 2  # far corners of each cell
    assert cells[20, 22] == 1 and cells[20, 27] == 2  # the shared membrane splits them
    assert cells[2, 2] == 0 and cells[38, 58] == 0  # background stays empty
    assert (cells[nuclei > 0] == nuclei[nuclei > 0]).all()


def dark_interior_cells():
    """Two touching cells whose membranes are the only signal (dark inside, as many membrane
    stains are), plus an enclosed space with no nucleus (a lumen) on the right."""
    img = np.zeros((40, 80), np.float32)
    img[10, 5:66] = img[30, 5:66] = 1.0  # top and bottom membranes
    for x in (5, 24, 45, 65):
        img[10:31, x] = 1.0  # side membranes: cell 1 | cell 2 | lumen
    return img


def test_seeded_cells_membrane_stain_with_a_dark_interior():
    nuclei = np.zeros((40, 80), np.uint16)
    yy, xx = np.indices(nuclei.shape)
    nuclei[((yy - 20) ** 2 + (xx - 14) ** 2) <= 9] = 1
    nuclei[((yy - 20) ** 2 + (xx - 34) ** 2) <= 9] = 2
    # The default threshold keeps only the membranes; the cells must still fill them.
    cells, _ = seeded_cells(nuclei, dark_interior_cells(), kind="membrane", threshold="otsu", smooth=0)
    assert cells[12, 7] == 1 and cells[28, 22] == 1  # far corners of cell 1
    assert cells[12, 26] == 2 and cells[28, 43] == 2  # far corners of cell 2
    assert (cells[10:31, 6:24] == 1).all() and (cells[11:30, 25:45] == 2).all()
    assert (cells[11:30, 46:65] == 0).all()  # the lumen has no nucleus: not a cell
    assert cells[2, 2] == 0 and cells[38, 78] == 0  # background stays empty
    # In 3D, per Z plane.
    cells3d, _ = seeded_cells(np.stack([nuclei] * 3), np.stack([dark_interior_cells()] * 3),
                              spacing=(2.0, 1.0, 1.0), kind="membrane", smooth=0)
    for plane in cells3d:
        assert (plane[11:30, 6:24] == 1).all() and (plane[11:30, 25:45] == 2).all()
        assert (plane[11:30, 46:65] == 0).all() and plane[2, 2] == 0


def test_seeded_cells_cytoplasm_distance_limit_and_cytoplasm():
    nuclei = two_nuclei()
    img = np.zeros((40, 60), np.float32)
    img[10:31, 5:46] = 1.0  # one bright block of cytoplasm around both nuclei
    cells, _ = seeded_cells(nuclei, img, kind="cytoplasm", threshold="otsu", smooth=0)
    assert set(np.unique(cells)) == {0, 1, 2} and cells[20, 6] == 1 and cells[20, 44] == 2
    limited, _ = seeded_cells(nuclei, img, kind="distance", threshold="otsu", max_distance=3, smooth=0)
    assert limited[20, 6] == 0 and limited[20, 8] == 1  # 3 px from nucleus 1's edge at x=11
    cyto = cytoplasm(cells, nuclei)
    assert not cyto[nuclei > 0].any() and cyto[20, 6] == 1


def test_seeded_cells_in_3d_and_input_checks():
    nuclei = np.zeros((6, 20, 20), np.uint16)
    nuclei[2:4, 8:12, 8:12] = 5
    img = np.zeros((6, 20, 20), np.float32)
    img[1:5, 4:16, 4:16] = 1.0
    cells, _ = seeded_cells(nuclei, img, spacing=(2.0, 0.5, 0.5), kind="cytoplasm", smooth=0.5)
    assert set(np.unique(cells)) == {0, 5} and cells[1, 5, 5] == 5 and cells[0, 10, 10] == 0
    with pytest.raises(ValueError, match="differ in size"):
        seeded_cells(nuclei, img[:3])
    with pytest.raises(ValueError, match="spacing"):
        expand(nuclei, 1, spacing=(1, 1))
    assert not seeded_cells(np.zeros_like(nuclei), img)[0].any()


def membrane_only_image():
    """Two touching square cells as seen in a membrane stain: each cell has its own bright
    one-pixel membrane, the two membranes touch, and the cell insides are as dark as the
    background. Returns (image, true cells)."""
    truth = np.zeros((40, 60), np.uint16)
    truth[10:31, 5:25] = 1
    truth[10:31, 25:46] = 2
    img = np.zeros((40, 60), np.float32)
    for label in (1, 2):
        cell = truth == label
        img[cell & ~ndi.binary_erosion(cell)] = 1.0
    return img, truth


def test_seeded_cells_membrane_with_dark_inside_keeps_each_ring():
    # With the default Otsu foreground only the membranes are kept. A cell's own ring is cut
    # off from its nucleus by the dark inside, but touches the neighbour's ring, and the
    # neighbour's nucleus touches that ring; the ring must still get the cell's own number.
    nuclei = two_nuclei()
    nuclei[nuclei == 2] = 0
    nuclei[disk(nuclei.shape, (20, 40), 4) > 0] = 2  # touches its own right membrane (x=45)
    img, truth = membrane_only_image()
    for smooth in (0, 1.0):
        cells, _ = seeded_cells(nuclei, img, kind="membrane", threshold="otsu", smooth=smooth)
        assert set(np.unique(cells)) == {0, 1, 2}
        assert (cells[nuclei > 0] == nuclei[nuclei > 0]).all()
        for label in (1, 2):
            ring = (truth == label) & ~ndi.binary_erosion(truth == label)
            assert not np.isin(cells[ring], [0, label], invert=True).any()  # no other cell's number
            inner = ndi.binary_erosion(truth == label)
            assert (cells[inner] == label).all()  # the dark inside belongs to the cell
        assert cells[20, 24] == 1 and cells[20, 25] == 2  # the two touching membranes
        # The dark outside stays background (after blurring, cells may take 1 px of it).
        outside = (truth == 0) if smooth == 0 else ~ndi.binary_dilation(truth > 0)
        assert not cells[outside].any()
    # A maximum distance still limits the cells.
    limited, _ = seeded_cells(nuclei, img, kind="membrane", threshold="otsu", smooth=0, max_distance=3)
    assert limited[20, 8] == 1 and limited[20, 6] == 0


# ---- structures inside objects ---------------------------------------------------------------

def two_cells():
    cells = np.zeros((40, 70), np.uint16)
    cells[5:35, 5:33] = 1
    cells[5:35, 37:65] = 2
    return cells


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_structures_spots_take_their_objects_number():
    cells = two_cells()
    truth = np.zeros(cells.shape, bool)
    for y, x in ((12, 12), (25, 22), (18, 28), (20, 50)):  # 3 in cell 1, 1 in cell 2
        truth |= disk(cells.shape, (y, x), 2) > 0
    outside = disk(cells.shape, (20, 68), 2) > 0  # a spot outside every cell
    rng = np.random.default_rng(1)
    img = (0.2 + 0.02 * rng.standard_normal(cells.shape)).astype(np.float32)
    img[cells > 0] += 0.1  # the cells are brighter than the background, spots more so
    img[truth | outside] = 1.0
    found, used = structures(cells, img, kind="spots", size=3, smooth=0)
    assert len(used) == 1
    assert set(np.unique(found)) == {0, 1, 2}
    assert (found[found > 0] == cells[found > 0]).all()  # each piece has its cell's number
    assert not found[outside].any() and not found[cells == 0].any()
    assert iou(found > 0, truth) > 0.8
    assert count_pieces(found) == {1: 3, 2: 1}


def test_structures_dark_nucleoli_in_a_dna_stain():
    nuclei = np.zeros((40, 70), np.uint16)
    nuclei[disk(nuclei.shape, (20, 18), 12) > 0] = 1
    nuclei[disk(nuclei.shape, (20, 52), 12) > 0] = 2
    nucleoli = (disk(nuclei.shape, (16, 15), 3) | disk(nuclei.shape, (24, 22), 2)
                | disk(nuclei.shape, (20, 52), 4)) > 0
    img = np.where(nuclei > 0, 1.0, 0.05).astype(np.float32)
    img[nucleoli] = 0.3  # nucleoli: holes in the DNA stain
    for kind in ("regions", "spots"):
        found, _ = structures(nuclei, img, kind=kind, size=4, dark=True, smooth=0)
        assert (found[found > 0] == nuclei[found > 0]).all()
        assert iou(found > 0, nucleoli) > 0.7, kind
        assert count_pieces(found) == {1: 2, 2: 1}, kind


def test_structures_filaments():
    cells = two_cells()
    line = np.zeros(cells.shape, bool)
    line[20, 8:30] = True  # a 1 px filament in cell 1
    line[10:30, 50] = True  # and one in cell 2
    rng = np.random.default_rng(2)
    img = (0.1 + 0.02 * rng.standard_normal(cells.shape)).astype(np.float32)
    img[line] = 1.0
    found, _ = structures(cells, img, kind="filaments", size=1.0, smooth=0)
    assert (found[found > 0] == cells[found > 0]).all()
    assert (found[line] > 0).mean() > 0.9
    assert (found > 0).sum() < 3 * line.sum()  # thin, not a blob
    assert count_pieces(found) == {1: 1, 2: 1}
    # 3D with long Z voxels: the ridge filter runs per plane.
    cells3 = np.stack([cells] * 3)
    found3, _ = structures(cells3, np.stack([img] * 3), spacing=(2.0, 0.5, 0.5), kind="filaments",
                           size=0.5, smooth=0)
    assert all((plane[line] > 0).mean() > 0.9 for plane in found3)


def test_structures_threshold_per_object_and_min_size():
    cells = two_cells()
    dim = disk(cells.shape, (15, 15), 2) > 0
    bright = disk(cells.shape, (15, 50), 2) > 0
    rng = np.random.default_rng(3)
    img = (0.005 * rng.standard_normal(cells.shape)).astype(np.float32)
    img[cells == 1] += 0.1  # cell 1 is dim, with a dim spot
    img[dim] = 0.2
    img[cells == 2] += 0.5  # cell 2 is bright all over, with a bright spot
    img[bright] = 1.0
    together, _ = structures(cells, img, kind="regions", threshold="otsu", smooth=0)
    assert (together[cells == 2] == 2).all()  # one threshold for all: cell 2 is all "structure"
    each, used = structures(cells, img, kind="regions", threshold="otsu", per_object=True, smooth=0,
                            min_size=2)  # (no single noise pixels)
    assert len(used) == 2
    assert iou(each > 0, dim | bright) == 1.0  # a threshold per cell finds just the spots
    # A tiny piece (1 px = 0.25 µm² at 0.5 µm pixels) is dropped by a minimum size.
    img[30, 10] = 0.2
    kept, _ = structures(cells, img, spacing=(0.5, 0.5), kind="regions", per_object=True, smooth=0)
    assert kept[30, 10] == 1
    dropped, _ = structures(cells, img, spacing=(0.5, 0.5), kind="regions", per_object=True, smooth=0,
                            min_size=1.0)
    assert dropped[30, 10] == 0 and dropped[dim].all()  # the 13 px spot (3.25 µm²) stays
    manual, used = structures(cells, img, kind="regions", threshold="manual", value=0.8, smooth=0)
    assert used == [0.8] and iou(manual > 0, bright) == 1.0


def test_structures_input_checks():
    cells = two_cells()
    img = np.zeros(cells.shape, np.float32)
    with pytest.raises(ValueError, match="differ in size"):
        structures(cells, img[:5])
    with pytest.raises(ValueError, match="unknown structure kind"):
        structures(cells, img, kind="blobs")
    with pytest.raises(ValueError, match="choose a threshold"):
        structures(cells, img, threshold="none")
    with pytest.raises(ValueError, match="threshold value"):
        structures(cells, img, threshold="manual")
    assert not structures(np.zeros_like(cells), img)[0].any()
    assert not structures(cells, img)[0].any()  # a flat channel has no structures
