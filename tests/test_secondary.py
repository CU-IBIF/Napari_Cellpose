import numpy as np
import pytest

from ibif_napari._secondary import (
    combine_channels,
    cytoplasm,
    expand,
    foreground,
    ring,
    seeded_cells,
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
