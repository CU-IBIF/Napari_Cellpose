import numpy as np
import pytest

pytest.importorskip("skan")

from _synthetic import SPACING, draw  # noqa: E402

from ibif_napari import _filaments as F  # noqa: E402

SP = np.asarray(SPACING)
# A neuron-like Y: a 12 um trunk with a side branch from its middle.
Y_SEGMENTS = [((2, 3, 2), (2, 3, 14)), ((2, 3, 8), (2.4, 7, 12))]
Y_LENGTH = 12 + np.linalg.norm([0.4, 4, 4])


def trace(img, spacing=SPACING, **kw):
    enh = F.enhance(img, spacing, 0.2, 0.8)
    mask, _ = F.segment(enh, None, spacing, 0.8)
    skel = F.skeleton_from_mask(mask, spacing, prune=kw.get("prune", 0.5), min_length=1.0)
    return F.measure(skel, spacing, img, mask), mask


def test_branched_filament_statistics():
    img = 50 * draw((12, 100, 160), Y_SEGMENTS) + 100
    r, _ = trace(img)
    (f,) = r.filaments.to_dict("records")
    assert f["length_um"] == pytest.approx(Y_LENGTH, rel=0.05)
    assert (f["segments"], f["branch_points"], f["terminal_points"]) == (3, 1, 3)
    assert 0.2 < f["mean_diameter_um"] < 0.8
    assert f["mean_intensity"] > 120  # the image, not the skeleton
    # Each segment: the two trunk halves (~6 um) and the branch (~5.7 um)
    lengths = sorted(r.segments["length_um"])
    assert lengths == pytest.approx([5.66, 6, 6], abs=0.6)
    assert set(r.segments["type"]) == {"terminal"}
    assert set(r.points["type"]) == {"root", "branch", "terminal"}
    # Positions are in um in the image's coordinates
    branch = r.points[r.points["type"] == "branch"].iloc[0]
    assert (branch["y_um"], branch["x_um"]) == pytest.approx((3, 8), abs=0.6)  # blurred
    # Sholl from the root (at the branch point, the thickest place): three arms
    assert r.sholl["intersections"].iloc[:4].tolist() == [3, 3, 3, 3]


def test_origin_shifts_positions():
    img = draw((12, 100, 160), Y_SEGMENTS)
    enh = F.enhance(img, SPACING, 0.2, 0.8)
    mask, _ = F.segment(enh)
    skel = F.skeleton_from_mask(mask, SPACING)
    r0 = F.measure(skel, SPACING, img, mask)
    r1 = F.measure(skel, SPACING, img, mask, origin=(10, 20, 30))
    d = r1.points[["z_um", "y_um", "x_um"]].to_numpy() - r0.points[["z_um", "y_um", "x_um"]].to_numpy()
    assert np.allclose(d, (10, 20, 30))


def test_short_spurs_are_pruned_and_short_pieces_dropped():
    # trunk with a 0.6 um spur, plus a separate 0.8 um speck
    img = draw((10, 80, 160), [((2, 3, 2), (2, 3, 14)), ((2, 3, 8), (2, 3.6, 8)), ((2, 6, 4), (2, 6, 4.8))])
    r, _ = trace(img, prune=1.0)
    assert len(r.filaments) == 1
    assert r.filaments["branch_points"].iloc[0] == 0
    assert r.filaments["length_um"].iloc[0] == pytest.approx(12, rel=0.05)


def test_2d_image():
    sp = (0.1, 0.1)
    img = draw((1, 80, 80), [((0, 1, 1), (0, 7, 7))], spacing=(1, *sp))[0]
    r, _ = trace(img, spacing=sp)
    assert r.filaments["length_um"].iloc[0] == pytest.approx(np.hypot(6, 6), rel=0.05)
    assert "elevation_deg" not in r.segments


def test_no_filaments():
    r = F.measure(np.zeros((4, 10, 10), bool), SPACING)
    assert r.filaments.empty and r.segments.empty and r.points.empty


def test_trace_path_follows_a_bent_filament():
    img = 50 * draw((10, 120, 120), [((2, 2, 2), (2, 2, 10)), ((2, 2, 10), (3.2, 10, 10))]) + 100
    start = np.round(np.array([2, 2, 2.3]) / SP)
    end = F.snap(np.round(np.array([2.8, 9.7, 10]) / SP), img, None, SPACING)  # clicked off-plane
    assert end[0] == 8  # moved to the plane where the filament is
    path, lo, mask = F.trace_path(img, SPACING, start, end, 0.2, 0.8)
    assert tuple(path[0]) == tuple(start) and tuple(path[-1]) == tuple(end)
    corner = np.round(np.array([2, 2, 10]) / SP)
    assert np.linalg.norm((path - corner) * SP, axis=1).min() < 0.5  # went round the bend
    length = np.linalg.norm(np.diff(path * SP, axis=0), axis=1).sum()
    assert length == pytest.approx(8 + np.hypot(1.2, 8), rel=0.08)
    assert mask.any() and mask.ndim == 3


def test_snap_to_existing_skeleton():
    img = np.zeros((5, 40, 40), np.float32)
    skel = np.zeros(img.shape, bool)
    skel[2, 20, 5:35] = True
    assert tuple(F.snap((2, 23, 10), img, skel, SPACING, radius_um=0.5)) == (2, 20, 10)
    assert tuple(F.snap((2, 30, 10), img, skel, SPACING, radius_um=0.5))[1:] == (30, 10)


def test_export_csv_and_swc(tmp_path):
    img = draw((12, 100, 160), Y_SEGMENTS)
    r, _ = trace(img)
    written = F.export(r, str(tmp_path / "cell"))
    assert [p.rsplit("/", 1)[-1] for p in written] == [
        "cell_filaments.csv", "cell_segments.csv", "cell_points.csv", "cell_sholl.csv", "cell.swc"]
    import pandas as pd

    assert not any(c.startswith("_") for c in pd.read_csv(tmp_path / "cell_points.csv").columns)
    rows = [l.split() for l in (tmp_path / "cell.swc").read_text().splitlines() if not l.startswith("#")]
    ids = {int(r[0]) for r in rows}
    parents = [int(r[6]) for r in rows]
    assert parents.count(-1) == 1  # one tree, rooted at the soma-like thickest point
    assert all(p in ids for p in parents if p != -1)
    assert sum(1 for r in rows if r[1] == "1") == 1
