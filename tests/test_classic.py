"""Threshold + watershed segmentation (the Segmentation section's second tab)."""
import numpy as np
import pytest

from ibif_napari import _classic


def discs(shape=(120, 160), centres=((40, 40), (40, 60), (90, 120)), r=11, background=100, peak=1000,
          noise=20, seed=0):
    """Bright discs (two touching) on a noisy background, and their truth labels."""
    yy, xx = np.indices(shape)
    truth = np.zeros(shape, np.int32)
    best = np.full(shape, np.inf)
    for i, (y, x) in enumerate(centres, start=1):
        d = np.hypot(yy - y, xx - x)
        inside = (d <= r) & (d < best)
        truth[inside], best[inside] = i, d[inside]
    rng = np.random.default_rng(seed)
    image = background + peak * (truth > 0) + rng.normal(0, noise, shape)
    return image.astype(np.float32), truth


def test_touching_objects_are_split_by_the_watershed():
    image, truth = discs()
    labels, info = _classic.segment(image)
    assert labels.dtype == np.uint32 and info["objects"] == 3 == labels.max()
    assert 300 < info["threshold"] < 900  # between background and objects, in the image's units
    # Connected components alone merge the touching pair.
    cc, _ = _classic.segment(image, split="none")
    assert cc.max() == 2
    # Each true disc is (almost all) one object.
    for i in (1, 2, 3):
        ids, counts = np.unique(labels[truth == i], return_counts=True)
        assert counts.max() / counts.sum() > 0.9 and ids[np.argmax(counts)] > 0


def test_thresholds_size_and_border():
    image, truth = discs(centres=((40, 40), (90, 120), (5, 150)))  # the last one is cut by the border
    for method in ("otsu", "li", "triangle", "yen", "mean"):
        assert _classic.segment(image, threshold=method)[1]["objects"] == 3
    assert _classic.segment(image, clear_border=True)[1]["objects"] == 2
    assert _classic.segment(image, min_size=330)[1]["objects"] == 2  # the cut disc is smaller
    labels, info = _classic.segment(image, threshold="manual", value=600, smooth=0, split="none")
    assert info["threshold"] == 600 and labels.max() == 3
    with pytest.raises(ValueError):
        _classic.segment(image, threshold="manual", value=None)
    # A local threshold copes with a strong illumination gradient, one global one does not.
    ramp = image + np.linspace(0, 1500, image.shape[1])[None, :]
    assert _classic.segment(ramp)[1]["objects"] != 3
    assert _classic.segment(ramp, threshold="local", block=61)[1]["objects"] == 3
    # Local contrast (CLAHE) brings out a dim object next to bright ones.
    image, truth = discs(centres=((40, 40), (90, 120), (40, 120)), noise=10)
    image = np.where(truth == 3, image - 850, image)  # 150 above background, the others 1000
    assert _classic.segment(image)[1]["objects"] == 2
    assert _classic.segment(image, clahe=True)[1]["objects"] == 3


def test_3d_and_planes_with_anisotropic_voxels():
    zz, yy, xx = np.indices((16, 60, 80))
    truth = np.zeros(zz.shape, np.int32)
    for i, (z, y, x) in enumerate(((8, 30, 25), (8, 30, 42), (8, 30, 65)), start=1):
        # Spheres of radius 8 µm with a 2 µm Z step and 1 µm pixels.
        truth[((zz - z) * 2) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= 64] = i
    rng = np.random.default_rng(1)
    image = 100 + 900 * (truth > 0) + rng.normal(0, 30, truth.shape)
    labels, info = _classic.segment(image, spacing=(2.0, 1.0, 1.0))
    assert info["objects"] == 3
    planes, info = _classic.segment_planes(image, spacing=(2.0, 1.0, 1.0))
    assert info["objects"] == 3  # each object linked across its planes
    for i in (1, 2, 3):
        assert len(np.unique(planes[truth == i])) <= 2  # its own number (and maybe a few 0s)
    with pytest.raises(ValueError):
        _classic.segment(np.zeros((2, 2, 2, 2)))


def test_empty_and_flat_images():
    assert _classic.segment(np.zeros((30, 30)))[1]["objects"] == 0
    assert _classic.segment(np.full((30, 30), 5.0))[0].max() == 0


# ---- the tab -----------------------------------------------------------------------------------


@pytest.fixture
def viewer(qapp):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    v = ViewerModel()
    image, _ = discs()
    stack = np.stack([image, np.roll(image, 7, axis=1)])  # 2 time points
    v.add_image(stack, name="nuclei", scale=(1, 0.5, 0.5), metadata={"ibif_axes": "TYX"})
    return v


def test_tab_preview_and_run(viewer, qtbot):
    from ibif_napari._classic_widget import SCOPE_ALL, ClassicWidget

    w = ClassicWidget(viewer)
    qtbot.addWidget(w)
    assert w.image.currentText() == "nuclei" and not w.mode.isEnabled()  # no Z
    assert not w.value.isEnabled() and not w.block.isEnabled()
    w.set_threshold("manual")
    assert w.value.isEnabled() and not w.factor.isEnabled()
    w.set_threshold("otsu")
    preview = w.preview()
    assert preview.name == "nuclei foreground" and preview.data.shape == (120, 160)
    assert 300 < w.value.value() < 900  # Manual would start from the automatic threshold
    worker = w.run()
    with qtbot.waitSignal(worker.finished, timeout=30000):
        pass
    out = viewer.layers["nuclei threshold T0"]
    assert out.data.max() == 3 and list(out.scale) == [0.5, 0.5]
    assert out.metadata["ibif_segmentation"] == "threshold + watershed"
    assert out.metadata["ibif_classic"]["threshold"] == "otsu"
    w.scope.setCurrentText(SCOPE_ALL)
    worker = w.run()
    with qtbot.waitSignal(worker.finished, timeout=30000):
        pass
    every = viewer.layers["nuclei threshold all"]
    assert every.data.shape == (2, 120, 160) and every.data[0].max() == every.data[1].max() == 3
    # The region / preview layers are not offered as regions to segment.
    assert "nuclei foreground" not in [w.region.itemText(i) for i in range(w.region.count())]


def test_tab_region(viewer, qtbot):
    from ibif_napari._classic_widget import ClassicWidget

    w = ClassicWidget(viewer)
    qtbot.addWidget(w)
    w.add_region_layer()
    region = viewer.layers[w.region.currentText()]
    region.data[70:110, 100:140] = 1  # around the lone disc only
    region.refresh()
    worker = w.run()
    with qtbot.waitSignal(worker.finished, timeout=30000):
        pass
    out = next(l for l in viewer.layers if l.name.startswith("nuclei threshold") and "region" in l.name)
    assert out.data.max() == 1 and out.data.shape == (40, 40)
    assert list(out.translate) == [70 * 0.5, 100 * 0.5]
