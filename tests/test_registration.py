"""Stack registration (Processing): transforms, estimation on synthetic movement with a
known truth, the step on stacks of every layout, records, and re-applying them."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from ibif_napari import _processing
from ibif_napari._processing import Pipeline, Stack
from ibif_napari._processing import registration as R
from ibif_napari.vv import synth


def _reads(scene, t, border):
    """Pixel coordinates (inside a border) and where the registered frame reads them."""
    shape = scene.frames.shape[1:]
    n = len(shape)
    coords = np.indices(shape).astype(float)
    sl = (slice(None),) + tuple(slice(border, -border) if i >= n - 2 else slice(2, -2) for i in range(n))
    coords = coords[sl]
    if t is None:
        return coords, coords
    read = coords + (t.flow[sl] if t.flow is not None else 0)
    return coords, np.tensordot(t.matrix[:n, :n], read, axes=1) + t.matrix[:n, n].reshape((n,) + (1,) * n)


def residual(scene, k, t, ref=0, border=20, t_ref=None):
    """RMS distance (pixels of the sample) between what registered frame k and the
    reference frame (registered by `t_ref`, if given) read at each pixel: 0 for a
    perfect registration."""
    _, read = _reads(scene, t, border)
    _, read_ref = _reads(scene, t_ref, border)
    d = scene.source(k, read) - scene.source(ref, read_ref)
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=0))))


def test_rigid_matrix_and_warp_move_an_image_as_described():
    img = np.zeros((41, 41), np.float32)
    img[20, 30] = 1.0  # 10 px right of the centre
    shifted = R.warp(img, R.Transform(R.rigid_matrix(img.shape, 0, (3, -2))), order=1)
    assert np.unravel_index(shifted.argmax(), img.shape) == (23, 28)
    turned = R.warp(img, R.Transform(R.rigid_matrix(img.shape, 90, (0, 0))), order=1)
    assert np.unravel_index(turned.argmax(), img.shape) == (30, 20)  # right -> down: clockwise as shown
    # Non-square pixels (X half of Y): turned by 90°, 10 µm to the right (20 px) ends 10 µm
    # down (10 px).
    wide = np.zeros((41, 61), np.float32)
    wide[20, 50] = 1.0
    m = R.rigid_matrix(wide.shape, 90, (0, 0), spacing=(1.0, 0.5))
    assert np.unravel_index(R.warp(wide, R.Transform(m)).argmax(), wide.shape) == (30, 30)
    labels = np.zeros((10, 10), np.uint32)
    labels[2:4, 2:4] = 70000
    moved = R.warp(labels, R.Transform(R.rigid_matrix(labels.shape, 0, (1.5, 0))), order=0)
    assert moved.dtype == np.uint32 and set(np.unique(moved)) == {0, 70000}
    assert np.isnan(R.warp(np.ones((5, 5)), R.Transform(R.rigid_matrix((5, 5), 0, (2, 0))), fill="nan")[0, 0])
    assert R.warp(np.ones((5, 5)), R.Transform(R.rigid_matrix((5, 5), 0, (2, 0))), fill="nearest")[0, 0] == 1


@pytest.mark.parametrize("seed", [0, 1])
def test_translation_is_recovered(seed):
    sc = synth.motion_scene(seed=seed)
    transforms = R.register_frames(sc.frames, "translation", "first", spacing=sc.spacing)
    errors = [residual(sc, k, t) for k, t in enumerate(transforms)]
    assert errors[0] == 0 and max(errors) < 0.3 and np.mean(errors) < 0.15


def test_rotation_is_recovered():
    sc = synth.motion_scene(seed=2, max_angle=10, n=5)
    transforms = R.register_frames(sc.frames, "rigid", "first", spacing=sc.spacing)
    for k, t in enumerate(transforms[1:], start=1):
        assert abs(t.angle - sc.truth[k]["angle"]) < 0.2  # (the scene measures angles the other way)
        assert residual(sc, k, t) < 0.3


def test_3d_time_points_and_z_planes():
    sc = synth.motion_scene(ndim=3, n=4, seed=3)
    transforms = R.register_frames(sc.frames, "translation", "first", spacing=sc.spacing)
    assert max(residual(sc, k, t) for k, t in enumerate(transforms)) < 0.3
    z = synth.motion_scene(along="Z", seed=4)
    planes = R.register_frames(z.frames, "translation", "previous", spacing=z.spacing[1:])
    # Each registered plane lines up with the registered plane before it.
    steps = [residual(z, k, planes[k], ref=k - 1, t_ref=planes[k - 1]) for k in range(1, len(planes))]
    assert np.mean(steps) < 0.25 and max(steps) < 0.5


def test_references_and_a_failed_frame():
    sc = synth.motion_scene(seed=5, n=6, max_shift=6)
    for ref, kw in (("mean", {"n_mean": 1}), ("chosen", {"frame": 0}), ("previous", {})):
        transforms = R.register_frames(sc.frames, "translation", ref, spacing=sc.spacing, **kw)
        assert max(residual(sc, k, t) for k, t in enumerate(transforms)) < 0.6, ref
    chosen = R.register_frames(sc.frames, "translation", "chosen", frame=3, spacing=sc.spacing)
    assert np.allclose(chosen[3].matrix, np.eye(3)) and residual(sc, 1, chosen[1], ref=3) < 0.3
    with pytest.raises(ValueError, match="outside"):
        R.register_frames(sc.frames, "translation", "chosen", frame=99)
    # A limit below the movement leaves frames as they are, marked not registered.
    capped = R.register_frames(sc.frames, "translation", "first", spacing=sc.spacing, max_shift=0.1)
    assert not all(t.ok for t in capped[1:]) and all(np.allclose(t.matrix, np.eye(3)) for t in capped if not t.ok)


def test_no_movement_is_left_alone():
    sc = synth.motion_scene(seed=6, n=3, max_shift=0.0)
    for t in R.register_frames(sc.frames, "translation", "first", spacing=sc.spacing):
        assert np.hypot(*t.shift) < 0.1  # within the noise (steps of 1/20 px)


def _stack_from(frames, axes, spacing, extra=0.0):
    data = {"A": frames.astype(np.float32), "B": (frames * 2 + extra).astype(np.float32)}
    return Stack(data, axes, spacing, "A")


def test_step_registers_every_channel_alike():
    sc = synth.motion_scene(seed=7, n=4)
    stack = _stack_from(sc.frames, "TYX", {"Y": 0.5, "X": 0.5}, extra=5)
    pipe = Pipeline()
    pipe.add("registration")
    result = pipe.run(stack)
    a, b = result.stack.channels["A"], result.stack.channels["B"]
    inner = (slice(None), slice(25, -25), slice(25, -25))
    assert np.allclose(b[inner], a[inner] * 2 + 5, atol=1e-3)  # the same transform on both
    records = result.steps[0]["records"]
    assert [r["T"] for r in records] == [0, 1, 2, 3]
    for r, truth in zip(records[1:], sc.truth[1:]):
        assert abs(r["shift Y (px)"] + truth["shift"][0]) < 0.3 and abs(r["shift X (µm)"] + truth["shift"][1] * 0.5) < 0.15
        assert r["correlation after"] > r["correlation before"] and r["registered"]
    assert "4 frames along T" in result.steps[0]["summary"]


def test_step_on_every_layout():
    rng = np.random.default_rng(0)
    base = ndi.gaussian_filter(rng.random((2, 3, 4, 40, 48)), (0, 0, 0.6, 2, 2)).astype(np.float32)  # P T Z Y X
    base[:, 1] = np.roll(base[:, 0], (2, -3), axis=(-2, -1))
    base[:, 2] = np.roll(base[:, 0], (-1, 4), axis=(-2, -1))
    pipe = Pipeline()
    pipe.add("registration")
    result = pipe.run(Stack({"A": base}, "PTZYX", {"Z": 1.0, "Y": 0.5, "X": 0.5}))
    records = result.steps[0]["records"]
    assert len(records) == 6 and {r["P"] for r in records} == {0, 1}
    r = next(r for r in records if r["P"] == 1 and r["T"] == 1)
    # (np.roll wraps the edge round, which costs a little precision on so small a frame)
    assert abs(r["shift Y (px)"] + 2) < 0.25 and abs(r["shift X (px)"] - 3) < 0.25 and abs(r["shift Z (px)"]) < 0.25
    # Along Z: the planes of every time point and position.
    pipe.set_params(0, axis="Z (planes)", reference="Previous frame (cumulative)")
    records = pipe.run(Stack({"A": base}, "PTZYX", {"Z": 1.0, "Y": 0.5, "X": 0.5})).steps[0]["records"]
    assert len(records) == 2 * 3 * 4 and set(records[0]) >= {"P", "T", "Z", "shift Y (px)", "matrix"}
    with pytest.raises(ValueError, match="no T"):
        pipe.set_params(0, axis="T (time points)")
        pipe.run(Stack({"A": base[0, 0]}, "ZYX", {}))


def test_step_keeps_the_data_type_and_fills_edges():
    frames = np.zeros((2, 30, 30), np.uint16)
    frames[0, 10:20, 10:20] = 1000
    frames[1, 13:23, 8:18] = 1000
    pipe = Pipeline()
    pipe.add("registration")
    out = pipe.run(Stack({"A": frames}, "TYX", {})).stack.channels["A"]
    assert out.dtype == np.uint16 and np.array_equal(out[1], frames[0])
    pipe.set_params(0, fill="NaN (empty)")
    out = pipe.run(Stack({"A": frames}, "TYX", {})).stack.channels["A"]
    assert out.dtype == np.float32 and np.isnan(out[1, -1, 0])


def test_records_reapply_to_labels():
    sc = synth.motion_scene(seed=8, n=3)
    pipe = Pipeline()
    pipe.add("registration")
    result = pipe.run(Stack({"A": sc.frames}, "TYX", {"Y": 0.5, "X": 0.5}))
    meta = {"ibif_processing": [{"key": "registration", "params": result.steps[0]["params"],
                                 "records": result.steps[0]["records"]}]}
    records, along = R.records_of(meta)
    assert along == "T" and len(records) == 3
    labels = np.zeros(sc.frames.shape, np.uint16)
    labels[:, 60:80, 60:80] = 7
    moved = R.apply_records(labels, "TYX", records, along, labels=True)
    assert moved.dtype == np.uint16 and set(np.unique(moved)) == {0, 7}
    image_moved = R.apply_records(sc.frames, "TYX", records, along)
    assert np.allclose(image_moved, result.stack.channels["A"], atol=1e-4)
    assert R.records_of({}) == (None, None)
    with pytest.raises(ValueError, match="no registration for T 3"):
        R.apply_records(np.zeros((4, 8, 8)), "TYX", records, "T")


def test_registration_is_in_the_library():
    spec = _processing.get("registration")
    assert spec.available and spec.needs_reference and _processing.library()[0] is spec
