"""Stack registration: correct movement between time points or between Z planes.

Following IMART (Dunn et al. 2014, IntraVital 3:e28210), the correction is made in
stages: a rigid registration of each frame (translation, or translation and rotation),
optionally followed by an affine refinement and a non-rigid (dense, smooth) correction
of what is left, such as tissue warping in intravital imaging. Here it is a new
implementation in Python, with scikit-image and SciPy:

* translation: phase correlation with sub-pixel upsampling (Guizar-Sicairos et al.
  2008, `skimage.registration.phase_cross_correlation`);
* rotation (rigid): a coarse angle from the polar transform of the Fourier magnitudes
  (which a translation does not change; Reddy & Chatterji 1996), then angle and shift
  refined together by maximising the normalised cross-correlation with the reference;
* affine: the rigid result refined to a full affine transform the same way;
* non-rigid: optical flow (iterative Lucas-Kanade, `optical_flow_ilk`) between the
  rigidly registered frame and the reference, smoothed, and applied on top.

The transform is measured on one channel (the reference channel, optionally smoothed)
and applied to every channel, so channels stay aligned with each other. Along T each
frame is a whole Z stack (or plane) and gets one transform; along Z each plane gets
its own 2D transform. In 3D (T registration of Z stacks) rotation and affine act in
the XY plane, the translation in Z, Y and X.

Transforms map each output (registered) pixel to where it is read in the input, as
`scipy.ndimage.affine_transform` takes them: homogeneous matrices in pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from . import register
from .base import Param, StepSpec

AXES = {"T (time points)": "T", "Z (planes)": "Z"}
MODELS = {"Translation": "translation", "Rigid (translation + rotation)": "rigid",
          "Affine": "affine", "Non-rigid (after rigid)": "nonrigid"}
REFERENCES = {"First frame": "first", "Mean of the first frames": "mean", "Chosen frame": "chosen",
              "Previous frame (cumulative)": "previous"}
FILLS = {"Zero": "zero", "Nearest edge value": "nearest", "NaN (empty)": "nan"}
INTERPOLATION = {"Linear": 1, "Cubic": 3}
SPATIAL = "ZYX"


# ---- transforms -----------------------------------------------------------------------------


@dataclass
class Transform:
    """`matrix` ((n+1) x (n+1), pixels) maps output coordinates to input ones; `flow`, if
    given, is a further displacement field (n, *shape), in pixels, read before the matrix
    (non-rigid). `angle` (degrees, in the YX plane) and `shift` (pixels, ZYX or YX) describe
    the rigid part for the records; `ok` is False when the frame could not be registered
    (it is then left as it was)."""

    matrix: np.ndarray
    flow: np.ndarray | None = None
    angle: float = 0.0
    shift: tuple = ()
    ok: bool = True

    @property
    def ndim(self):
        return self.matrix.shape[0] - 1

    @classmethod
    def identity(cls, ndim):
        return cls(np.eye(ndim + 1), shift=(0.0,) * ndim)


def rigid_matrix(shape, angle_deg=0.0, shift=(0.0, 0.0), spacing=None, linear=None):
    """The output->input matrix of: rotate the input by `angle_deg` about the centre of
    the YX plane (positive = clockwise as displayed, with Y down), apply `linear` (a 2x2 YX matrix,
    instead of the rotation, for affine), then shift by `shift` (pixels, one per axis).
    `spacing` (per axis) makes the rotation round in µm when pixels are not square."""
    n = len(shape)
    sp = np.ones(n) if spacing is None else np.asarray(spacing, float)
    centre = (np.asarray(shape, float) - 1) / 2.0
    forward = np.eye(n)  # input -> output, in µm, YX block only
    if linear is None:
        a = np.deg2rad(angle_deg)
        linear = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    forward[-2:, -2:] = linear
    # In pixels: S^-1 F S, S = diag(spacing).
    forward_px = np.diag(1 / sp) @ forward @ np.diag(sp)
    inverse = np.linalg.inv(forward_px)
    m = np.eye(n + 1)
    m[:n, :n] = inverse
    m[:n, n] = centre - inverse @ (centre + np.asarray(shift, float))
    return m


def warp(image, transform, order=1, fill="zero"):
    """`image` moved by `transform` (output = input read at matrix @ output, after the flow)."""
    image = np.asarray(image)
    nan = fill == "nan"
    if np.issubdtype(image.dtype, np.floating):
        src = image
    else:  # 32-bit labels need float64 to keep every number exact
        src = image.astype(np.float64 if image.dtype.itemsize >= 4 else np.float32)
    mode = "nearest" if fill == "nearest" else "constant"
    cval = np.nan if nan else 0.0
    m = transform.matrix
    if transform.flow is None:
        if np.allclose(m, np.eye(m.shape[0])):
            out = src.copy()
        else:
            out = ndi.affine_transform(src, m[:-1, :-1], m[:-1, -1], order=order, mode=mode, cval=cval,
                                       prefilter=order > 1)
    else:
        grid = np.indices(image.shape, dtype=np.float32)
        coords = grid + transform.flow  # where the flow reads, in the rigidly registered frame
        coords = np.tensordot(m[:-1, :-1], coords, axes=1) + m[:-1, -1].reshape((-1,) + (1,) * image.ndim)
        out = ndi.map_coordinates(src, coords, order=order, mode=mode, cval=cval, prefilter=order > 1)
    if not nan and not np.issubdtype(image.dtype, np.floating):
        info = np.iinfo(image.dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(image.dtype)
    return out


def ncc(a, b, mask=None):
    """Normalised cross-correlation of two images (over `mask`, and where both are finite)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        keep &= mask
    if keep.sum() < 4:
        return 0.0
    a, b = a[keep] - a[keep].mean(), b[keep] - b[keep].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


# ---- estimating ---------------------------------------------------------------------------


def _prepare(image, smooth_px):
    image = np.asarray(image, np.float32)
    if np.any(smooth_px):
        image = ndi.gaussian_filter(image, smooth_px)
    finite = np.isfinite(image)
    if not finite.all():
        image = np.where(finite, image, np.nanmean(image) if finite.any() else 0.0)
    return image


def _translation(reference, moving, upsample):
    """Shift (pixels) that registers `moving` onto `reference`: phase correlation of the
    mean-subtracted frames under a Hann window (which stops the edges, where content
    enters and leaves the field, from biasing it), to 1/`upsample` of a pixel.
    The window pulls the estimate a little towards zero when the frames differ in where
    their bright content sits under it, so the frame is first moved by the whole pixels
    found (exactly, without interpolation) and the small remainder measured again."""
    from skimage.registration import phase_cross_correlation

    import warnings

    if not np.ptp(reference) or not np.ptp(moving):  # an empty frame (e.g. a plane shifted out)
        return np.zeros(reference.ndim)
    w = _window(reference.shape)
    ref = (reference - reference.mean()) * w

    def measure(mov):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # its error estimate, which is not used
            shift, _, _ = phase_cross_correlation(ref, (mov - mov.mean()) * w,
                                                  upsample_factor=max(1, int(upsample)), normalization=None)
        return np.asarray(shift, float)

    coarse = np.round(measure(moving)).astype(int)
    if not coarse.any():
        return measure(moving)
    moved = ndi.shift(moving, coarse, order=0, mode="nearest")  # whole pixels: exact
    return coarse + measure(moved)


def _window(shape):
    """A Hann window over every axis of 8 or more pixels (a short axis, such as a few Z
    planes, would lose most of its planes to it)."""
    w = np.ones(shape)
    for ax, n in enumerate(shape):
        h = np.hanning(n) if n >= 8 else np.ones(n)
        w = w * h.reshape([-1 if i == ax else 1 for i in range(len(shape))])
    return w


def _coarse_angle(reference, moving):
    """Rotation of `moving` relative to `reference` (degrees, the YX plane, within ±90),
    from the polar transform of their Fourier magnitudes, which a shift does not change."""
    from skimage.registration import phase_cross_correlation
    from skimage.transform import warp_polar

    def spectrum(img):
        img = img - img.mean()
        mag = np.abs(np.fft.fftshift(np.fft.fft2(img * _window(img.shape))))
        return np.log1p(mag)

    a, b = spectrum(reference), spectrum(moving)
    radius = min(a.shape) // 2 - 1
    pa = warp_polar(a, radius=radius, output_shape=(720, radius))[:, radius // 8: radius // 2]
    pb = warp_polar(b, radius=radius, output_shape=(720, radius))[:, radius // 8: radius // 2]
    shift, _, _ = phase_cross_correlation(pa, pb, upsample_factor=10, normalization=None)
    angle = float(shift[0]) * 360.0 / 720.0
    return (angle + 90.0) % 180.0 - 90.0  # the magnitudes cannot tell θ from θ + 180°


def _refine(reference, moving, shape, spacing, params0, model, crop):
    """Angle and shift (rigid), or the full 2x2 linear part and shift (affine), that
    maximise the correlation of the warped `moving` with `reference` (2D, YX)."""
    from scipy.optimize import minimize

    sl = tuple(slice(c, n - c) for c, n in zip(crop, shape))
    ref = reference[sl]

    def matrix(p):
        if model == "affine":
            return rigid_matrix(shape, shift=p[4:6], spacing=spacing,
                                linear=np.array([[1 + p[0], p[1]], [p[2], 1 + p[3]]]))
        return rigid_matrix(shape, p[0], p[1:3], spacing)

    def cost(p):
        out = warp(moving, Transform(matrix(p)), order=1, fill="nan")[sl]
        return 1.0 - ncc(ref, out)

    step = np.array([0.5, 0.5, 0.5]) if model == "rigid" else np.array([0.01] * 4 + [0.5, 0.5])
    res = minimize(cost, params0, method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-7, "maxfev": 600 if model == "rigid" else 1500,
                            "direc": np.diag(step)})
    return res.x, matrix(res.x)


def estimate(reference, moving, model="translation", spacing=None, smooth=0.0, upsample=20,
             max_shift=0.0):
    """The `Transform` that registers `moving` onto `reference` (2D YX or 3D ZYX, the same
    shape). `smooth` (µm) blurs both first; `max_shift` (µm, 0 = no limit) marks a larger
    shift as a failed registration (identity, ok=False). Non-rigid gets its flow from
    `nonrigid_flow` separately."""
    reference = np.asarray(reference)
    moving = np.asarray(moving)
    if reference.shape != moving.shape:
        raise ValueError(f"the frames differ in size: {reference.shape} and {moving.shape}")
    nd = reference.ndim
    sp = np.ones(nd) if spacing is None else np.asarray(spacing, float)
    smooth_px = [smooth / s for s in sp] if smooth else 0
    ref, mov = _prepare(reference, smooth_px), _prepare(moving, smooth_px)
    shift = _translation(ref, mov, upsample)
    angle = 0.0
    if model == "translation":
        m = rigid_matrix(ref.shape, 0.0, shift, sp)
    else:
        # The rotation is measured in the YX plane: on the frames themselves in 2D, on their
        # maximum projections over Z in 3D.
        ref2, mov2 = (ref, mov) if nd == 2 else (ref.max(axis=0), mov.max(axis=0))
        sp2 = sp[-2:]
        angle = _coarse_angle(ref2, mov2)
        rotated = warp(mov2, Transform(rigid_matrix(mov2.shape, angle, (0, 0), sp2)), fill="nearest")
        shift2 = _translation(ref2, rotated, upsample)
        crop = [max(2, int(0.1 * n)) for n in ref2.shape]
        p, m2 = _refine(ref2, mov2, ref2.shape, sp2, np.r_[angle, shift2], "rigid", crop)
        angle = float(p[0])
        if model == "affine":
            a = np.deg2rad(angle)
            r = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
            p, m2 = _refine(ref2, mov2, ref2.shape, sp2, np.r_[(r - np.eye(2)).ravel(), p[1:3]], "affine", crop)
        if nd == 2:
            m = m2
            shift = np.asarray(p[-2:], float)
        else:
            # Z shift: phase correlation after the XY correction.
            xy = np.eye(4)
            xy[1:3, 1:3] = m2[:2, :2]
            xy[1:3, 3] = m2[:2, 2]
            moved = warp(mov, Transform(xy), fill="nearest")
            dz = _translation(ref, moved, upsample)[0]
            m = xy.copy()
            m[0, 3] = -dz
            shift = np.r_[dz, p[-2:]]
    t = Transform(m, angle=angle, shift=tuple(float(s) for s in shift))
    if max_shift and float(np.sqrt(np.sum((np.asarray(t.shift) * sp) ** 2))) > max_shift:
        return Transform(np.eye(nd + 1), angle=t.angle, shift=t.shift, ok=False)
    return t


def nonrigid_flow(reference, moving, spacing=None, smooth=1.0, radius=7, max_displacement=0.0):
    """A smooth displacement field (pixels, (n, *shape)) that moves `moving` (already
    rigidly registered) onto `reference`: optical flow (iterative Lucas-Kanade), smoothed
    by a Gaussian of `smooth` µm, and capped at `max_displacement` µm (0 = no cap)."""
    from skimage.registration import optical_flow_ilk

    sp = np.ones(np.ndim(reference)) if spacing is None else np.asarray(spacing, float)
    ref = _prepare(reference, 0)
    mov = _prepare(moving, 0)
    lo, hi = np.percentile(ref, (0.5, 99.5))
    scale = (hi - lo) or 1.0
    flow = optical_flow_ilk((ref - lo) / scale, (mov - lo) / scale, radius=int(radius), num_warp=10,
                            prefilter=True)
    flow = np.asarray(flow, np.float32)
    if smooth:
        flow = np.stack([ndi.gaussian_filter(f, [smooth / s for s in sp]) for f in flow])
    if max_displacement:
        size = np.sqrt(sum((f * s) ** 2 for f, s in zip(flow, sp)))
        over = size > max_displacement
        if over.any():
            flow = flow * np.where(over, max_displacement / np.maximum(size, 1e-9), 1.0)
    return flow


def _compose(first, then):
    """The transform that applies `first`, then `then` (output->input matrices compose
    the other way round)."""
    return Transform(first.matrix @ then.matrix, angle=first.angle + then.angle,
                     shift=tuple(np.add(first.shift, then.shift)), ok=first.ok and then.ok)


# ---- a series of frames ---------------------------------------------------------------------


def register_frames(frames, model="translation", reference="first", frame=0, n_mean=5, spacing=None,
                    smooth=0.0, upsample=20, max_shift=0.0, flow_smooth=1.0, flow_radius=7,
                    max_displacement=0.0, progress=None):
    """Transforms (one per frame) that register `frames` (N, *spatial) to the reference:
    the first frame, the mean of the first `n_mean`, frame `frame`, or the previous frame
    (chained, so slow drift is followed). `progress(i, n)` is called per frame."""
    frames = np.asarray(frames)
    n = frames.shape[0]
    nd = frames.ndim - 1
    rigid_model = "rigid" if model == "nonrigid" else model
    if reference == "mean":
        ref = frames[: max(1, min(n, int(n_mean)))].astype(np.float32).mean(axis=0)
    elif reference == "chosen":
        if not 0 <= int(frame) < n:
            raise ValueError(f"reference frame {frame} is outside 0-{n - 1}")
        ref = frames[int(frame)]
    else:
        ref = frames[0]
    out = []
    previous = Transform.identity(nd)
    for i in range(n):
        if progress is not None:
            progress(i, n)
        if reference == "previous":
            if i == 0:
                out.append(Transform.identity(nd))
                continue
            step = estimate(frames[i - 1], frames[i], rigid_model, spacing, smooth, upsample, max_shift)
            t = _compose(previous, step)
            previous = t
        elif reference == "chosen" and i == int(frame) or (reference == "first" and i == 0):
            t = Transform.identity(nd)
        else:
            t = estimate(ref, frames[i], rigid_model, spacing, smooth, upsample, max_shift)
        if model == "nonrigid" and t.ok:
            target = ref if reference != "previous" else frames[0]
            rigid = warp(frames[i], t, fill="nan")
            filled = np.where(np.isfinite(rigid), rigid, np.nanmean(rigid))
            t = Transform(t.matrix, nonrigid_flow(target, filled, spacing, flow_smooth, flow_radius,
                                                  max_displacement), t.angle, t.shift, t.ok)
        out.append(t)
    return out


# ---- the processing step --------------------------------------------------------------------


def _series(axes, along):
    """Indices to loop over (every axis that is not spatial and not `along`)."""
    return [i for i, a in enumerate(axes) if a not in SPATIAL and a != along]


def register_stack(stack, params, progress):
    """The Processing step: registers `stack` along T or Z; returns (stack, records,
    summary)."""
    along = AXES[params["axis"]]
    model = MODELS[params["model"]]
    reference = REFERENCES[params["reference"]]
    fill = FILLS[params["fill"]]
    order = INTERPOLATION[params["interpolation"]]
    axes = stack.axes
    if along not in axes:
        raise ValueError(f"the image has no {params['axis']} axis to register along (its axes are {axes})")
    t_axis = axes.index(along)
    spatial = [i for i, a in enumerate(axes) if a in SPATIAL and a != along]
    if along == "T" and not spatial:
        raise ValueError("registration needs spatial axes")
    loops = _series(axes, along) if along == "T" else [i for i, a in enumerate(axes) if a not in "YX" and a != "Z"]
    spacing = [stack.spacing.get(axes[i], 1.0) for i in spatial]
    channels = {name: np.array(a, copy=True) if fill != "nan" else np.asarray(a, np.float32).copy()
                for name, a in stack.channels.items()}
    ref_data = np.asarray(stack.channels[stack.reference])
    loop_shape = [ref_data.shape[i] for i in loops]
    records = []
    series = list(np.ndindex(*loop_shape))
    total = len(series) * ref_data.shape[t_axis]
    done = 0
    for pos in series:
        index = [slice(None)] * len(axes)
        for i, p in zip(loops, pos):
            index[i] = p
        index = tuple(index)
        # The frames of this series, with the registered axis first.
        kept = [i for i in range(len(axes)) if i not in loops]
        move_from = kept.index(t_axis)
        frames = np.moveaxis(ref_data[index], move_from, 0)

        def tick(i, n, done=done):
            progress((done + i) / total, f"frame {i + 1} of {n}" + (f", series {series.index(pos) + 1} of "
                                                                     f"{len(series)}" if len(series) > 1 else ""))

        transforms = register_frames(
            frames, model, reference, params["frame"], params["n_mean"], spacing, params["smooth"],
            params["upsample"], params["max_shift"], params["flow_smooth"], params["flow_radius"],
            params["max_displacement"], tick)
        done += frames.shape[0]
        for name, data in channels.items():
            block = np.moveaxis(np.asarray(stack.channels[name])[index], move_from, 0)
            moved = np.stack([warp(f, t, order, fill) for f, t in zip(block, transforms)])
            data[index] = np.moveaxis(moved, 0, move_from).astype(data.dtype, copy=False)
        ref_after = np.moveaxis(channels[stack.reference][index], move_from, 0)
        target = (frames[0] if reference in ("first", "previous") else
                  frames[int(params["frame"])] if reference == "chosen" else
                  frames[: max(1, int(params["n_mean"]))].astype(np.float32).mean(axis=0))
        target_after = (ref_after[0] if reference in ("first", "previous") else
                        ref_after[int(params["frame"])] if reference == "chosen" else
                        np.nanmean(ref_after[: max(1, int(params["n_mean"]))].astype(np.float32), axis=0))
        for k, t in enumerate(transforms):
            row = {axes[i]: int(p) for i, p in zip(loops, pos)}
            row[along] = k
            for a, s, sp in zip([axes[i] for i in spatial], t.shift, spacing):
                row[f"shift {a} (px)"] = float(s)
                row[f"shift {a} (µm)"] = float(s) * sp
            if model != "translation":
                row["rotation (°)"] = float(t.angle)
            if t.flow is not None:
                size = np.sqrt(sum((f * s) ** 2 for f, s in zip(t.flow, spacing)))
                row["non-rigid mean (µm)"] = float(size.mean())
                row["non-rigid max (µm)"] = float(size.max())
            row["matrix"] = [[float(v) for v in r] for r in t.matrix]  # output -> input, pixels
            row["correlation before"] = ncc(target, frames[k])
            row["correlation after"] = ncc(target_after, ref_after[k])
            row["registered"] = bool(t.ok)
            records.append(row)
    progress(1.0, "done")
    shifts = [np.sqrt(sum(r[k] ** 2 for k in r if k.startswith("shift") and k.endswith("(µm)"))) for r in records]
    failed = sum(not r["registered"] for r in records)
    before = np.mean([r["correlation before"] for r in records])
    after = np.mean([r["correlation after"] for r in records])
    summary = (f"{len(records)} frames along {along}, largest shift {max(shifts):.2f} µm, correlation with "
               f"the reference {before:.3f} → {after:.3f}" + (f"; {failed} not registered (over the "
                                                                f"maximum shift)" if failed else ""))
    return stack.with_channels(channels), records, summary


def records_of(layer_metadata):
    """The registration records ([{..., "matrix"}]) a processed layer carries, and the
    axis they were registered along; (None, None) if it has none."""
    for step in reversed((layer_metadata or {}).get("ibif_processing") or []):
        if step.get("key") == "registration" and step.get("records"):
            return step["records"], AXES.get(step["params"]["axis"], "T")
    return None, None


def apply_records(data, axes, records, along, labels=False, fill="zero"):
    """`data` (with `axes`, e.g. "TZYX") moved frame by frame by the rigid / affine
    transforms in `records` (from a registration of an image of the same size): to bring
    a segmentation or another channel into line with the registered image. Labels are
    moved with nearest-neighbour interpolation, so they keep their numbers. The non-rigid
    part of a registration is not stored, so it is not applied."""
    data = np.asarray(data)
    if along not in axes:
        raise ValueError(f"the layer has no {along} axis")
    if len(axes) != data.ndim:
        raise ValueError(f"axes {axes!r} do not match the data's {data.ndim} dimensions")
    loops = [i for i, a in enumerate(axes) if a not in SPATIAL and a != along] if along == "T" else \
        [i for i, a in enumerate(axes) if a not in "YX" and a != "Z"]
    by_key = {}
    for row in records:
        by_key[tuple(int(row.get(axes[i], 0)) for i in loops) + (int(row[along]),)] = np.asarray(row["matrix"])
    out = np.empty_like(data)
    t_axis = axes.index(along)
    for pos in np.ndindex(*[data.shape[i] for i in loops]):
        for k in range(data.shape[t_axis]):
            index = [slice(None)] * data.ndim
            for i, p in zip(loops, pos):
                index[i] = p
            index[t_axis] = k
            index = tuple(index)
            m = by_key.get(tuple(pos) + (k,))
            if m is None:
                raise ValueError(f"no registration for {along} {k}" + (f" at {pos}" if pos else ""))
            frame = data[index]
            if m.shape[0] - 1 != frame.ndim:
                raise ValueError("the registration was made on frames of another number of dimensions")
            out[index] = warp(frame, Transform(m), order=0 if labels else 1, fill="zero" if labels else fill)
    return out


STEP = register(StepSpec(
    key="registration",
    title="Stack registration",
    category="Motion",
    needs_reference=True,
    run=register_stack,
    description=(
        "Corrects movement between time points (drift, breathing, heartbeat in intravital "
        "imaging) or between the planes of a Z stack. As in IMART, a rigid registration of "
        "each frame comes first, optionally refined to affine or followed by a smooth non-rigid "
        "correction. It is measured on the reference channel and applied to every channel."),
    params=(
        Param("axis", "Correct along", "choice", "T (time points)", "Register time points to each "
              "other (each a whole Z stack), or the planes of each Z stack", options=tuple(AXES)),
        Param("model", "Transform", "choice", "Translation", "Translation: shifts only. Rigid: "
              "shifts and rotation. Affine: also scaling and shear. Non-rigid: rigid, then a smooth "
              "warp for tissue that deforms", options=tuple(MODELS)),
        Param("reference", "Reference", "choice", "First frame", "What each frame is registered to; "
              "Previous frame follows slow drift but errors add up", options=tuple(REFERENCES)),
        Param("frame", "Reference frame", "int", 0, "Frame to register to with Reference: Chosen "
              "frame (0 = the first)", minimum=0, maximum=100000, step=1, decimals=0),
        Param("n_mean", "Frames averaged", "int", 5, "With Reference: Mean of the first frames",
              minimum=1, maximum=1000, step=1, decimals=0),
        Param("smooth", "Smoothing", "float", 0.0, "Gaussian blur of the reference channel before "
              "measuring (not of the result); helps with noisy images", minimum=0.0, maximum=50.0,
              step=0.1, unit="µm"),
        Param("upsample", "Sub-pixel precision", "int", 20, "Shifts are measured to 1/this of a pixel",
              minimum=1, maximum=100, step=1, decimals=0),
        Param("max_shift", "Maximum shift", "float", 0.0, "A frame that seems to move further is left "
              "as it is and reported (0 = no limit)", minimum=0.0, maximum=10000.0, step=1.0, unit="µm"),
        Param("flow_smooth", "Warp smoothness", "float", 2.0, "Non-rigid: Gaussian smoothing of the "
              "warp; larger = smoother, more rigid", minimum=0.0, maximum=100.0, step=0.5, unit="µm"),
        Param("flow_radius", "Warp window", "int", 7, "Non-rigid: window of the optical flow, in "
              "pixels", minimum=2, maximum=50, step=1, decimals=0),
        Param("max_displacement", "Maximum warp", "float", 0.0, "Non-rigid: largest local "
              "correction (0 = no limit)", minimum=0.0, maximum=1000.0, step=0.5, unit="µm"),
        Param("fill", "Edges", "choice", "Zero", "What fills the edge uncovered by the shift",
              options=tuple(FILLS)),
        Param("interpolation", "Interpolation", "choice", "Linear", options=tuple(INTERPOLATION)),
    ),
))
