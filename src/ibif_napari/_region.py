"""Restrict Cellpose to a sub-volume drawn as a Labels (painted) or Shapes layer.

Region layers can have any shape/scale: their extent is mapped to the image's pixel grid
through world coordinates (scale and translate). A 2D region applies to every Z plane.
"""
from __future__ import annotations

import numpy as np


def _to_world(layer, coords, n):
    """Data coordinates of the last `n` axes of `layer` -> world coordinates."""
    scale = np.asarray(layer.scale[-n:], float)
    translate = np.asarray(layer.translate[-n:], float)
    return np.asarray(coords, float) * scale + translate


def _to_image(image, world, n):
    scale = np.asarray(image.scale[-n:], float)
    translate = np.asarray(image.translate[-n:], float)
    return (np.asarray(world, float) - translate) / scale


def region_crop(image, axes, region, z_range=None):
    """Pixel ranges {axis: (start, stop)} of `image` covered by `region`.

    `axes` are the image's axis letters (e.g. "TZYX"); Y and X always come from the region,
    Z from a 3D labels region painted on several planes and/or `z_range` (inclusive
    (first, last) planes), whichever is narrower. Returns None if the region is empty.
    """
    shape = dict(zip(axes, image.data.shape))
    crop = {}
    if region is not None:
        lo, hi = _region_extent(region)
        if lo is None:
            return None
        n = len(lo)
        lo_w, hi_w = _to_world(region, lo, n), _to_world(region, hi, n)
        img_axes = "ZYX"[-n:] if n <= 3 else None
        lo_i, hi_i = _to_image(image, lo_w, n), _to_image(image, hi_w, n)
        for a, l, h in zip(img_axes, lo_i, hi_i):
            if a not in shape:
                continue
            # Pixels whose centres fall inside the extent (napari: pixel i is centred on i).
            start = int(np.clip(np.ceil(min(l, h) - 1e-6), 0, shape[a]))
            stop = int(np.clip(np.floor(max(l, h) + 1e-6) + 1, 0, shape[a]))
            if stop <= start:
                return None
            crop[a] = (start, stop)
    if "Z" in shape and z_range is not None:
        first, last = z_range
        start, stop = crop.get("Z", (0, shape["Z"]))
        start, stop = max(start, first), min(stop, last + 1)
        if stop <= start:
            return None
        crop["Z"] = (start, stop)
    return crop


def _region_extent(region):
    """(lower, upper) corners of the region in its own data coordinates (upper exclusive),
    over its spatial axes: YX for shapes and 2D labels, ZYX for 3D labels."""
    if region._type_string == "shapes":
        if not len(region.data):
            return None, None
        vertices = np.concatenate([np.asarray(s)[:, -2:] for s in region.data])
        return vertices.min(axis=0), vertices.max(axis=0)
    data = np.asarray(region.data)
    n = 3 if data.ndim >= 3 else 2
    mask = _project(data != 0, n)
    if not mask.any():
        return None, None
    idx = np.nonzero(mask)
    lo = np.array([i.min() for i in idx], float) - 0.5
    hi = np.array([i.max() for i in idx], float) + 0.5
    if n == 3 and lo[0] + 1 == hi[0]:
        # Painted on a single plane of a 3D layer: treat it as a 2D outline for all planes.
        return lo[1:], hi[1:]
    return lo, hi


def _project(mask, n):
    """Collapse leading (time/position) axes so only the last n remain."""
    while mask.ndim > n:
        mask = mask.any(axis=0)
    return mask


def painted_mask(image, region, crop):
    """Painted area of a labels region on the image's grid inside `crop`, or None (shapes
    regions use their bounding box only).

    A region painted in 3D (different shapes on different planes) gives a Z Y X mask; one
    painted on a single plane, or copied to every plane of the Z range, gives a Y X mask
    used on every plane.
    """
    if region is None or region._type_string != "labels":
        return None
    mask = _project(np.asarray(region.data) != 0, 2 if np.ndim(region.data) < 3 else 3)
    if mask.ndim == 3:
        painted = mask[mask.any(axis=(1, 2))]
        if "Z" in crop and len(painted) > 1 and not (painted == painted[:1]).all():
            return _resample(image, region, mask, [crop["Z"], crop["Y"], crop["X"]])
        mask = mask.any(axis=0)
    return _resample(image, region, mask, [crop["Y"], crop["X"]])


def _resample(image, region, mask, ranges):
    """`mask` (the region's last len(ranges) axes) on the image's pixels in `ranges`."""
    n = len(ranges)
    out = np.ones([b - a for a, b in ranges], bool)
    index = []
    for k, (a, b) in enumerate(ranges):
        axis = k - n  # from the end
        world = np.arange(a, b) * image.scale[axis] + image.translate[axis]
        r = np.round((world - region.translate[axis]) / region.scale[axis]).astype(int)
        inside = (r >= 0) & (r < mask.shape[axis])
        shape = [1] * n
        shape[k] = len(r)
        out &= inside.reshape(shape)
        index.append(np.clip(r, 0, mask.shape[axis] - 1))
    return out & mask[np.ix_(*index)]


painted_mask_yx = painted_mask  # older name


def keep_inside(labels, mask_yx, fraction=0.5):
    """Remove objects with less than `fraction` of their pixels inside the mask: a Y X
    mask applies to every Z plane, a Z Y X mask to the volume."""
    if mask_yx is None or not labels.any():
        return labels
    inside_mask = np.broadcast_to(mask_yx, labels.shape)
    n = int(labels.max()) + 1
    total = np.bincount(labels.ravel(), minlength=n)
    inside = np.bincount(labels[inside_mask].ravel(), minlength=n)
    keep = inside >= fraction * np.maximum(total, 1)
    keep[0] = False
    lut = np.where(keep, np.arange(n), 0).astype(labels.dtype)
    return lut[labels]


def add_region_layer(viewer, image, axes, z_range, on_range_grown=None):
    """Add a labels layer over `image` (axis letters `axes`) to paint a region on.

    For a Z stack the layer is 3D: what is painted on one plane is shown (and used) on
    exactly the planes of `z_range`, until it is painted with the 3D brush on the image
    (then it keeps the shape painted). Returns (layer, ZExtendedRegion or None).
    """
    spatial = [axes.index(a) for a in ("ZYX" if "Z" in axes else "YX")]
    shape = tuple(image.data.shape[i] for i in spatial)
    region = viewer.add_labels(
        np.zeros(shape, dtype=np.uint8),
        name=f"{image.name} region",
        scale=[image.scale[i] for i in spatial],
        translate=[image.translate[i] for i in spatial],
        opacity=0.4,
        metadata={"ibif_region": True},
    )
    region.brush_size = max(10, min(shape[-2:]) // 15)
    region.mode = "paint"
    sync = ZExtendedRegion(region, z_range, on_range_grown) if "Z" in axes else None
    return region, sync


class ZExtendedRegion:
    """Keep a 3D (Z, Y, X) region labels layer showing one painted footprint on exactly
    the planes of a Z range.

    Painting or erasing on any plane in the range changes the footprint on all of them;
    painting on a plane outside the range calls `on_range_grown(first, last)` so the
    range can be widened to include it. Call `set_range` when the range changes.

    Once the layer is painted in 3D (metadata "ibif_region_3d", set by the 3D brush) it
    is left as painted.
    """

    def __init__(self, layer, z_range, on_range_grown=None):
        self.layer = layer
        self.on_range_grown = on_range_grown
        self.footprint = np.zeros(layer.data.shape[-2:], dtype=layer.data.dtype)
        self.z_range = None
        self.set_range(*z_range)
        layer.events.paint.connect(self._on_paint)
        # Undo/redo (Ctrl+Z) restore only the plane that was edited and emit no event:
        # re-sync the other planes afterwards.
        for name in ("undo", "redo"):
            original = getattr(layer, name)

            def wrapped(original=original):
                original()
                self._on_paint()

            setattr(layer, name, wrapped)

    def _planes(self):
        if self.z_range is None:
            return range(0)
        first, last = self.z_range
        return range(max(first, 0), min(last, self.layer.data.shape[0] - 1) + 1)

    @property
    def free(self):
        return bool((self.layer.metadata or {}).get("ibif_region_3d"))

    def set_range(self, first, last):
        new = (first, last) if first <= last else None
        if new == self.z_range or (self.free and self.z_range is not None):
            self.z_range = new
            return
        self.z_range = new
        data = self.layer.data
        data[:] = 0
        planes = self._planes()
        if len(planes):
            data[planes.start : planes.stop] = self.footprint
        self.layer.refresh()

    def _on_paint(self, event=None):
        if self.free:
            return
        data = self.layer.data
        # napari may announce an edit before writing it (single edits) or after (brush
        # strokes); applying its (indices, old, new) records first handles both.
        for atom in getattr(event, "value", None) or []:
            if isinstance(atom, tuple) and len(atom) == 3:
                indices, _, value = atom
                data[indices] = value
        planes = self._planes()
        fp = self.footprint
        new = fp.copy()
        inside = data[planes.start : planes.stop] if len(planes) else data[:0]
        for plane in inside[(inside != fp).any(axis=(1, 2))]:
            changed = plane != fp
            new[changed] = plane[changed]
        outside = [z for z in range(data.shape[0]) if z not in planes and data[z].any()]
        for z in outside:
            painted = data[z] != 0
            new[painted] = data[z][painted]
        changed = new != fp
        self.footprint = new
        if outside:
            first = min([*outside, *([planes.start] if len(planes) else [])])
            last = max([*outside, *([planes.stop - 1] if len(planes) else [])])
            if self.on_range_grown is not None:
                self.on_range_grown(first, last)  # normally calls set_range
            if self.z_range != (first, last):
                self.set_range(first, last)
                return
        if changed.any() and len(planes):
            data[planes.start : planes.stop, changed] = new[changed]
            self.layer.refresh()
