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


def painted_mask_yx(image, region, crop):
    """Painted area of a labels region on the image's YX grid inside `crop`, or None
    (shapes regions use their bounding box only)."""
    if region is None or region._type_string != "labels":
        return None
    mask = _project(np.asarray(region.data) != 0, 2 if np.ndim(region.data) < 3 else 3)
    if mask.ndim == 3:
        mask = mask.any(axis=0)
    (y0, y1), (x0, x1) = crop["Y"], crop["X"]
    yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    world = _to_world(image, np.stack([yy, xx], axis=-1), 2)
    ry, rx = np.round(_to_image(region, world, 2)).astype(int).transpose(2, 0, 1)
    inside = (ry >= 0) & (ry < mask.shape[0]) & (rx >= 0) & (rx < mask.shape[1])
    out = np.zeros(yy.shape, bool)
    out[inside] = mask[ry[inside], rx[inside]]
    return out


def keep_inside(labels, mask_yx, fraction=0.5):
    """Remove objects with less than `fraction` of their pixels inside `mask_yx`
    (applied to every Z plane)."""
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
