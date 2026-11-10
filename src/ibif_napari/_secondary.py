"""Secondary objects from a first segmentation (e.g. Cellpose nuclei).

- `expand`: grow every object by a distance, without overlapping its neighbours.
- `ring`: a band around every object (gap + width), or just inside its edge.
- `seeded_cells`: whole cells from seeds (nuclei) in another channel, by a seeded
  watershed inside a thresholded foreground (like CellProfiler's IdentifySecondaryObjects).
- `structures`: structures inside each object (vesicles, puncta, nucleoli, filaments,
  organelles) found in a channel. All the pieces inside object 17 together form object 17,
  so they link to it (and to everything else numbered 17) in the measurements.

Every secondary object keeps the label number of the object it came from, so nucleus 17
and cell 17 belong together. Distances are in physical units with per-axis `spacing`
(e.g. µm for Z, Y, X), so anisotropic 3D stacks are handled correctly.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

THRESHOLDS = ("otsu", "li", "triangle", "manual", "none")
CHANNEL_KINDS = ("membrane", "cytoplasm", "distance")
STRUCTURE_KINDS = ("spots", "filaments", "regions")
# Weight of the distance term in the cytoplasm watershed (per pixel, gradient scaled to 0-1).
# Chosen with the V&V suite's synthetic cells on independent seeds (docs/vv/README.md).
CYTOPLASM_COMPACTNESS = 0.03


def _spacing(labels, spacing):
    if spacing is None:
        return (1.0,) * labels.ndim
    spacing = tuple(float(s) for s in spacing)
    if len(spacing) != labels.ndim:
        raise ValueError(f"spacing has {len(spacing)} values for a {labels.ndim}D image")
    return spacing


def nearest_object(labels, spacing=None):
    """(label of the nearest object, distance to it) for every pixel; objects map to
    themselves at distance 0."""
    background = labels == 0
    if not background.any():
        return labels.copy(), np.zeros(labels.shape)
    if background.all():
        return labels.copy(), np.full(labels.shape, np.inf)
    distance, index = ndi.distance_transform_edt(background, sampling=_spacing(labels, spacing),
                                                 return_indices=True)
    return labels[tuple(index)], distance


def expand(labels, distance, spacing=None):
    """Grow every object by `distance`; where two objects meet, each pixel goes to the
    nearer one (no overlap, no merging)."""
    labels = np.asarray(labels)
    nearest, dist = nearest_object(labels, spacing)
    out = np.where(dist <= distance, nearest, 0)
    return out.astype(labels.dtype, copy=False)


def ring(labels, width, gap=0.0, spacing=None, inside=False):
    """A band of `width` around each object, starting `gap` from its edge.

    With `inside=True` the band is inside each object instead: the pixels within `width`
    of its edge (a rim or shell), with `gap` ignored.
    """
    labels = np.asarray(labels)
    sp = _spacing(labels, spacing)
    if inside:
        from skimage.segmentation import find_boundaries

        # Distance inside each object to its own edge (background or another object).
        edge = find_boundaries(labels, mode="inner") & (labels > 0)
        if not edge.any():
            return np.zeros_like(labels)
        dist = ndi.distance_transform_edt(~edge, sampling=sp)
        return np.where((labels > 0) & (dist < width), labels, 0).astype(labels.dtype, copy=False)
    nearest, dist = nearest_object(labels, sp)
    band = (labels == 0) & (dist > gap) & (dist <= gap + width)
    return np.where(band, nearest, 0).astype(labels.dtype, copy=False)


def combine_channels(channels):
    """Average of the channels after scaling each to 0-1 (0.5th-99.8th percentile), so a
    dim and a bright stain count equally."""
    out = None
    for c in channels:
        c = np.asarray(c, dtype=np.float32)
        lo, hi = np.percentile(c, (0.5, 99.8))
        scaled = np.clip((c - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(c)
        out = scaled if out is None else out + scaled
    if out is None:
        raise ValueError("choose at least one channel")
    return out / len(channels)


def foreground(image, method="otsu", value=None, factor=1.0):
    """Pixels belonging to cells: image above a threshold (times `factor`).
    `method` is otsu / li / triangle, manual (`value`), or none (everything)."""
    if method == "none":
        return np.ones(image.shape, bool), None
    if method == "manual":
        if value is None:
            raise ValueError("enter a threshold value")
        threshold = float(value)
    else:
        from skimage import filters

        func = {"otsu": filters.threshold_otsu, "li": filters.threshold_li,
                "triangle": filters.threshold_triangle}[method]
        threshold = float(func(image)) * factor
    return image > threshold, threshold


def seeded_cells(seeds, image, spacing=None, kind="membrane", threshold="otsu", value=None,
                 factor=1.0, max_distance=None, smooth=1.0):
    """Whole cells grown from `seeds` (e.g. nuclei labels) in `image` (one channel, or the
    combined channels), keeping each seed's label number.

    kind: "membrane" (bright cell borders: the watershed runs on the image; dark regions
          next to a seed are that cell's inside, other dark regions are outside the cells),
          "cytoplasm" (bright cell interior: a compact watershed on the image's gradient,
          with the seeds given the intensity of the cytoplasm next to them), or
          "distance" (ignore intensity: each foreground pixel goes to the nearest seed).
    The cells are limited to the foreground (`threshold`, see `foreground`; for membranes,
    plus the dark inside of each cell) and, if given, to `max_distance` from their seed. `smooth` is a Gaussian sigma in the same units as
    `spacing`. Returns (cells, threshold used).
    """
    from skimage import filters
    from skimage.segmentation import watershed

    seeds = np.asarray(seeds)
    image = np.asarray(image, dtype=np.float32)
    if image.shape != seeds.shape:
        raise ValueError(f"the channel ({image.shape}) and the seeds ({seeds.shape}) differ in size")
    if kind not in CHANNEL_KINDS:
        raise ValueError(f"unknown channel kind {kind!r}")
    sp = _spacing(seeds, spacing)
    if kind == "cytoplasm" and seeds.any() and not seeds.all():
        # Many cytoplasm stains leave the nucleus dark. Its edge would then be the steepest
        # gradient around each seed, and the first seed to cross its own edge would flood
        # the neighbouring cells. The seeds belong to their cells anyway, so give them the
        # intensity of the nearest cytoplasm before taking the gradient.
        _, index = ndi.distance_transform_edt(seeds > 0, sampling=sp, return_indices=True)
        image = image[tuple(index)]
    if smooth:
        image = ndi.gaussian_filter(image, sigma=[smooth / s for s in sp])
    fg, used = foreground(image, threshold, value, factor)
    fg |= seeds > 0
    nearest, dist = nearest_object(seeds, sp)
    within = dist <= max_distance if max_distance else None
    mask = fg if within is None else fg & within
    if not seeds.any():
        return np.zeros_like(seeds), used
    if kind == "distance":
        cells = np.where(mask, nearest, 0)
    elif kind == "membrane":
        cells = _membrane_watershed(image, seeds, fg, within)
    else:
        # Gradient scaled to ~0-1, plus a little distance from the seed (a compact
        # watershed, like CellProfiler's propagation): where neighbouring cells are about
        # equally bright, noise in the gradient no longer lets one cell flood the other.
        landscape = filters.sobel(image)
        top = float(np.percentile(landscape[mask], 99)) if mask.any() else 0.0
        if top > 0:
            landscape = landscape / top
        cells = watershed(landscape, markers=seeds, mask=mask, compactness=CYTOPLASM_COMPACTNESS)
    # A pixel is only kept if it is connected to its own seed (no stray islands).
    return _connected_to_seed(cells, seeds).astype(seeds.dtype, copy=False), used


def _membrane_watershed(image, seeds, foreground_mask, allowed=None):
    """Seeded watershed of a membrane image, with the dark outside flooded as well.

    In a membrane stain the inside of a cell is often as dark as the background, so the
    foreground holds little more than the bright membranes. A watershed limited to the
    foreground cannot cross the dark gap from a seed to its own membrane, while touching
    membranes still connect it to its neighbours' membranes, which it then takes. So here
    each dark region (outside the foreground) that touches a seed is taken as the inside of
    that cell, and every dark region that touches no seed is outside the cells and floods
    from its own marker (so an enclosed space with no seed, a lumen, stays background).
    Cells and outside meet on the ridge of the membrane, so the result hardly depends on the
    threshold; each cell then takes back the foreground pixels next to it on the outside of
    that ridge, one pixel deep, so its membrane is part of it. `allowed` (e.g. the maximum
    distance) limits where the cells may grow.
    """
    from skimage.segmentation import watershed

    dark, n = ndi.label(~foreground_mask)
    inside = np.zeros(n + 1, bool)
    inside[0] = True
    inside[np.unique(dark[ndi.binary_dilation(seeds > 0) & (dark > 0)])] = True
    outside = ~inside[dark]
    if allowed is not None:
        outside &= allowed
    if not outside.any():  # e.g. threshold "none", or no outside within the distance limit
        return watershed(image, markers=seeds, mask=allowed)
    markers = seeds.astype(np.int64)
    out_label = int(markers.max()) + 1
    markers[outside] = out_label
    cells = watershed(image, markers=markers, mask=allowed)
    cells[cells == out_label] = 0
    # The membrane itself is part of its cell: each foreground pixel the outside took next to
    # a cell (sharing a face with it) goes to that cell.
    if cells.any():
        dist, index = ndi.distance_transform_edt(cells == 0, return_indices=True)
        near = foreground_mask & (cells == 0) & (dist <= 1)
        if allowed is not None:
            near &= allowed
        cells[near] = cells[tuple(index)][near]
    return cells


def _connected_to_seed(cells, seeds):
    out = np.zeros_like(cells)
    for label, sl in enumerate(ndi.find_objects(cells), start=1):
        if sl is None:
            continue
        piece = cells[sl] == label
        parts, n = ndi.label(piece)
        if n > 1:
            keep = np.unique(parts[(seeds[sl] == label) & piece])
            piece = np.isin(parts, keep[keep > 0])
        out[sl][piece] = label
    return out


def cytoplasm(cells, nuclei):
    """Cells minus their nuclei."""
    return np.where(nuclei > 0, 0, cells).astype(cells.dtype, copy=False)


def structures(parents, image, spacing=None, kind="spots", size=1.0, threshold="otsu", value=None,
               factor=1.0, per_object=False, dark=False, smooth=0.0, min_size=0.0):
    """Structures inside each object of `parents` (e.g. vesicles in the cytoplasm, nucleoli
    in nuclei), found in `image`. Every structure pixel takes the number of the object it
    lies in, so all the pieces inside object 17 are one object, numbered 17.

    kind: "spots" (small bright blobs: vesicles, puncta, nucleoli; the local background,
          a Gaussian blur of width `size`, is subtracted first, so `size` should be about
          the diameter of the largest spot),
          "filaments" (thin lines: a ridge filter (Sato) tuned to lines `size` wide), or
          "regions" (any area brighter than the rest of the object, e.g. organelles; the
          smoothed intensity is thresholded as it is).
    dark: look for structures darker than their surroundings instead (e.g. nucleoli in a
          DNA stain).
    threshold: otsu / li / triangle (times `factor`) or manual (`value`), computed from the
          pixels inside all the objects together, or inside each object on its own with
          `per_object` (better when objects differ in brightness, but an object with no
          structures still gets some).
    min_size: pieces smaller than this (µm² / µm³ with `spacing`) are dropped.
    `size` and `smooth` are in the units of `spacing`. Returns (structures, thresholds used).
    """
    from skimage import filters

    parents = np.asarray(parents)
    image = np.asarray(image, dtype=np.float32)
    if image.shape != parents.shape:
        raise ValueError(f"the channel ({image.shape}) and the objects ({parents.shape}) differ in size")
    if kind not in STRUCTURE_KINDS:
        raise ValueError(f"unknown structure kind {kind!r}")
    if threshold not in ("otsu", "li", "triangle", "manual"):
        raise ValueError("choose a threshold to find the structures")
    sp = _spacing(parents, spacing)
    inside = parents > 0
    if not inside.any():
        return np.zeros_like(parents), []
    if dark:
        image = image.max() - image
    if smooth:
        image = ndi.gaussian_filter(image, sigma=[smooth / s for s in sp])
    if kind == "spots":
        background = ndi.gaussian_filter(image, sigma=[size / s for s in sp])
        response = np.clip(image - background, 0, None)
    elif kind == "filaments":
        response = _ridges(image, size / 2.0, sp)
    else:
        response = image

    def cut(values):
        if threshold == "manual":
            if value is None:
                raise ValueError("enter a threshold value")
            return float(value)
        if values.size < 2 or values.min() == values.max():
            return None
        func = {"otsu": filters.threshold_otsu, "li": filters.threshold_li,
                "triangle": filters.threshold_triangle}[threshold]
        return float(func(values)) * factor

    found = np.zeros(parents.shape, bool)
    used = []
    if per_object:
        for label, sl in enumerate(ndi.find_objects(parents), start=1):
            if sl is None:
                continue
            own = parents[sl] == label
            t = cut(response[sl][own])
            if t is not None:
                found[sl] |= own & (response[sl] > t)
                used.append(t)
    else:
        t = cut(response[inside])
        if t is not None:
            found = inside & (response > t)
            used.append(t)
    out = np.where(found, parents, 0).astype(parents.dtype, copy=False)
    if min_size and found.any():
        # Pieces in touching objects are split by number below.
        pieces, n = ndi.label(out > 0, structure=ndi.generate_binary_structure(out.ndim, out.ndim))
        pieces = np.where(out > 0, pieces * (int(out.max()) + 1) + out, 0)
        ids, inverse, counts = np.unique(pieces, return_inverse=True, return_counts=True)
        small = (counts * float(np.prod(sp)) < min_size) & (ids != 0)
        out[small[inverse.reshape(out.shape)]] = 0
    return out, used


def _ridges(image, radius, spacing):
    """Bright-line (ridge) response, Sato's filter at the line's radius. With voxels much
    longer in Z than across (Z step >= 2 x the pixel size) it runs per Z plane."""
    from skimage.filters import sato

    if image.ndim == 3 and spacing[0] >= 2 * max(spacing[1:]):
        sigma = max(radius / max(spacing[1:]), 0.5)
        return np.stack([sato(p, sigmas=[sigma], black_ridges=False) for p in image])
    sigma = max(radius / max(spacing), 0.5)
    return sato(image, sigmas=[sigma], black_ridges=False)


def count_pieces(labels):
    """{label: number of separate pieces}, e.g. vesicles per cell. Pixels that touch at a
    corner or edge belong to the same piece."""
    labels = np.asarray(labels)
    full = ndi.generate_binary_structure(labels.ndim, labels.ndim)
    out = {}
    for label, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is not None:
            out[label] = int(ndi.label(labels[sl] == label, structure=full)[1])
    return out
