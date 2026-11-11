"""Classical segmentation: threshold, then split touching objects by watershed.

No deep learning, no GPU: an alternative to Cellpose for testing, for bright well
separated objects (nuclei, spots, beads), or when no GPU or server is available.
2D images and 3D stacks (with their voxel size) are handled the same way:

1. **Local contrast** (optional): CLAHE, contrast-limited adaptive histogram
   equalisation (Zuiderveld 1994), evens out bright and dim regions so one threshold
   fits the whole image.
2. **Smoothing**: a Gaussian blur (sigma in pixels; Z follows the voxel size) removes
   noise that would otherwise break objects up.
3. **Threshold**: a global automatic threshold (Otsu 1979, Li, triangle, Yen, mean) times
   a factor, a *local* (adaptive) threshold for uneven backgrounds, or a manual value.
4. **Clean up**: fill holes, remove specks smaller than the minimum size.
5. **Split touching objects** (watershed, Beucher & Meyer 1993):
   - *distance*: seeds at the centres of the objects (maxima of the distance to the
     background), so round objects that touch are cut along their neck;
   - *intensity*: seeds at the brightest points (e.g. nuclei brighter in the middle);
   - *none*: each connected component is one object (connected-components labelling).
   Seeds closer than half the object diameter are merged, which stops one object being
   cut into several.
6. Remove objects smaller than the minimum size (and, optionally, those touching the
   image border) and number them 1, 2, 3...
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

THRESHOLDS = ("otsu", "li", "triangle", "yen", "mean", "local", "manual")
SPLITS = ("distance", "intensity", "none")


@dataclass
class Settings:
    clahe: bool = False
    clahe_clip: float = 0.05  # CLAHE clip limit (0-1; higher = more contrast)
    clahe_tile: int = 0  # CLAHE tile size in pixels (0 = 1/8 of the image)
    smooth: float = 1.0  # Gaussian sigma in XY pixels (0 = none)
    threshold: str = "otsu"
    factor: float = 1.0  # global thresholds are multiplied by this
    value: float | None = None  # manual threshold (in the image's own units)
    block: int = 51  # local threshold: neighbourhood size in pixels (odd)
    offset: float = -0.05  # local threshold: subtracted from the local mean (0-1 scale);
    # negative = a pixel must be that much brighter than its neighbourhood
    fill_holes: bool = True
    split: str = "distance"
    diameter: float = 0.0  # typical object diameter in XY pixels (0 = from the objects)
    min_size: int = 15  # pixels / voxels
    clear_border: bool = False


def segment(image, spacing=None, settings=None, **kwargs):
    """Labels (uint32, 0 = background) of a 2D or 3D image, and a dict of what was
    used (the threshold, the number of objects). `spacing` is the voxel size per axis
    (Z larger than XY in most stacks); `settings` a `Settings`, or its fields as kwargs."""
    s = settings or Settings(**kwargs)
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(f"segment a 2D or 3D image, not {image.ndim}D")
    spacing = np.asarray(spacing if spacing is not None else (1.0,) * image.ndim, float)
    xy = float(spacing[-1]) or 1.0
    rel = spacing / xy  # each axis in XY pixels (Z is usually > 1)
    fg, info, work = foreground(image, s, rel)
    labels = split(fg, work, s, rel)
    labels = _finish(labels, s)
    info["objects"] = int(labels.max())
    return labels, info


def segment_planes(stack, spacing=None, settings=None, stitch=0.3, **kwargs):
    """Each Z plane of a stack segmented in 2D, then objects on neighbouring planes that
    overlap by at least `stitch` (IoU) given one number, like Cellpose's stitching.
    For stacks with few planes or very anisotropic voxels, where 3D smoothing and
    distances blur across planes."""
    s = settings or Settings(**kwargs)
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("segment_planes needs a Z stack")
    # One threshold for the whole stack (a plane above or below the cells would
    # otherwise get a threshold in its noise), with smoothing within the planes only.
    rel = np.array([np.inf, 1.0, 1.0])
    fg, info, work = foreground(stack, s, rel, planes=True)
    out = np.zeros(stack.shape, np.uint32)
    for z in range(stack.shape[0]):
        plane = _finish(split(fg[z], work[z], s, rel[1:]), s)
        if z and stitch > 0 and plane.any():
            top = int(out[:z].max())
            plane = np.where(plane > 0, plane + top, 0).astype(np.uint32)
            plane = _stitch(out[z - 1], plane, stitch)
        out[z] = plane
    from skimage.segmentation import relabel_sequential

    out, _, _ = relabel_sequential(out)
    info["objects"] = int(out.max())
    return out.astype(np.uint32), info


def _stitch(previous, plane, min_iou):
    """Give each object of `plane` the number of the object on the plane before that it
    overlaps most, when their IoU is at least `min_iou` (each old number used once)."""
    both = (previous > 0) & (plane > 0)
    if not both.any():
        return plane
    pairs, overlap = np.unique(np.stack([plane[both], previous[both]]), axis=1, return_counts=True)
    size_new = np.bincount(plane.ravel())
    size_old = np.bincount(previous.ravel())
    iou = overlap / (size_new[pairs[0]] + size_old[pairs[1]] - overlap)
    lut = np.arange(int(plane.max()) + 1, dtype=np.uint32)
    taken = set()
    for k in np.argsort(-iou):
        new, old = int(pairs[0][k]), int(pairs[1][k])
        if iou[k] < min_iou:
            break
        if lut[new] != new or old in taken:
            continue
        lut[new] = old
        taken.add(old)
    return lut[plane]


def foreground(image, settings=None, rel=None, planes=False, **kwargs):
    """(mask, info, processed image): the pixels above the threshold, after local
    contrast, smoothing, hole filling and removing specks. `planes`: a stack whose
    planes are handled one by one (no smoothing or neighbourhoods across Z)."""
    s = settings or Settings(**kwargs)
    image = np.asarray(image)
    rel = np.ones(image.ndim) if rel is None else np.asarray(rel, float)
    work, lo, hi = _normalise(image)
    if s.clahe:
        from skimage import exposure

        tile = s.clahe_tile or None
        if tile:  # in XY pixels; fewer planes per tile in Z when Z steps are larger
            tile = [1 if np.isinf(r) else max(2, int(round(tile / r))) for r in rel]
        work = exposure.equalize_adapthist(work, kernel_size=tile, clip_limit=s.clahe_clip)
    if s.smooth > 0:
        work = ndi.gaussian_filter(work, sigma=[s.smooth / r for r in rel])
    info = {}
    if s.threshold == "manual":
        if s.value is None:
            raise ValueError("enter a threshold value")
        level = (float(s.value) - lo) / (hi - lo)  # the value in the image's own units
        info["threshold"] = float(s.value)
        fg = work > level
    elif s.threshold == "local":
        from skimage.filters import threshold_local

        block = int(s.block) | 1  # odd
        size = [1 if np.isinf(r) else max(3, int(round(block / r)) | 1) for r in rel]
        level = threshold_local(work, block_size=size, offset=s.offset)
        fg = work > level
        info["threshold"] = "local"
    else:
        from skimage import filters

        func = {"otsu": filters.threshold_otsu, "li": filters.threshold_li,
                "triangle": filters.threshold_triangle, "yen": filters.threshold_yen,
                "mean": filters.threshold_mean}[s.threshold]
        level = float(func(work)) * s.factor
        fg = work > level
        # In the image's units when no contrast change was made (else on the 0-1 scale).
        info["threshold"] = level * (hi - lo) + lo if not s.clahe else level
    if s.fill_holes:
        fg = _fill_holes(fg)
    if s.min_size > 0:
        fg = np.stack([_remove_small(p, s.min_size) for p in fg]) if planes else _remove_small(fg, s.min_size)
    info["foreground"] = float(fg.mean()) if fg.size else 0.0
    return fg, info, work


def split(fg, image, settings=None, rel=None, **kwargs):
    """Labels of the foreground: connected components, or a seeded watershed."""
    from skimage.segmentation import watershed

    s = settings or Settings(**kwargs)
    rel = np.ones(fg.ndim) if rel is None else np.asarray(rel, float)
    components, n = ndi.label(fg, structure=ndi.generate_binary_structure(fg.ndim, 1))
    if s.split == "none" or n == 0:
        return components.astype(np.uint32)
    if s.split == "distance":
        landscape = ndi.distance_transform_edt(fg, sampling=rel)
        # A little smoothing stops one object getting several centres on a ragged edge.
        landscape = ndi.gaussian_filter(landscape, sigma=[0.5 / r for r in rel])
    elif s.split == "intensity":
        landscape = np.asarray(image, float)
    else:
        raise ValueError(f"unknown split method {s.split!r}")
    diameter = s.diameter or _typical_diameter(components, n, rel)
    markers = _seeds(landscape, components, diameter, rel)
    labels = watershed(-landscape, markers=markers, mask=fg)
    # A component that got no seed (too small or flat) stays one object.
    missing = (labels == 0) & fg
    if missing.any():
        extra, _ = ndi.label(missing)
        labels = np.where(missing, extra + labels.max(), labels)
    return labels.astype(np.uint32)


def _seeds(landscape, components, diameter, rel):
    """One marker per object centre: maxima of `landscape` at least half a diameter
    apart (per axis, in pixels), within the components."""
    from skimage.feature import peak_local_max

    radius = max(1.0, diameter / 2.0)  # centres closer than half a diameter are one object
    footprint = _ellipsoid([max(1, int(round(radius / r))) for r in rel])
    peaks = peak_local_max(landscape, footprint=footprint, labels=components, exclude_border=False)
    markers = np.zeros(landscape.shape, bool)
    markers[tuple(peaks.T)] = True
    # Plateaus give several neighbouring peaks: one marker each.
    markers, _ = ndi.label(ndi.binary_dilation(markers, structure=ndi.generate_binary_structure(landscape.ndim, 2)))
    return markers


def _typical_diameter(components, n, rel):
    """Median equivalent diameter of the connected components (in XY pixels)."""
    sizes = np.bincount(components.ravel())[1:]
    sizes = sizes[sizes > 0] * float(np.prod(rel))
    if not len(sizes):
        return 10.0
    v = float(np.median(sizes))
    return float(np.cbrt(6 * v / np.pi)) if components.ndim == 3 else float(np.sqrt(4 * v / np.pi))


def _ellipsoid(radii):
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
    return sum((g / max(r, 1e-9)) ** 2 for g, r in zip(grids, radii)) <= 1.0


def _normalise(image):
    """The image on a 0-1 scale (0.1-99.9 percentiles), and those percentiles."""
    image = image.astype(np.float32, copy=False)
    finite = image[np.isfinite(image)]
    if not finite.size:
        return np.zeros(image.shape, np.float32), 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(finite, (0.1, 99.9)))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0), lo, hi


def _fill_holes(fg):
    """Fill holes in 2D, and in each Z plane of a stack (a hole in 3D must be closed in
    every direction, which nuclei imaged with gaps between planes often are not)."""
    if fg.ndim == 2:
        return ndi.binary_fill_holes(fg)
    out = ndi.binary_fill_holes(fg)
    for z in range(out.shape[0]):
        out[z] = ndi.binary_fill_holes(out[z])
    return out


def _remove_small(fg, min_size):
    labels, n = ndi.label(fg)
    if n == 0:
        return fg
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def _finish(labels, s):
    if s.clear_border:
        from skimage.segmentation import clear_border

        labels = clear_border(labels)
    if s.min_size > 0 and labels.max():
        sizes = np.bincount(labels.ravel())
        small = sizes < s.min_size
        small[0] = False
        labels = np.where(small[labels], 0, labels)
    from skimage.segmentation import relabel_sequential

    labels, _, _ = relabel_sequential(labels.astype(np.uint32))
    return labels.astype(np.uint32)


__all__ = ["Settings", "segment", "segment_planes", "foreground", "split", "THRESHOLDS", "SPLITS"]
