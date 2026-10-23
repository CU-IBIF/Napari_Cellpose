"""Filament tracing: tube enhancement, skeletons, statistics and click-to-trace paths.

Plain numpy/scipy/scikit-image/skan functions on one 2D (YX) or 3D (ZYX) volume with
voxel sizes in µm, so they can be tested without Qt (and later run elsewhere).

Pipeline: `enhance` (multi-scale Hessian "tubeness", correct for anisotropic voxels)
-> `segment` (threshold) -> `skeleton_from_mask` (skeletonize, prune spurs, drop short
pieces) -> `measure` (per filament / segment / point statistics and Sholl). `trace_path`
finds the brightest path between two clicked points for click-to-trace.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

BLOCK_VOXELS = 2_000_000  # filtered at a time (in Y blocks): bounds memory to ~0.5 GB
SEGMENT_TYPES = {0: "isolated", 1: "terminal", 2: "internal", 3: "loop"}


# ---- enhancement and segmentation ---------------------------------------------------


def sigmas(d_min, d_max):
    """Gaussian scales (µm) that respond best to filaments d_min..d_max µm across."""
    d_min, d_max = sorted((max(d_min, 1e-3), max(d_max, 1e-3)))
    n = 1 if d_max / d_min < 1.2 else min(5, int(math.ceil(math.log2(d_max / d_min))) + 1)
    return list(np.geomspace(d_min, d_max, n) / (2 * math.sqrt(2)))


def _gaussian_kernels(sigma):
    """Sampled Gaussian and its 1st/2nd derivative kernels (pixels), corrected so that a
    flat background gives exactly 0 and a ramp / parabola its true slope / curvature.
    (Plain sampled derivative kernels leak the background in at sub-voxel sigmas, which
    are typical along Z.)"""
    r = max(int(4 * sigma + 0.5), 2)
    x = np.arange(-r, r + 1, dtype=float)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    d1 = x * g
    d1 /= np.sum(x * d1)
    d2 = (x**2 / sigma**2 - 1) * g
    d2 -= g * d2.sum()
    d2 /= np.sum(x**2 * d2) / 2
    return g, d1, d2


def _tubeness(block, spacing, sigma):
    """Bright-tube response at one scale (µm), from the Hessian in µm units.

    3D: sqrt(l1*l2) where the two largest-magnitude eigenvalues are negative (as Fiji's
    Tubeness), damped towards 0 for blobs (all three negative and similar). 2D: -l1.
    """
    nd = block.ndim
    kernels = [_gaussian_kernels(max(sigma / sp, 0.5)) for sp in spacing]
    h = {}
    for i in range(nd):
        for j in range(i, nd):
            out = block
            for axis in range(nd):
                order = (axis == i) + (axis == j)
                out = ndi.correlate1d(out, kernels[axis][order], axis=axis, mode="nearest")
            h[i, j] = out.astype(np.float64) * (sigma**2 / (spacing[i] * spacing[j]))
    if nd == 2:
        mean = (h[0, 0] + h[1, 1]) / 2
        l1 = mean - np.hypot((h[0, 0] - h[1, 1]) / 2, h[0, 1])
        return np.maximum(-l1, 0).astype(np.float32)
    l1, l2, l3 = _eigvalsh3(h[0, 0], h[0, 1], h[0, 2], h[1, 1], h[1, 2], h[2, 2])
    tube = np.where((l1 < 0) & (l2 < 0), np.sqrt(np.abs(l1 * l2)), 0)
    blob = np.where(l3 < 0, np.clip(l3 / np.minimum(l2, -1e-12), 0, 1), 0)
    return (tube * (1 - blob)).astype(np.float32)


def _eigvalsh3(a00, a01, a02, a11, a12, a22):
    """Eigenvalues (ascending) of symmetric 3x3 matrices given by their 6 entries, in
    closed form (much faster than numpy's per-matrix LAPACK calls)."""
    q = (a00 + a11 + a22) / 3
    p1 = a01**2 + a02**2 + a12**2
    b00, b11, b22 = a00 - q, a11 - q, a22 - q
    p = np.sqrt((b00**2 + b11**2 + b22**2 + 2 * p1) / 6)
    safe = np.where(p > 0, p, 1)
    det = (b00 * (b11 * b22 - a12**2) - a01 * (a01 * b22 - a12 * a02) + a02 * (a01 * a12 - b11 * a02))
    r = np.clip(det / (2 * safe**3), -1, 1)
    phi = np.arccos(r) / 3
    high = q + 2 * p * np.cos(phi)
    low = q + 2 * p * np.cos(phi + 2 * np.pi / 3)
    return low, 3 * q - high - low, high


def enhance(volume, spacing, d_min, d_max):
    """Multi-scale tubeness of `volume`, scaled to 0..1 (1 = the brightest filaments)."""
    steps = iter_enhance(volume, spacing, d_min, d_max)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def iter_enhance(volume, spacing, d_min, d_max):
    """`enhance` as a generator: yields (blocks done, total) and returns the result."""
    vol = np.asarray(volume, np.float32)
    spacing = [float(s) for s in spacing]
    scales = sigmas(d_min, d_max)
    out = np.zeros(vol.shape, np.float32)
    ny = vol.shape[-2]
    margin = int(math.ceil(4 * max(scales) / spacing[-2])) + 1
    rows = max(32, BLOCK_VOXELS // max(vol.size // ny, 1))
    starts = range(0, ny, rows)
    for n, y0 in enumerate(starts):
        y1 = min(y0 + rows, ny)
        a, b = max(y0 - margin, 0), min(y1 + margin, ny)
        block = vol[..., a:b, :]
        best = np.zeros(block.shape, np.float32)
        for s in scales:
            np.maximum(best, _tubeness(block, spacing, s), out=best)
        out[..., y0:y1, :] = best[..., y0 - a : y0 - a + (y1 - y0), :]
        yield n + 1, len(starts)
    positive = out[out > 0]
    top = float(np.percentile(positive, 99.9)) if positive.size else 0.0
    if top > 0:
        np.clip(out / top, 0, 1, out=out)
    return out


def auto_threshold(enhanced):
    """Otsu threshold of the filament response (0..1), ignoring the empty background."""
    from skimage.filters import threshold_otsu

    values = enhanced[enhanced > 0]
    if values.size < 2 or values.min() == values.max():
        return 0.5
    return float(threshold_otsu(values))


def segment(enhanced, threshold=None, spacing=None, max_hole_um=None):
    """Filament mask: response above `threshold` (None = automatic). Returns (mask, threshold).

    Holes in a plane up to `max_hole_um` across are filled: they would otherwise become
    small loops in the skeleton. Larger enclosed areas (filaments around a cell) are kept.
    """
    if threshold is None:
        threshold = auto_threshold(enhanced)
    mask = enhanced > threshold
    labels, n = ndi.label(mask, np.ones((3,) * mask.ndim))
    if n:
        sizes = np.bincount(labels.ravel())
        mask = (sizes >= 4)[labels] & mask  # specks of noise
    if max_hole_um and spacing is not None:
        max_px = max_hole_um**2 / (spacing[-1] * spacing[-2])
        for plane in mask.reshape(-1, *mask.shape[-2:]):
            holes, n = ndi.label(ndi.binary_fill_holes(plane) & ~plane)
            if n:
                small = np.bincount(holes.ravel()) <= max_px
                small[0] = False
                plane |= small[holes]
    return mask, threshold


# ---- skeletons ------------------------------------------------------------------------


def _skan():
    from skan import Skeleton, summarize

    return Skeleton, summarize


def _summary(skel, spacing):
    Skeleton, summarize = _skan()
    s = Skeleton(skel, spacing=spacing)
    return s, summarize(s, separator="_")


def _drop_short(skel, spacing, min_length):
    """Remove skeleton pieces (connected components) shorter than `min_length` µm."""
    labels, n = ndi.label(skel, np.ones((3,) * skel.ndim))
    if not n:
        return skel
    lengths = np.zeros(n + 1)
    if np.bincount(labels.ravel())[1:].max() > 1:
        _, df = _summary(skel, spacing)
        src = df[[f"image_coord_src_{i}" for i in range(skel.ndim)]].to_numpy(int)
        np.add.at(lengths, labels[tuple(src.T)], df["branch_distance"].to_numpy())
    keep = lengths >= max(min_length, 1e-9)
    keep[0] = False
    return keep[labels]


def skeleton_from_mask(mask, spacing, prune=1.0, min_length=2.0):
    """Centre lines of `mask`: spurs shorter than `prune` µm are trimmed (the shortest at
    each branch point first, repeatedly) and pieces shorter than `min_length` µm dropped."""
    from skimage.morphology import skeletonize

    skel = skeletonize(mask)
    for _ in range(25):
        skel = _drop_short(skel, spacing, min_length)
        if prune <= 0 or not skel.any():
            break
        s, df = _summary(skel, spacing)
        spurs = df[(df["branch_type"] == 1) & (df["branch_distance"] < prune)]
        removed = set()
        for i in spurs.sort_values("branch_distance").index:
            ids = s.path(i)
            junction = ids[0] if s.degrees[ids[0]] > 2 else ids[-1]
            if junction in removed:
                continue  # one per branch point per round, so the longest arm survives
            removed.add(junction)
            pixels = s.coordinates[ids[s.degrees[ids] <= 2]]
            skel[tuple(pixels.T)] = False
        if not removed:
            break
        skel = skeletonize(skel)
    return skel


# ---- measurements ---------------------------------------------------------------------


@dataclass
class Result:
    """Statistics of a traced skeleton. Lengths in µm; positions in µm in the image's
    coordinates (pixel * spacing + origin)."""

    labels: np.ndarray  # the skeleton as given, labelled by filament (0 = dropped)
    filaments: pd.DataFrame
    segments: pd.DataFrame
    points: pd.DataFrame  # root / branch / terminal points; pixel columns for display
    sholl: pd.DataFrame
    spacing: tuple
    origin: tuple
    # For SWC export: skeleton voxels, their graph, radii (µm) and each filament's root.
    coordinates: np.ndarray = field(repr=False, default=None)
    graph: object = field(repr=False, default=None)
    radius: np.ndarray = field(repr=False, default=None)
    pixel_filament: np.ndarray = field(repr=False, default=None)
    roots: dict = field(default_factory=dict)


def _axes(nd):
    return "zyx"[-nd:]


def _empty(skel, spacing, origin):
    ax = _axes(skel.ndim)
    return Result(
        labels=np.zeros(skel.shape, np.uint32),
        filaments=pd.DataFrame(columns=FILAMENT_COLUMNS),
        segments=pd.DataFrame(columns=SEGMENT_COLUMNS),
        points=pd.DataFrame(columns=["filament", "type", *[f"{a}_um" for a in ax]]),
        sholl=pd.DataFrame(columns=["filament", "radius_um", "intersections"]),
        spacing=tuple(spacing),
        origin=tuple(origin),
    )


FILAMENT_COLUMNS = [
    "filament", "length_um", "segments", "branch_points", "terminal_points",
    "mean_diameter_um", "volume_um3", "area_um2", "mean_intensity", "max_branch_depth",
    "straightness",
]
SEGMENT_COLUMNS = [
    "filament", "segment", "type", "length_um", "straight_um", "straightness",
    "mean_diameter_um", "volume_um3", "area_um2", "mean_intensity", "branch_depth",
    "distance_from_root_um",
]


def measure(skel, spacing, intensity=None, mask=None, origin=None, sholl_step=1.0):
    """Statistics of the skeleton `skel` (any non-zero voxel; thinned for measuring).

    `intensity` gives mean intensities; `mask` (the segmented filaments) gives diameters,
    volumes and areas, which are NaN without it. Each filament's beginning point (root)
    is its thickest node when diameters are known (usually the soma), otherwise one end
    of its longest path. Branch depth counts branch points between the root and a
    segment. Sholl counts crossings of spheres every `sholl_step` µm around the root.
    """
    import networkx as nx
    from skimage.morphology import skeletonize

    nd = skel.ndim
    sp = np.asarray(spacing, float)
    origin = np.zeros(nd) if origin is None else np.asarray(origin, float)
    given = np.asarray(skel) != 0
    skel = skeletonize(given)
    labels, n = ndi.label(skel, np.ones((3,) * nd))
    # Lone voxels have no segments.
    sizes = np.bincount(labels.ravel())
    if n:
        skel &= (sizes > 1)[labels]
        labels, n = ndi.label(skel, np.ones((3,) * nd))
    if not n:
        return _empty(skel, spacing, origin)
    source = None if intensity is None else np.asarray(intensity, np.float32)
    s, df = _summary(skel, tuple(sp))
    coords = s.coordinates
    deg = s.degrees
    pix_fil = labels[tuple(coords.T)]
    xy = float(sp[-2:].min())
    if mask is not None:
        edt = ndi.distance_transform_edt(np.asarray(mask, bool), sampling=sp)
        d = edt[tuple(coords.T)]
        # A filament w voxels wide has its centre w/2 + 1/2 voxels from the background.
        radius = np.where(d > 0, np.maximum(d - xy / 2, xy / 2), np.nan)
    else:
        radius = np.full(len(coords), np.nan)

    # Junctions are often clusters of adjacent voxels: count each cluster once.
    jmask = np.zeros(skel.shape, bool)
    jmask[tuple(coords[deg > 2].T)] = True
    jlabels, n_junctions = ndi.label(jmask, np.ones((3,) * nd))
    cluster = np.where(deg > 2, jlabels[tuple(coords.T)], 0)  # per voxel, 0 = not junction

    rows = []
    graph = nx.MultiGraph()
    for i in range(len(df)):
        ids = s.path(i)
        a, b = int(ids[0]), int(ids[-1])
        if cluster[a] and cluster[a] == cluster[b] and len(ids) <= 3:
            continue  # inside one junction cluster
        p = coords[ids] * sp
        steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
        length = float(steps.sum())
        r = radius[ids]
        rm = (r[1:] + r[:-1]) / 2
        straight = float(np.linalg.norm(p[-1] - p[0]))
        row = dict(
            filament=int(pix_fil[a]),
            type=SEGMENT_TYPES.get(int(df["branch_type"].iat[i]), "?"),
            length_um=length,
            straight_um=straight,
            straightness=straight / length if length > 0 else np.nan,
            mean_diameter_um=float(2 * np.sum(rm * steps) / length) if length > 0 else np.nan,
            volume_um3=float(np.pi * np.sum(rm**2 * steps)),
            area_um2=float(2 * np.pi * np.sum(rm * steps)),
            mean_intensity=float(source[tuple(coords[ids].T)].mean()) if source is not None else np.nan,
            _src=a,
            _dst=b,
            _path=i,
        )
        if nd == 3:
            dz, dy, dx = p[-1] - p[0]
            row["elevation_deg"] = float(np.degrees(np.arctan2(abs(dz), np.hypot(dy, dx))))
        for name, pt in (("start", p[0]), ("end", p[-1])):
            for ax, v in zip(_axes(nd), pt + origin):
                row[f"{name}_{ax}_um"] = float(v)
        rows.append(row)
        graph.add_edge(a, b, key=len(rows) - 1, weight=max(length, 1e-9))
    seg = pd.DataFrame(rows)

    # Roots, distances and branch depth, per filament.
    roots, depth_of, dist_of = {}, {}, {}
    for fil in range(1, n + 1):
        nodes = [v for v in graph.nodes if pix_fil[v] == fil]
        if not nodes:
            continue
        sub = graph.subgraph(nodes)
        if not np.all(np.isnan(radius[nodes])):
            root = nodes[int(np.nanargmax(radius[nodes]))]
        else:
            ends = [v for v in nodes if deg[v] == 1] or nodes
            far = nx.single_source_dijkstra_path_length(sub, ends[0])
            root = max(far, key=far.get)
        dist, paths = nx.single_source_dijkstra(sub, root)
        roots[fil] = int(root)
        for v, path in paths.items():
            dist_of[v] = dist[v]
            seen = {cluster[u] for u in path[1:] if cluster[u]} - {cluster[root]}
            depth_of[v] = len(seen)
    if len(seg):
        near = [
            src if dist_of.get(src, np.inf) <= dist_of.get(dst, np.inf) else dst
            for src, dst in zip(seg["_src"], seg["_dst"])
        ]
        seg["branch_depth"] = [depth_of.get(v, 0) for v in near]
        seg["distance_from_root_um"] = [dist_of.get(v, np.nan) for v in near]
        seg.insert(1, "segment", np.arange(1, len(seg) + 1))

    # Points: one per junction cluster, every terminal voxel, and the roots.
    points = []
    for c in range(1, n_junctions + 1):
        members = np.nonzero(cluster == c)[0]
        points.append(("branch", int(pix_fil[members[0]]), coords[members].mean(axis=0)))
    for v in np.nonzero(deg == 1)[0]:
        points.append(("terminal", int(pix_fil[v]), coords[v].astype(float)))
    for fil, v in roots.items():
        points.append(("root", fil, coords[v].astype(float)))
    pts = pd.DataFrame(
        [
            {"filament": f, "type": t, **{f"{a}_um": float(v) for a, v in zip(_axes(nd), c * sp + origin)},
             **{f"_{a}_px": float(v) for a, v in zip(_axes(nd), c)}}
            for t, f, c in points
        ]
    )

    fil_rows = []
    for fil in range(1, n + 1):
        part = seg[seg["filament"] == fil] if len(seg) else seg
        mine = pts[pts["filament"] == fil] if len(pts) else pts
        length = float(part["length_um"].sum()) if len(part) else 0.0
        w = part["length_um"].to_numpy() if len(part) else np.zeros(0)

        def weighted(col):
            v = part[col].to_numpy(float) if len(part) else np.zeros(0)
            ok = ~np.isnan(v)
            return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() and w[ok].sum() > 0 else np.nan

        ends = mine[mine["type"] == "terminal"] if len(mine) else mine
        span = np.nan
        if len(ends) >= 2:
            e = ends[[f"{a}_um" for a in _axes(nd)]].to_numpy()
            span = float(np.max(np.linalg.norm(e[:, None] - e[None], axis=-1)))
        fil_rows.append(dict(
            filament=fil,
            length_um=length,
            segments=len(part),
            branch_points=int((mine["type"] == "branch").sum()) if len(mine) else 0,
            terminal_points=len(ends),
            mean_diameter_um=weighted("mean_diameter_um"),
            volume_um3=float(part["volume_um3"].sum()) if len(part) else 0.0,
            area_um2=float(part["area_um2"].sum()) if len(part) else 0.0,
            mean_intensity=weighted("mean_intensity"),
            max_branch_depth=int(part["branch_depth"].max()) if len(part) else 0,
            # Distance between the two farthest ends / total length (1 = straight line).
            straightness=span / length if length > 0 and not np.isnan(span) else np.nan,
            **({f"root_{a}_um": float(v) for a, v in zip(_axes(nd), coords[roots[fil]] * sp + origin)}
               if fil in roots else {}),
        ))
    fils = pd.DataFrame(fil_rows)

    sholl = _sholl(s, coords, sp, pix_fil, roots, seg, sholl_step)
    seg = seg.drop(columns=["_src", "_dst", "_path"]) if len(seg) else pd.DataFrame(columns=SEGMENT_COLUMNS)
    # Label the voxels as given (painted or traced, before thinning) by filament.
    given_labels, n_given = ndi.label(given, np.ones((3,) * nd))
    lut = np.zeros(n_given + 1, np.uint32)
    lut[given_labels[skel]] = labels[skel]
    return Result(
        labels=lut[given_labels], filaments=fils, segments=seg, points=pts, sholl=sholl,
        spacing=tuple(sp), origin=tuple(origin), coordinates=coords, graph=s.graph,
        radius=radius, pixel_filament=pix_fil, roots=roots,
    )


def _sholl(s, coords, sp, pix_fil, roots, seg, step):
    """Crossings of spheres (circles in 2D) every `step` µm around each filament's root."""
    if not step or step <= 0 or not len(seg):
        return pd.DataFrame(columns=["filament", "radius_um", "intersections"])
    counts = {}
    for fil, path_i in zip(seg["filament"], seg["_path"]):
        root = coords[roots[fil]] * sp
        d = np.linalg.norm(coords[s.path(path_i)] * sp - root, axis=1)
        shell = np.floor(d / step).astype(int)
        lo, hi = np.minimum(shell[:-1], shell[1:]), np.maximum(shell[:-1], shell[1:])
        c = counts.setdefault(fil, {})
        for a, b in zip(lo[lo != hi], hi[lo != hi]):
            for k in range(a + 1, b + 1):
                c[k] = c.get(k, 0) + 1
    rows = [
        dict(filament=fil, radius_um=k * step, intersections=c.get(k, 0))
        for fil, c in sorted(counts.items()) if c
        for k in range(1, max(c) + 1)
    ]
    return pd.DataFrame(rows, columns=["filament", "radius_um", "intersections"])


# ---- click to trace -------------------------------------------------------------------


def snap(point, volume, skeleton=None, spacing=(1, 1, 1), radius_um=1.0, z_planes=2):
    """Move a clicked voxel onto the existing skeleton if one is within `radius_um`,
    otherwise to the brightest voxel within ±`z_planes` planes (clicks are on one plane)."""
    p = np.round(np.asarray(point, float)).astype(int)
    p = np.clip(p, 0, np.array(volume.shape) - 1)
    sp = np.asarray(spacing, float)
    if skeleton is not None:
        r = np.ceil(radius_um / sp).astype(int)
        lo, hi = np.maximum(p - r, 0), np.minimum(p + r + 1, skeleton.shape)
        near = np.argwhere(np.asarray(skeleton[tuple(slice(a, b) for a, b in zip(lo, hi))]) != 0)
        if len(near):
            d = np.linalg.norm((near + lo - p) * sp, axis=1)
            if d.min() <= radius_um:
                return near[int(np.argmin(d))] + lo
    if volume.ndim == 3 and z_planes:
        z0, z1 = max(p[0] - z_planes, 0), min(p[0] + z_planes + 1, volume.shape[0])
        column = np.asarray(volume[z0:z1, p[1], p[2]], float)
        p[0] = z0 + int(np.argmax(column))
    return p


def trace_path(volume, spacing, start, end, d_min, d_max, threshold=None):
    """Brightest path between two voxels, following the filament response.

    Only a box around the two points (plus a margin) is filtered, so each click is quick.
    Returns (path voxels (N, ndim), box start, mask of the box) - the mask (filament
    response above `threshold`, or automatic) gives the traced filament's diameter.
    """
    from skimage.graph import MCP_Geometric

    sp = np.asarray(spacing, float)
    start, end = np.asarray(start, int), np.asarray(end, int)
    dist = float(np.linalg.norm((end - start) * sp))
    margin = np.ceil(max(2 * d_max, 0.25 * dist) / sp).astype(int) + 2
    lo = np.maximum(np.minimum(start, end) - margin, 0)
    hi = np.minimum(np.maximum(start, end) + margin + 1, volume.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    enh = enhance(np.asarray(volume[box]), sp, d_min, d_max)
    cost = 1.0 / (enh + 0.02)  # 50x cheaper along bright filaments than background
    mcp = MCP_Geometric(cost, sampling=tuple(sp), fully_connected=True)
    mcp.find_costs([tuple(start - lo)], [tuple(end - lo)])
    path = np.asarray(mcp.traceback(tuple(end - lo))) + lo
    mask, _ = segment(enh, threshold, sp, d_max)
    return path, lo, mask


# ---- export ---------------------------------------------------------------------------


def export(result, base):
    """Write <base>_filaments.csv, _segments.csv, _points.csv, _sholl.csv and <base>.swc.

    Returns the paths written."""
    out = []
    for name, table in (
        ("filaments", result.filaments),
        ("segments", result.segments),
        ("points", result.points),
        ("sholl", result.sholl),
    ):
        path = f"{base}_{name}.csv"
        table.drop(columns=[c for c in table.columns if c.startswith("_")]).to_csv(path, index=False)
        out.append(path)
    path = f"{base}.swc"
    write_swc(result, path)
    out.append(path)
    return out


def write_swc(result, path):
    """Neuron-morphology SWC: one tree per filament, from its root (type 1, soma) along
    every skeleton voxel (type 3, dendrite). Coordinates and radii in µm."""
    from scipy.sparse.csgraph import breadth_first_order

    lines = ["# ibif-napari filament tracer; units: um", "# id type x y z radius parent"]
    if result.coordinates is not None and len(result.roots):
        sp, origin = np.asarray(result.spacing), np.asarray(result.origin)
        xyz = result.coordinates * sp + origin
        if xyz.shape[1] == 2:
            xyz = np.column_stack([np.zeros(len(xyz)), xyz])
        radius = np.nan_to_num(result.radius, nan=float(sp[-2:].min()) / 2)
        next_id = 1
        for fil, root in sorted(result.roots.items()):
            order, pred = breadth_first_order(result.graph, root, directed=False)
            ids = {}
            for v in order:
                ids[v] = next_id
                parent = ids.get(pred[v], -1) if v != root else -1
                z, y, x = xyz[v]
                kind = 1 if v == root else 3
                lines.append(f"{next_id} {kind} {x:.4f} {y:.4f} {z:.4f} {radius[v]:.4f} {parent}")
                next_id += 1
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
