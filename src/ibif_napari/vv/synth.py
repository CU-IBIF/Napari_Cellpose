"""Synthetic microscopy scenes with known truth, for verification and validation.

A scene is a 2D or 3D field of cells. Each cell has an ellipsoidal nucleus (random size,
elongation and orientation, in µm) inside a larger ellipsoidal cell body; neighbouring
cells meet along a Voronoi-like border. From these shapes the generator makes:

* truth labels: ``nuclei`` and ``cells`` (cell i contains nucleus i, same number);
* noise-free "phantom" channels (constant intensity per object):
  ``nuclei`` (stained nuclei), ``cytoplasm`` (cell body minus nucleus) and
  ``membrane`` (cell borders);
* "ideal" channels: the phantom blurred by a Gaussian point spread function (PSF);
* "acquired" channels: ideal x peak + background + Gaussian noise at a set
  signal-to-noise ratio (SNR), stored as uint16 like a camera image;
* ``truth.csv``: per object, the shape parameters, analytic volume, voxel counts,
  centroid (µm), and the expected intensity statistics of every channel.

The truth is computed here with plain NumPy, independently of the plugin's own code
(which is what gets tested). Everything is reproducible from the scene's seed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

CHANNELS = ("nuclei", "cytoplasm", "membrane")


@dataclass
class SceneSpec:
    """Parameters of one synthetic scene (lengths in µm)."""

    name: str
    shape: tuple  # (Y, X) or (Z, Y, X) pixels
    spacing: tuple  # µm per pixel, same order
    n_objects: int = 20
    nucleus_radius_um: tuple = (3.0, 4.5)  # range of the largest semi-axis
    elongation: tuple = (1.0, 1.4)  # largest / smallest semi-axis
    cell_scale: float = 1.9  # cell body semi-axes / nucleus semi-axes
    min_gap_um: float = 1.0  # smallest gap between nuclei
    psf_sigma_um: tuple = (0.4, 0.4)  # Gaussian PSF sigma per axis (µm)
    snr: float = 12.0  # mean nuclear signal / noise sigma
    background: float = 100.0  # camera offset (counts)
    peak: float = 1000.0  # counts of a nucleus with intensity 1
    nucleus_intensity: tuple = (0.6, 1.0)
    cytoplasm_intensity: tuple = (0.3, 0.5)
    membrane_intensity: float = 0.8
    timepoints: int = 1
    intensity_drift: float = 0.0  # per time point: intensities x (1 + drift * t)
    channels: tuple = ("nuclei",)
    seed: int = 0
    notes: str = ""
    origin_um: tuple = field(default=None)  # world position of pixel 0 (default 0)

    @property
    def ndim(self):
        return len(self.shape)

    def to_json(self):
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = tuple(v)
        return cls(**d)


# ---- standard scenes ------------------------------------------------------------------------


def standard_scenes(quick=False):
    """The scenes the V&V suite uses. `quick` makes smaller ones (for CI and tests)."""
    n2, n3, z3 = (128, 64, 12) if quick else (256, 112, 20)

    def k(full, small):
        return small if quick else full

    scenes = [
        SceneSpec("nuclei_2d", (n2, n2), (0.5, 0.5), n_objects=k(24, 7), seed=11,
                  notes="2D nuclei, moderate noise (SNR 12)"),
        SceneSpec("nuclei_3d", (z3, n3, n3), (1.0, 0.5, 0.5), n_objects=k(14, 4),
                  psf_sigma_um=(0.9, 0.4, 0.4), nucleus_radius_um=(3.0, 4.0), seed=12,
                  notes="3D nuclei, anisotropic voxels (Z step twice the pixel size)"),
        SceneSpec("cells_2d", (n2, n2), (0.5, 0.5), n_objects=k(14, 4), cell_scale=2.2,
                  min_gap_um=4.0, channels=("nuclei", "cytoplasm", "membrane"), seed=13,
                  notes="2D cells: nuclear, cytoplasm and membrane channels"),
        SceneSpec("cells_3d", (z3, n3, n3), (1.0, 0.5, 0.5), n_objects=k(9, 4), cell_scale=2.0,
                  min_gap_um=3.5, channels=("nuclei", "cytoplasm", "membrane"),
                  psf_sigma_um=(0.9, 0.4, 0.4), nucleus_radius_um=(2.5, 3.5), seed=14,
                  notes="3D cells for secondary objects and measurements"),
        SceneSpec("timelapse_3d", (8, 48, 48), (1.0, 0.5, 0.5), n_objects=4, timepoints=3,
                  intensity_drift=0.25, channels=("nuclei", "cytoplasm"),
                  psf_sigma_um=(0.9, 0.4, 0.4), nucleus_radius_um=(2.5, 3.0), cell_scale=1.8,
                  min_gap_um=3.0, seed=15, origin_um=(5.0, -3.0, 10.0),
                  notes="3 time points, intensities rise 25% per step, offset origin"),
    ]
    # Touching nuclei (they press into each other): connected components merge them, the
    # watershed of the Threshold + watershed tab must split them.
    scenes += [
        SceneSpec("touching_2d", k((256, 256), (128, 128)), (0.5, 0.5), n_objects=k(50, 12),
                  min_gap_um=k(-3.0, -3.5), elongation=(1.0, 1.2), seed=k(33, 36),
                  notes="2D nuclei packed so that some touch"),
        SceneSpec("touching_3d", (k(20, 12), k(112, 64), k(112, 64)), (1.0, 0.5, 0.5), n_objects=k(16, 6),
                  min_gap_um=-2.0, elongation=(1.0, 1.2), psf_sigma_um=(0.9, 0.4, 0.4),
                  nucleus_radius_um=(3.0, 4.0), seed=32,
                  notes="3D nuclei packed so that some touch (anisotropic voxels)"),
    ]
    for snr in (20.0, 8.0, 4.0, 2.0):
        scenes.append(SceneSpec(f"snr_{snr:g}", (n2, n2), (0.5, 0.5), n_objects=k(20, 6), snr=snr,
                                seed=21, notes=f"2D nuclei at SNR {snr:g} (robustness sweep)"))
    return scenes


# ---- generation -------------------------------------------------------------------------


def _rotation(rng, ndim):
    if ndim == 2:
        a = rng.uniform(0, np.pi)
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _place(spec, rng):
    """Centres (µm), semi-axes (µm) and rotations of the nuclei, without overlaps."""
    ndim = spec.ndim
    extent = np.asarray(spec.shape, float) * np.asarray(spec.spacing, float)
    objs = []
    tries = 0
    while len(objs) < spec.n_objects and tries < 20000:
        tries += 1
        rmax = rng.uniform(*spec.nucleus_radius_um)
        ratio = rng.uniform(*spec.elongation)
        axes = np.full(ndim, rmax)
        axes[1:] = rmax / ratio if ndim == 2 else [rmax / rng.uniform(1, ratio), rmax / ratio]
        axes = np.sort(axes)[::-1]
        # Keep whole nuclei inside the image (with a margin for the blur); in 3D the
        # first (Z) extent is small, so cap the nucleus's Z reach instead.
        margin = rmax + 1.5
        lo, hi = np.full(ndim, margin), extent - margin
        if ndim == 3:
            lo[0], hi[0] = rmax * 0.8 + 0.5, extent[0] - rmax * 0.8 - 0.5
        if np.any(hi <= lo):
            raise ValueError(f"{spec.name}: image too small for the nuclei")
        centre = rng.uniform(lo, hi)
        rot = _rotation(rng, ndim)
        if ndim == 3:
            # Keep the nucleus within the stack: shrink the Z reach if needed.
            reach_z = np.sqrt(np.sum((rot[0] * axes) ** 2))
            room = min(centre[0], extent[0] - centre[0]) - 0.5
            if reach_z > room:
                axes = axes * (room / reach_z)
                if axes.min() < spec.nucleus_radius_um[0] * 0.5:
                    continue
        if any(np.linalg.norm(centre - o[0]) < axes.max() + o[1].max() + spec.min_gap_um for o in objs):
            continue
        objs.append((centre, axes, rot))
    if len(objs) < spec.n_objects:
        raise ValueError(f"{spec.name}: could only place {len(objs)} of {spec.n_objects} nuclei")
    return objs


def _labels(spec, objs):
    """Truth nuclei and cells on the pixel grid (pixel centres)."""
    sp = np.asarray(spec.spacing, float)
    grid = np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(spec.shape, sp)], indexing="ij"), -1)
    best = np.full(spec.shape, np.inf)  # normalised distance to the nearest cell body
    owner = np.zeros(spec.shape, np.int32)
    q_nuc = np.full(spec.shape, np.inf)
    for i, (c, a, rot) in enumerate(objs, start=1):
        reach = a.max() * spec.cell_scale + 2 * sp.max()
        lo = np.maximum(np.floor((c - reach) / sp).astype(int), 0)
        hi = np.minimum(np.ceil((c + reach) / sp).astype(int) + 1, spec.shape)
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        local = (grid[sl] - c) @ rot / a  # coordinates in the nucleus's own frame
        q = np.sqrt(np.sum(local ** 2, axis=-1))
        qc = q / spec.cell_scale
        closer = qc < best[sl]
        best[sl] = np.where(closer, qc, best[sl])
        owner[sl] = np.where(closer, i, owner[sl])
        q_nuc[sl] = np.where(closer, q, q_nuc[sl])
    cells = np.where(best <= 1.0, owner, 0).astype(np.uint16)
    nuclei = np.where((q_nuc <= 1.0) & (cells > 0), owner, 0).astype(np.uint16)
    return nuclei, cells


def _borders(cells):
    """Pixels of each cell that touch background or another cell (inner border)."""
    border = np.zeros(cells.shape, bool)
    for ax in range(cells.ndim):
        for shift in (1, -1):
            neighbour = np.roll(cells, shift, axis=ax)
            edge = [slice(None)] * cells.ndim
            edge[ax] = 0 if shift == 1 else -1
            neighbour[tuple(edge)] = 0
            border |= (cells > 0) & (neighbour != cells)
    return border


def _phantoms(spec, nuclei, cells, rng, n):
    """Noise-free channels and the per-object intensities used."""
    nuc_i = rng.uniform(*spec.nucleus_intensity, n + 1)
    cyto_i = rng.uniform(*spec.cytoplasm_intensity, n + 1)
    nuc_i[0] = cyto_i[0] = 0.0
    out = {}
    if "nuclei" in spec.channels:
        out["nuclei"] = nuc_i[nuclei]
    if "cytoplasm" in spec.channels:
        out["cytoplasm"] = np.where(nuclei > 0, 0.0, cyto_i[cells])
    if "membrane" in spec.channels:
        out["membrane"] = np.where(_borders(cells), spec.membrane_intensity, 0.0)
    return out, nuc_i, cyto_i


def _blur(img, spec):
    sigma = [s / p for s, p in zip(spec.psf_sigma_um, spec.spacing)]
    return ndi.gaussian_filter(img.astype(np.float64), sigma=sigma, mode="constant")


def _stats(labels, image, ids):
    """Independent per-object statistics with np.bincount / sorting (not scipy.ndimage)."""
    flat_l = labels.ravel()
    flat_i = image.ravel().astype(np.float64)
    n = int(ids.max()) + 1 if len(ids) else 1
    count = np.bincount(flat_l, minlength=n)
    total = np.bincount(flat_l, weights=flat_i, minlength=n)
    sq = np.bincount(flat_l, weights=flat_i ** 2, minlength=n)
    mean = total / np.maximum(count, 1)
    var = np.maximum(sq / np.maximum(count, 1) - mean ** 2, 0)
    order = np.lexsort((flat_i, flat_l))
    sorted_l, sorted_i = flat_l[order], flat_i[order]
    starts = np.searchsorted(sorted_l, ids, side="left")
    ends = np.searchsorted(sorted_l, ids, side="right")
    return {
        "mean": mean[ids], "sum": total[ids], "std": np.sqrt(var[ids]),
        "min": sorted_i[starts], "max": sorted_i[ends - 1],
        "median": np.array([np.median(sorted_i[s:e]) for s, e in zip(starts, ends)]),
    }


@dataclass
class Scene:
    spec: SceneSpec
    nuclei: np.ndarray  # truth labels (spatial)
    cells: np.ndarray
    acquired: dict  # channel -> array, (T,) + spatial if timepoints > 1, uint16
    ideal: dict  # channel -> float array (same shape as acquired), counts without noise
    noise_sigma: float
    truth: pd.DataFrame  # one row per object (and time point)


def generate(spec):
    """Make a Scene from its spec (deterministic for a given seed)."""
    rng = np.random.default_rng(spec.seed)
    objs = _place(spec, rng)
    nuclei, cells = _labels(spec, objs)
    n = len(objs)
    phantoms, nuc_i, cyto_i = _phantoms(spec, nuclei, cells, rng, n)
    blurred = {k: _blur(v, spec) for k, v in phantoms.items()}
    mean_signal = spec.peak * float(np.mean(nuc_i[1:])) if n else spec.peak
    sigma = mean_signal / spec.snr
    acquired, ideal = {}, {}
    for name, b in blurred.items():
        frames_i, frames_a = [], []
        for t in range(spec.timepoints):
            ideal_t = spec.background + spec.peak * (1 + spec.intensity_drift * t) * b
            noisy = ideal_t + rng.normal(0, sigma, ideal_t.shape)
            frames_i.append(ideal_t)
            frames_a.append(np.clip(np.round(noisy), 0, 65535).astype(np.uint16))
        ideal[name] = np.stack(frames_i) if spec.timepoints > 1 else frames_i[0]
        acquired[name] = np.stack(frames_a) if spec.timepoints > 1 else frames_a[0]
    truth = _truth_table(spec, objs, nuclei, cells, ideal, acquired, nuc_i, cyto_i)
    return Scene(spec, nuclei, cells, acquired, ideal, sigma, truth)


def _truth_table(spec, objs, nuclei, cells, ideal, acquired, nuc_i, cyto_i):
    sp = np.asarray(spec.spacing, float)
    origin = np.asarray(spec.origin_um if spec.origin_um is not None else (0.0,) * spec.ndim, float)
    ids = np.arange(1, len(objs) + 1)
    axes_names = "ZYX"[-spec.ndim:]
    unit = "µm³" if spec.ndim == 3 else "µm²"
    rows = []
    for t in range(spec.timepoints):
        base = {"label": ids}
        if spec.timepoints > 1:
            base = {"T": np.full(len(ids), t), **base}
        table = pd.DataFrame(base)
        for name, lab in (("nucleus", nuclei), ("cell", cells)):
            count = np.bincount(lab.ravel(), minlength=len(ids) + 1)[ids]
            table[f"{name} voxels"] = count
            table[f"{name} size ({unit})"] = count * float(np.prod(sp))
            com = np.array(ndi.center_of_mass(np.ones(lab.shape), lab, ids)).reshape(len(ids), -1)
            # (center_of_mass on the truth only; checked against the analytic centre below)
            for k, ax in enumerate(axes_names):
                table[f"{name} centroid {ax} (µm)"] = com[:, k] * sp[k] + origin[k]
        for k, ax in enumerate(axes_names):
            table[f"centre {ax} (µm)"] = [o[0][k] + origin[k] for o in objs]
        for k in range(spec.ndim):
            table[f"semi-axis {k} (µm)"] = [o[1][k] for o in objs]
        a = np.array([o[1] for o in objs])
        table[f"nucleus analytic size ({unit})"] = (
            4 / 3 * np.pi * a.prod(axis=1) if spec.ndim == 3 else np.pi * a.prod(axis=1)
        )
        table["nucleus intensity"] = nuc_i[ids]
        table["cytoplasm intensity"] = cyto_i[ids]
        scale = 1 + spec.intensity_drift * t
        table["intensity scale"] = scale
        for ch in ideal:
            img_i = ideal[ch][t] if spec.timepoints > 1 else ideal[ch]
            img_a = acquired[ch][t] if spec.timepoints > 1 else acquired[ch]
            for name, lab in (("nucleus", nuclei), ("cell", cells)):
                st = _stats(lab, img_i, ids)
                table[f"{name} {ch} ideal mean"] = st["mean"]
                st = _stats(lab, img_a, ids)
                for k, v in st.items():
                    table[f"{name} {ch} acquired {k}"] = v
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


# ---- files --------------------------------------------------------------------------------


def save(scene, folder):
    """Write a scene: scene.json, <channel>.tif, nuclei_truth.tif, cells_truth.tif,
    ideal_<channel>.tif and truth.csv."""
    import tifffile

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    spec = scene.spec
    meta = {"spec": spec.to_json(), "noise_sigma": scene.noise_sigma,
            "axes": ("T" if spec.timepoints > 1 else "") + "ZYX"[-spec.ndim:],
            "channels": list(scene.acquired)}
    (folder / "scene.json").write_text(json.dumps(meta, indent=2))
    for name, arr in scene.acquired.items():
        tifffile.imwrite(folder / f"{name}.tif", arr)
        tifffile.imwrite(folder / f"ideal_{name}.tif", scene.ideal[name].astype(np.float32))
    tifffile.imwrite(folder / "nuclei_truth.tif", scene.nuclei)
    tifffile.imwrite(folder / "cells_truth.tif", scene.cells)
    scene.truth.to_csv(folder / "truth.csv", index=False)
    return folder


def load(folder):
    """Read a scene written by `save`. (The ideal images are kept as float32.)"""
    import tifffile

    folder = Path(folder)
    meta = json.loads((folder / "scene.json").read_text())
    spec = SceneSpec.from_json(meta["spec"])
    acquired = {c: tifffile.imread(folder / f"{c}.tif") for c in meta["channels"]}
    ideal = {c: tifffile.imread(folder / f"ideal_{c}.tif").astype(np.float64) for c in meta["channels"]}
    return Scene(spec, tifffile.imread(folder / "nuclei_truth.tif"), tifffile.imread(folder / "cells_truth.tif"),
                 acquired, ideal, float(meta["noise_sigma"]), pd.read_csv(folder / "truth.csv"))


def generate_all(out, quick=False, specs=None):
    """Generate and save the standard scenes into `out`; returns their folders."""
    out = Path(out)
    folders = []
    for spec in specs or standard_scenes(quick):
        folders.append(save(generate(spec), out / spec.name))
    manifest = {"quick": quick, "scenes": [f.name for f in folders]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return folders


# ---- inclusions (spots inside cells) ---------------------------------------------------------


@dataclass
class InclusionScene:
    """Spots inside cells, some of them touching, with their truth.

    cells: parent objects (1...n); spots: truth inclusions numbered one by one; parent:
    {spot: cell}; image: blurred spots on a dim cytoplasm, with Gaussian noise;
    touching: number of touching pairs; spacing (µm) and radius (µm) of the spots."""

    cells: np.ndarray
    spots: np.ndarray
    parent: dict
    image: np.ndarray
    spacing: tuple
    radius: float
    touching: int


def inclusion_scene(ndim=2, seed=0, snr=8.0, grid=3, touch_share=0.5):
    """A grid of `grid` x `grid` square cells; each holds 1-3 round spots (radius 0.4-0.5 µm)
    placed well apart, and about `touch_share` of the cells also a pair of spots pressed
    together (centres 1.6-1.85 radii apart, so they touch along a neck). 2D at 0.1 µm
    pixels, or 3D at 0.3 x 0.1 x 0.1 µm (Z steps 3x the pixel size), each spot (pair) at
    its own depth. Overlapping voxels go to the nearer spot centre. The image is the spots
    (1.0) on cytoplasm (0.15), blurred by a 0.05 µm PSF, with noise of sigma 1 / `snr`."""
    rng = np.random.default_rng(seed)
    spacing = (0.1, 0.1) if ndim == 2 else (0.3, 0.1, 0.1)
    side, gap = 70, 6  # cell size and gap, pixels (XY)
    n_xy = grid * (side + gap) + gap
    shape = (n_xy, n_xy) if ndim == 2 else (16, n_xy, n_xy)
    cells = np.zeros(shape, np.uint16)
    corners = {}
    for k, (i, j) in enumerate(np.ndindex(grid, grid), start=1):
        y0, x0 = gap + i * (side + gap), gap + j * (side + gap)
        cells[..., y0:y0 + side, x0:x0 + side] = k
        corners[k] = (y0, x0)
    depth = (shape[0] - 1) * spacing[0] if ndim == 3 else 0.0

    def z_of():  # a depth at least 0.6 µm inside the stack
        return float(rng.uniform(0.6, depth - 0.6)) if ndim == 3 else None

    placed = []  # (centre (µm, Z Y X or Y X), radius, cell)
    touching = 0
    for k, (y0, x0) in corners.items():
        lo = np.array([y0, x0]) * 0.1 + 0.8
        hi = np.array([y0 + side - 1, x0 + side - 1]) * 0.1 - 0.8
        mine = []

        def free(c, mine=mine):
            return all(np.hypot(*(c - m)) > 1.5 for m in mine)

        if rng.random() < touch_share:
            r1, r2 = rng.uniform(0.4, 0.5, 2)
            for _ in range(100):
                c1 = rng.uniform(lo + 0.6, hi - 0.6)
                angle = rng.uniform(0, 2 * np.pi)
                c2 = c1 + rng.uniform(1.6, 1.85) * max(r1, r2) * np.array([np.sin(angle), np.cos(angle)])
                if np.all((c2 >= lo) & (c2 <= hi)) and free(c1) and free(c2):
                    z = z_of()
                    for c, r in ((c1, r1), (c2, r2)):
                        mine.append(c)
                        placed.append((np.r_[z, c] if ndim == 3 else c, r, k))
                    touching += 1
                    break
        for _ in range(int(rng.integers(1, 4))):
            r = rng.uniform(0.4, 0.5)
            for _ in range(100):
                c = rng.uniform(lo, hi)
                if free(c):
                    mine.append(c)
                    placed.append((np.r_[z_of(), c] if ndim == 3 else c, r, k))
                    break
    mesh = np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing)], indexing="ij", sparse=True)
    best = np.full(shape, np.inf)
    spots = np.zeros(shape, np.uint32)
    for n, (c, r, _) in enumerate(placed, start=1):
        d = np.sqrt(sum((m - ci) ** 2 for m, ci in zip(mesh, c))) / r  # in radii
        take = (d <= 1.0) & (d < best)
        spots[take] = n
        best = np.where(take, d, best)
    phantom = np.where(cells > 0, 0.15, 0.0) + (spots > 0) * 0.85
    image = ndi.gaussian_filter(phantom, sigma=[0.05 / s for s in spacing])
    image = (image + rng.normal(0, 1.0 / snr, shape)).astype(np.float32)
    return InclusionScene(cells, spots, {n: k for n, (_, _, k) in enumerate(placed, start=1)}, image,
                          spacing, 0.45, touching)


# ---- movement (stack registration) ----------------------------------------------------------


@dataclass
class MotionScene:
    """Frames of a textured sample that moves (and warps) by a known amount.

    frames: (N, Y, X) or (N, Z, Y, X) along T, or a (Z, Y, X) stack whose planes drift
    along Z; spacing (µm per pixel of a frame); truth: per frame, the rotation (degrees,
    YX), shift (pixels, ZYX / YX) and warp amplitude (pixels). `source(k, coords)` gives,
    for pixel coordinates of frame k (shape (n, ...)), where they come from in the sample
    (the motion-free canvas): registration is perfect when the registered frame k reads,
    at every pixel, what the reference frame reads there."""

    frames: np.ndarray
    spacing: tuple
    truth: list
    offset: np.ndarray  # where pixel 0 of a frame is in the sample, per axis
    centre: np.ndarray
    along: str
    ideal: np.ndarray

    def source(self, k, coords):
        coords = np.asarray(coords, float)
        t = self.truth[k]
        n = coords.shape[0]
        q = coords + self.offset.reshape((n,) + (1,) * (coords.ndim - 1))
        shift = np.asarray(t["shift"], float).reshape((n,) + (1,) * (coords.ndim - 1))
        c = self.centre.reshape((n,) + (1,) * (coords.ndim - 1))
        d = q - c - shift
        a = np.deg2rad(-t["angle"])
        y, x = d[-2], d[-1]
        ry, rx = np.cos(a) * y - np.sin(a) * x, np.sin(a) * y + np.cos(a) * x
        out = np.concatenate([d[:-2], ry[None], rx[None]]) + c
        if t["warp"]:
            out = out - _warp_field(q, t["warp"], t["phase"])
        return out


def _warp_field(q, amplitude, phase):
    """A smooth displacement (pixels) like tissue moving with breathing: a slow wave across
    the field, larger in the middle."""
    y, x = q[-2], q[-1]
    dy = amplitude * np.sin(2 * np.pi * x / 160.0 + phase) * np.exp(-((y - 100) / 120.0) ** 2)
    dx = 0.6 * amplitude * np.cos(2 * np.pi * y / 140.0 + phase)
    zeros = [np.zeros_like(y)] * (q.shape[0] - 2)
    return np.stack(zeros + [dy, dx])


def _sample(seed, shape, spacing):
    """A motion-free sample: cell-like blobs on a textured background, with structure at
    several scales so every frame can be registered."""
    rng = np.random.default_rng(seed)
    background = ndi.gaussian_filter(rng.random(shape), [4.0 / s * spacing[-1] for s in spacing])
    background = (background - background.min()) / (np.ptp(background) or 1)
    blobs = np.zeros(shape)
    n = int(np.prod(shape) / 900)
    for _ in range(n):
        blobs[tuple(rng.integers(0, s) for s in shape)] = rng.uniform(1.0, 3.0)
    blobs = ndi.gaussian_filter(blobs, [2.5 * spacing[-1] / s for s in spacing])
    return 0.3 * background + blobs / (blobs.max() or 1)


def motion_scene(ndim=2, along="T", n=8, seed=0, snr=8.0, max_shift=15.0, max_angle=0.0, warp=0.0,
                 size=160, depth=12):
    """Frames with known movement. Along T: `n` frames (2D, or 3D Z stacks of `depth`
    planes at 2 x 0.5 x 0.5 µm with a Z shift too), each moved by a random shift up to
    `max_shift` pixels (in Z up to 1.5 planes), rotated up to `max_angle` degrees in YX,
    and warped smoothly by up to `warp` pixels; the first frame does not move. Along Z: a
    3D stack whose planes drift in YX like a random walk (steps up to `max_shift` / 4,
    plane 0 fixed). Gaussian noise gives the signal-to-noise ratio `snr` (mean signal /
    sigma). The frames are read from a larger motion-free sample with cubic interpolation,
    so the movement comes in from outside the frame like a real field of view."""
    rng = np.random.default_rng(seed)
    margin = int(np.ceil(max_shift + abs(max_angle) / 45 * size + 2 * warp)) + 8
    if along == "Z":
        spacing = (1.0, 0.5, 0.5)
        shape = (depth, size + 2 * margin, size + 2 * margin)
    elif ndim == 3:
        spacing = (2.0, 0.5, 0.5)
        shape = (depth + 2 * 4, size + 2 * margin, size + 2 * margin)
    else:
        spacing = (0.5, 0.5)
        shape = (size + 2 * margin,) * 2
    sample = _sample(seed, shape, spacing)
    centre = (np.asarray(shape, float) - 1) / 2
    truth = []
    if along == "Z":
        walk = np.cumsum(rng.uniform(-max_shift / 4, max_shift / 4, (depth, 2)), axis=0)
        walk -= walk[0]
        scene = MotionScene(None, spacing, [], np.array([margin, margin], float), centre[1:], "Z", sample)
        planes = []
        for z in range(depth):
            t = {"angle": 0.0, "shift": tuple(walk[z]), "warp": 0.0, "phase": 0.0}
            scene.truth.append(t)
            coords = np.indices((size, size)).astype(float)
            src = scene.source(z, coords)
            planes.append(ndi.map_coordinates(sample[z], src, order=3, mode="nearest"))
        frames = np.stack(planes)
    else:
        nd = len(shape)
        out_shape = (depth,) * (nd == 3) + (size, size)
        offset = np.array(([4] if nd == 3 else []) + [margin, margin], float)
        scene = MotionScene(None, spacing, truth, offset, centre, "T", sample)
        frames = []
        for k in range(n):
            if k == 0:
                shift, angle, amp, phase = np.zeros(nd), 0.0, 0.0, 0.0
            else:
                shift = rng.uniform(-max_shift, max_shift, nd)
                if nd == 3:
                    shift[0] = rng.uniform(-1.5, 1.5)
                angle = float(rng.uniform(-max_angle, max_angle)) if max_angle else 0.0
                amp = float(rng.uniform(0.5, 1.0) * warp) if warp else 0.0
                phase = float(rng.uniform(0, 2 * np.pi))
            truth.append({"angle": angle, "shift": tuple(float(s) for s in shift), "warp": amp, "phase": phase})
            coords = np.indices(out_shape).astype(float)
            frames.append(ndi.map_coordinates(sample, scene.source(k, coords), order=3, mode="nearest"))
        frames = np.stack(frames)
    signal = float(np.mean(frames))
    frames = frames + rng.normal(0, signal / snr, frames.shape)
    scene.frames = frames.astype(np.float32)
    return scene
